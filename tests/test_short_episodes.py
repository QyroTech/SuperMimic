"""Short episodes retain real model prefixes and reset-safe policy histories."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
import torch
from test_reward_training import environment as environment
from test_reward_training import training_config
from test_runtime_training import assert_state_equal, make_batch

from supermimic.algorithm import SuperMimic
from supermimic.collection import Q1Collector
from supermimic.data import Q1Motions
from supermimic.models import Critic, MotionPolicy, WorldModel
from supermimic.normalization import NetworkProfile, RunningNorm
from supermimic.replay import TemporalReplay
from supermimic.robot import Q1Robot
from supermimic.train import train


def numbered_replay(capacity: int, steps: int, length: int) -> TemporalReplay:
  replay = TemporalReplay(capacity, 2, 79, 26, 72, "cpu")
  for step in range(steps):
    values = torch.tensor([float(step), float(step + 1000)])
    ends = torch.full((2,), (step + 1) % length == 0, dtype=torch.bool)
    replay.add(
      values[:, None].expand(-1, 79),
      (values + 0.5)[:, None].expand(-1, 79),
      values[:, None].expand(-1, 22),
      values[:, None].expand(-1, 26),
      values[:, None].expand(-1, 72),
      ends.float(),
      ends,
      torch.tensor([0, 1]),
      torch.full((2,), step, dtype=torch.long),
    )
  return replay


@pytest.mark.parametrize("length", [1, 3, 8, 9, 10, 12])
@pytest.mark.parametrize("wrapped", [False, True])
def test_history_padding_and_model_prefixes(length: int, wrapped: bool):
  capacity, steps = (13, 29) if wrapped else (32, 24)
  replay = numbered_replay(capacity, steps, length)
  oldest = max(0, steps - capacity)
  torch.manual_seed(83)
  for window in (1, 8):
    batch = replay.sample(512, window=window, history=10)
    assert batch.valid is not None
    assert batch.history.shape == (512, 10, 72)
    assert batch.states.shape == (512, window, 79)
    assert batch.valid[:, 0].all()
    # Validate every frame against its exact chronological episode identity.
    first = (batch.frames // length * length).clamp_min(oldest)
    expected_history = (
      torch.maximum(batch.frames[:, None] + torch.arange(-9, 1), first[:, None])
      + 1000 * batch.clips[:, None]
    )
    torch.testing.assert_close(batch.history[:, :, 0], expected_history.float())
    last = torch.minimum(
      batch.frames // length * length + length - 1,
      torch.full_like(batch.frames, steps - 1),
    )
    expected_positions = torch.minimum(
      batch.frames[:, None] + torch.arange(window), last[:, None]
    )
    assert torch.equal(
      batch.valid, batch.frames[:, None] + torch.arange(window) <= last[:, None]
    )
    expected_states = expected_positions + 1000 * batch.clips[:, None]
    torch.testing.assert_close(batch.states[:, :, 0], expected_states.float())
    torch.testing.assert_close(batch.next_states[:, :, 0], batch.states[:, :, 0] + 0.5)
    torch.testing.assert_close(batch.context[:, 0], batch.history[:, -1, 0])
    expected_terminal = (expected_positions + 1) % length == 0
    assert torch.equal(batch.terminated.bool(), expected_terminal)
    if not wrapped:
      assert {0, 1, 8}.issubset(set(batch.frames.tolist()))


def test_replay_smaller_than_history_and_window_is_usable():
  replay = numbered_replay(2, 1, 8)
  batch = replay.sample(4, window=8, history=10)
  assert batch.valid is not None
  assert batch.valid[:, 0].all() and not batch.valid[:, 1:].any()
  assert (batch.history == batch.history[:, :1]).all()
  assert (batch.states == batch.states[:, :1]).all()


@pytest.mark.parametrize("args", [(0, 8, 10), (1, 0, 10), (1, 8, 0)])
def test_replay_rejects_nonpositive_sizes(args: tuple[int, int, int]):
  with pytest.raises(ValueError, match="Positive"):
    numbered_replay(2, 1, 8).sample(*args)


def test_resume_history_does_not_join_previous_episode():
  old = numbered_replay(16, 12, 100)
  restored = TemporalReplay(16, 2, 79, 26, 72, "cpu")
  restored.load_state_dict(old.state_dict())
  new = numbered_replay(16, 2, 100).state_dict()
  for step in range(2):
    restored.add(
      new["states"][step],
      new["next_states"][step],
      new["actions"][step],
      new["context"][step],
      new["observations"][step],
      new["terminated"][step],
      new["boundaries"][step],
      new["clips"][step],
      new["frames"][step],
    )
  batch = restored.sample(512, window=8, history=10)
  fresh = batch.frames < 2
  assert fresh.any()
  expected = (batch.frames[fresh, None] + torch.arange(-9, 1)).clamp_min(
    0
  ) + 1000 * batch.clips[fresh, None]
  torch.testing.assert_close(batch.history[fresh, :, 0], expected.float())
  assert batch.valid is not None
  old_rows = ~fresh & (12 - batch.frames < 8)
  assert not batch.valid[old_rows, 12 - batch.frames[old_rows]].any()


def force_short_episodes(monkeypatch: pytest.MonkeyPatch, length: int) -> list[int]:
  calls = [0]
  heads = Q1Collector._heads

  def short_heads(self: Q1Collector, *args):
    reward, terminated, observation = heads(self, *args)
    calls[0] += 1
    terminated = torch.full_like(terminated, calls[0] % length == 0)
    return reward, terminated, observation

  monkeypatch.setattr(Q1Collector, "_heads", short_heads)
  return calls


@pytest.mark.parametrize("length", [3, 8, 9])
def test_cpu_warmup_updates_model_without_sampling_policy(
  environment: tuple[Q1Robot, Q1Motions],
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  length: int,
):
  robot, motions = environment
  collection = force_short_episodes(monkeypatch, length)
  sample = TemporalReplay.sample
  fit = SuperMimic.fit_model_tensors
  updates = []
  policy_samples = []

  def guarded_sample(self: TemporalReplay, batch_size: int, window=8, history=10):
    if window == 1:
      policy_samples.append(1)
      raise ValueError("Deliberately unavailable policy replay during warmup")
    return sample(self, batch_size, window, history)

  def checked_fit(self: SuperMimic, batch):
    before = [p.detach().clone() for p in self.model.parameters()]
    result = fit(self, batch)
    assert any(
      not torch.equal(a, b)
      for a, b in zip(before, self.model.parameters(), strict=True)
    )
    assert all(torch.isfinite(value) for value in result.values())
    updates.append(result)
    return result

  monkeypatch.setattr(TemporalReplay, "sample", guarded_sample)
  monkeypatch.setattr(SuperMimic, "fit_model_tensors", checked_fit)
  cfg = replace(
    training_config(robot, motions, tmp_path / "warmup"),
    iterations=2,
    model_warmup_iterations=2,
    network_profile="running_silu",
    dynamics_representation="heading",
  )
  saved = torch.load(train(cfg), weights_only=True)
  assert len(updates) == 2 and not policy_samples
  assert collection == [45]  # 40 initial rows, then 2 + 3; no readiness retries.
  assert all(int(s["step"]) == 2 for s in saved["model_optimizer"]["state"].values())
  assert not saved["actor_optimizer"]["state"]
  assert not saved["critic_optimizer"]["state"]


def test_cpu_actor_starts_after_short_episode_warmup(
  environment: tuple[Q1Robot, Q1Motions],
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
):
  robot, motions = environment
  force_short_episodes(monkeypatch, 3)
  cfg = replace(
    training_config(robot, motions, tmp_path / "actor"),
    iterations=3,
    model_warmup_iterations=2,
    network_profile="running_silu",
    dynamics_representation="heading",
    horizon=3,
  )
  saved = torch.load(train(cfg), weights_only=True)
  for name in ("actor", "critic"):
    assert saved[f"{name}_optimizer"]["state"]
    assert all(
      int(s["step"]) == 1 for s in saved[f"{name}_optimizer"]["state"].values()
    )
  metrics = [
    json.loads(line)
    for line in (cfg.output_dir / "metrics.jsonl").read_text().splitlines()
  ]
  assert [row["actor_update"] for row in metrics] == [0, 0, 1]


@pytest.mark.parametrize("representation", ["legacy", "heading"])
@pytest.mark.parametrize("fused", [False, True])
def test_masked_suffix_cannot_affect_model_loss_gradients_or_statistics(
  environment: tuple[Q1Robot, Q1Motions],
  representation: Literal["legacy", "heading"],
  fused: bool,
):
  robot, motions = environment
  batch = make_batch(robot, motions, 4)
  batch.valid = torch.tensor(
    [[True, False, False, False], [True, True, False, False], [True] * 4]
  )
  batch.terminated[0, 0] = batch.terminated[1, 1] = 1
  changed = replace(
    batch,
    states=batch.states.clone(),
    next_states=batch.next_states.clone(),
    actions=batch.actions.clone(),
    terminated=batch.terminated.clone(),
  )
  for name in ("states", "next_states", "actions", "terminated"):
    getattr(changed, name)[~batch.valid] = float("nan")
  learners = []
  profile: NetworkProfile = "running_silu" if representation == "heading" else "legacy"
  for current in (batch, changed):
    torch.manual_seed(691)
    learner = SuperMimic(
      robot,
      motions,
      WorldModel(robot, (16,), representation=representation, network_profile=profile),
      MotionPolicy(robot, (16,), network_profile=profile),
      Critic(robot, (16,), network_profile=profile),
      fuse_model_loss=fused,
    )
    if fused:
      learner.enable_compile("aot_eager")
    learner.update_normalizers(current)
    if representation == "heading":
      assert isinstance(learner.model.input_norm, RunningNorm)
      assert isinstance(learner.model.output_norm, RunningNorm)
      assert learner.model.input_norm.count == learner.model.output_norm.count == 7
    result = learner.fit_model_tensors(current)
    assert all(torch.isfinite(value) for value in result.values())
    assert all(
      p.grad is None or torch.isfinite(p.grad).all() for p in learner.model.parameters()
    )
    learners.append((learner, result))
  first, second = learners
  assert_state_equal(first[1], second[1])
  assert_state_equal(first[0].model.state_dict(), second[0].model.state_dict())
  assert_state_equal(
    first[0].model_optimizer.state_dict(), second[0].model_optimizer.state_dict()
  )


def test_terminal_transition_has_loss_and_masked_suffix_has_zero_gradient(
  environment: tuple[Q1Robot, Q1Motions],
):
  robot, motions = environment
  learner = SuperMimic(
    robot,
    motions,
    WorldModel(robot, (16,)),
    MotionPolicy(robot, (16,)),
    Critic(robot, (16,)),
  )
  label = motions.load(0)[:2].clone()
  state = label.clone()
  state[:, 0] += 0.1
  state.requires_grad_()
  logits = torch.zeros(2, requires_grad=True)
  losses = learner._model_losses(
    state, logits, label, torch.ones(2), torch.tensor([True, False])
  )
  (losses[0] + losses[1]).backward()
  assert state.grad is not None and logits.grad is not None
  assert state.grad[0].abs().sum() > 0 and logits.grad[0] < 0
  assert not state.grad[1].any() and logits.grad[1] == 0
