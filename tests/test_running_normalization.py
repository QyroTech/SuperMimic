"""Calibration math, heading increments and compatible training state."""

from dataclasses import replace
from pathlib import Path

import pytest
import test_reward_training
import torch
from test_reward_training import training_config
from test_runtime_training import make_batch

from supermimic.algorithm import SuperMimic
from supermimic.data import Q1Motions
from supermimic.models import Critic, MotionPolicy, WorldModel, frozen
from supermimic.normalization import RunningNorm
from supermimic.robot import Q1Robot, normalize_quat
from supermimic.train import train

environment = test_reward_training.environment


def test_global_statistics_mask_freeze_and_gradient():
  values = torch.tensor([[1.0, -2.0], [3.0, 2.0], [5.0, 4.0]])
  other = torch.tensor([[-1.0, 1.0], [0.0, 3.0]])
  norm = RunningNorm(2, max_count=5)
  pointers = [tensor.data_ptr() for tensor in norm.buffers()]

  def combine(moments: torch.Tensor) -> torch.Tensor:
    return moments + torch.cat(
      (moments.new_tensor([len(other)]), other.sum(0), other.square().sum(0))
    )

  norm.update(values, combine)
  joint = torch.cat((values, other))
  torch.testing.assert_close(norm.mean, joint.mean(0))
  torch.testing.assert_close(norm.var, joint.var(0, unbiased=False))
  assert norm.count == 5
  before = {name: tensor.clone() for name, tensor in norm.state_dict().items()}
  norm.update(torch.full((2, 2), 99.0))
  norm.update(torch.full((1, 2), float("nan")), valid=torch.tensor([False]))
  x = torch.ones(2, 2, requires_grad=True)
  gradient = torch.autograd.grad(norm(x).sum(), x)[0]
  torch.testing.assert_close(gradient, norm.std().reciprocal().expand_as(x))
  assert pointers == [tensor.data_ptr() for tensor in norm.buffers()]
  assert all(torch.equal(before[name], t) for name, t in norm.state_dict().items())


def test_heading_output_calibration_inverts_existing_integrator(
  environment: tuple[Q1Robot, Q1Motions],
):
  robot, _ = environment
  model = WorldModel(
    robot, (16,), representation="heading", network_profile="running_silu"
  )
  torch.manual_seed(49)
  state = torch.randn(5, robot.state_dim) * 0.2
  state[:, 3:7] = normalize_quat(state[:, 3:7])
  action = torch.randn(5, 22) * 0.1
  increment = torch.randn(5, 56) * 0.05
  assert model.heading_state is not None
  successor = model.heading_state.integrate(state, increment, action)
  torch.testing.assert_close(
    model.output_target(state, successor), increment, atol=2e-5, rtol=1e-4
  )
  assert isinstance(model.output_norm, RunningNorm)
  model.output_norm.update(model.output_target(state, successor))
  encoded = model.output_norm(increment)
  torch.testing.assert_close(model.output_norm.denormalize(encoded), increment)


def test_compiled_normalized_model_keeps_analytic_inputs(
  environment: tuple[Q1Robot, Q1Motions],
):
  robot, motions = environment
  model = WorldModel(
    robot, (16,), representation="heading", network_profile="running_silu"
  )
  actor = MotionPolicy(robot, (16,), network_profile="running_silu")
  critic = Critic(robot, (16,), network_profile="running_silu")
  learner = SuperMimic(
    robot, motions, model, actor, critic, horizon=3, compile_critic=True
  )
  batch = make_batch(robot, motions, 3)
  learner.update_normalizers(batch)
  assert isinstance(model.input_norm, RunningNorm)
  assert isinstance(actor.obs_norm, RunningNorm)
  assert model.input_norm.count == 9
  assert actor.obs_norm.count == 30
  last = model.dynamics[-1]
  assert isinstance(last, torch.nn.Linear)
  torch.nn.init.normal_(last.weight, std=0.03)
  state = batch.states[:, 0].clone().requires_grad_()
  history = batch.history.clone().requires_grad_()
  refs = motions.sequence(batch.clips, batch.frames, torch.arange(1, 11))
  counts = model.input_norm.count.clone()
  with frozen(model, critic):
    expected = learner.imagined_step(state, history, refs, batch.context)
    before = torch.autograd.grad(
      expected[1].sum() + expected[5].sum(), (state, history)
    )
    learner.enable_compile("aot_eager")
    actual = learner.imagined_step(state, history, refs, batch.context)
    after = torch.autograd.grad(actual[1].sum() + actual[5].sum(), (state, history))
  for left, right in zip(actual, expected, strict=True):
    torch.testing.assert_close(left, right)
  for left, right in zip(after, before, strict=True):
    torch.testing.assert_close(left, right)
    assert torch.isfinite(left).all() and left.abs().sum() > 0
  assert torch.equal(model.input_norm.count, counts)
  assert all(p.grad is None for p in model.parameters())
  assert all(p.grad is None for p in critic.parameters())


def test_running_checkpoint_restores_statistics_and_rejects_profile_change(
  environment: tuple[Q1Robot, Q1Motions],
  tmp_path: Path,
):
  robot, motions = environment
  cfg = replace(
    training_config(robot, motions, tmp_path / "normalized"),
    network_profile="running_silu",
    dynamics_representation="heading",
    mlp_precision="bf16",
  )
  path = train(cfg)
  checkpoint = torch.load(path, weights_only=True)
  assert (
    checkpoint["identity"]["network_architecture"]["network_profile"] == "running_silu"
  )
  assert checkpoint["identity"]["mlp_precision_actual"] == "fp32"
  assert checkpoint["model"]["input_norm.count"] > 0
  assert checkpoint["model"]["output_norm.count"] > 0
  assert checkpoint["actor"]["obs_norm.count"] > 0
  assert checkpoint["actor"]["goal_norm.count"] > 0
  assert checkpoint["critic"]["input_norm.count"] > 0
  for name in ("model", "actor", "critic"):
    assert checkpoint[f"{name}_optimizer"]["state"]
  actor = MotionPolicy(
    robot,
    tuple(checkpoint["identity"]["actor_hidden"]),
    network_profile="running_silu",
  )
  actor.load_state_dict(checkpoint["actor"])
  assert isinstance(actor.obs_norm, RunningNorm)
  torch.testing.assert_close(
    actor.obs_norm.count, checkpoint["actor"]["obs_norm.count"]
  )
  with pytest.raises(ValueError, match="representation changed"):
    train(
      replace(
        cfg,
        resume=path,
        output_dir=tmp_path / "wrong_profile",
        iterations=3,
        network_profile="legacy",
      )
    )


def test_model_warmup_resumes_and_then_opens_actor(
  environment: tuple[Q1Robot, Q1Motions],
  tmp_path: Path,
):
  robot, motions = environment
  cfg = replace(
    training_config(robot, motions, tmp_path / "warmup"),
    network_profile="running_silu",
    dynamics_representation="heading",
    model_warmup_iterations=2,
    horizon=32,
  )
  path = train(cfg)
  saved = torch.load(path, weights_only=True)
  assert saved["config"]["horizon"] == 32
  assert saved["model_optimizer"]["state"]
  assert not saved["actor_optimizer"]["state"]
  assert not saved["critic_optimizer"]["state"]
  final = train(
    replace(cfg, resume=path, output_dir=tmp_path / "after_warmup", iterations=3)
  )
  restored = torch.load(final, weights_only=True)
  assert restored["actor_optimizer"]["state"]
  assert restored["critic_optimizer"]["state"]
