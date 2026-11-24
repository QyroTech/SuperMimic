"""Convert source motion .pkl files into the 50 Hz Q1 training-NPZ contract.

Source schema (per .pkl): {fps, root_pos[T,3], root_rot[T,4], dof_pos[T,D], ...}.
 - root_rot is XYZW (w in the last column, verified from standing frames); the
   output stores MuJoCo WXYZ.
 - dof_pos is 24 for amass/lafan (extra left/right wrist_yaw at cols 18 and 23)
   and 22 for yuanke; after dropping the two wrist columns the remaining 22 are
   exactly the q1_v3 joint order.
 - Source fps: lafan=30, yuanke=120 (the file's own fps field of 30 is wrong per
   the data owner), amass=each file's own fps (~29-30, non-uniform).

Output NPZ matches data.py: qpos[T,29] (root_pos, WXYZ quat, 22 joints),
robot_name, root_quat_order=wxyz, fps=50.0, robot_joint_names.

Run from the repo root with:
  python -I scripts/data/convert_pkl_to_train_npz.py
(pickle executes code on load; only run this on motion data you trust.)
"""

import argparse
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# 24-dim dof_pos has two extra wrist joints that q1_v3 lacks.
DROP_COLS_24 = (18, 23)  # left_wrist_yaw, right_wrist_yaw
# Per-dataset source frame rate; None means read each file's own fps field.
SOURCE_FPS = {"motion_amass": None, "motion_lafan": 30.0, "motion_yuanke": 120.0}
TARGET_FPS = 50.0
MIN_FRAMES = 20


def joint_names_and_ranges(xml_path):
  root = ET.parse(xml_path).getroot()
  joints = [j for j in root.iter("joint") if j.get("name")]
  names = [j.get("name") for j in joints]
  ranges = np.array([[float(x) for x in j.get("range").split()] for j in joints])
  return names, ranges


def resample_qpos(qpos, source_fps, target_fps=TARGET_FPS):
  """Linear XYZ/joints and shortest-path quaternion SLERP (from resample.py)."""
  values = np.array(qpos, dtype=np.float64, copy=True)
  if values.ndim != 2 or values.shape[1] != 29 or len(values) < 2:
    raise ValueError("expected >=2 frames of 29-dim qpos")
  if not np.isfinite(values).all():
    raise ValueError("nonfinite pose")
  norms = np.linalg.norm(values[:, 3:7], axis=-1)
  if np.any(np.abs(norms - 1) > 0.01):
    raise ValueError("root quaternions must be unit length")
  values[:, 3:7] /= norms[:, None]
  if np.isclose(source_fps, target_fps):
    return values.astype(np.float32)
  duration = (len(values) - 1) / source_fps
  count = int(np.floor(duration * target_fps + 1e-9)) + 1
  phase = np.arange(count, dtype=np.float64) / target_fps * source_fps
  phase = np.minimum(phase, len(values) - 1)
  left = np.minimum(phase.astype(np.int64), len(values) - 2)
  blend = (phase - left)[:, None]
  result = (1 - blend) * values[left] + blend * values[left + 1]
  qa, qb = values[left, 3:7], values[left + 1, 3:7]
  dot = (qa * qb).sum(-1, keepdims=True)
  qb = np.where(dot < 0, -qb, qb)
  dot = np.abs(dot).clip(0, 1)
  theta = np.arccos(dot)
  denom = np.sin(theta)
  large = denom > 1e-6
  wa = np.divide(np.sin((1 - blend) * theta), denom, out=(1 - blend).copy(), where=large)
  wb = np.divide(np.sin(blend * theta), denom, out=blend.copy(), where=large)
  rot = wa * qa + wb * qb
  result[:, 3:7] = rot / np.linalg.norm(rot, axis=-1, keepdims=True)
  return result.astype(np.float32)


def dof_to_22(dof):
  d = dof.shape[1]
  if d == 22:
    return dof
  if d == 24:
    return np.delete(dof, DROP_COLS_24, axis=1)
  raise ValueError(f"unexpected dof width {d}")


def build_qpos(record, quat_order="xyzw"):
  root_pos = np.asarray(record["root_pos"], dtype=np.float64)
  root_rot = np.asarray(record["root_rot"], dtype=np.float64)
  dof = dof_to_22(np.asarray(record["dof_pos"], dtype=np.float64))
  if not (len(root_pos) == len(root_rot) == len(dof)):
    raise ValueError("root_pos, root_rot and dof_pos differ in length")
  if root_pos.shape[1] != 3 or root_rot.shape[1] != 4 or dof.shape[1] != 22:
    raise ValueError(f"bad shapes {root_pos.shape} {root_rot.shape} {dof.shape}")
  qpos = np.empty((len(root_pos), 29), dtype=np.float64)
  qpos[:, :3] = root_pos
  qpos[:, 3:7] = root_rot[:, [3, 0, 1, 2]] if quat_order == "xyzw" else root_rot
  qpos[:, 7:] = dof
  norms = np.linalg.norm(qpos[:, 3:7], axis=-1)
  if np.any(norms < 1e-6) or not np.isfinite(qpos).all():
    raise ValueError("nonfinite or degenerate pose")
  qpos[:, 3:7] /= norms[:, None]
  return qpos


def source_fps_for(dataset, record):
  fps = SOURCE_FPS[dataset]
  if fps is None:
    fps = float(np.asarray(record["fps"]).reshape(-1)[0])
  if not np.isfinite(fps) or fps <= 0:
    raise ValueError(f"bad source fps {fps}")
  return fps


def convert_one(path, dataset, names, out_path, quat_order):
  with path.open("rb") as fp:
    record = pickle.load(fp)
  src_fps = source_fps_for(dataset, record)
  qpos = build_qpos(record, quat_order)
  out = resample_qpos(qpos, src_fps, TARGET_FPS)
  if len(out) < MIN_FRAMES:
    raise ValueError(f"{len(out)} frames < {MIN_FRAMES} after resample")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    out_path,
    qpos=out,
    robot_name=np.array("q1_v3"),
    root_quat_order=np.array("wxyz"),
    fps=np.array(TARGET_FPS),
    robot_joint_names=np.array(names),
    source_file=np.array(str(path)),
    source_fps=np.array(src_fps),
    source_frames=np.array(len(qpos)),
  )
  return len(qpos), len(out), src_fps, out[:, 7:]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--datasets-dir", default="datasets")
  ap.add_argument("--out-dir", default="datasets/train_50hz")
  ap.add_argument("--robot-xml", default="assets/q1_v3/xml/q1_v3_mc.xml")
  ap.add_argument("--quat-order", default="xyzw", choices=["xyzw", "wxyz"])
  ap.add_argument("--datasets", nargs="*", default=list(SOURCE_FPS))
  ap.add_argument("--limit", type=int, default=0, help="per-dataset cap for a dry run")
  args = ap.parse_args()

  names, ranges = joint_names_and_ranges(args.robot_xml)
  if len(names) != 22:
    sys.exit(f"expected 22 joints, got {len(names)}")
  datasets_dir, out_dir = Path(args.datasets_dir), Path(args.out_dir)

  grand_ok = grand_fail = 0
  for dataset in args.datasets:
    root = datasets_dir / dataset
    paths = sorted(root.rglob("*.pkl"))
    if args.limit:
      paths = paths[: args.limit]
    ok = fail = 0
    over = np.zeros(22)  # fraction of frames outside the v3 joint range, worst clip
    for path in paths:
      rel = path.relative_to(datasets_dir).with_suffix(".npz")
      out_path = out_dir / rel
      try:
        _, n_out, src, dof = convert_one(path, dataset, names, out_path, args.quat_order)
        frac = ((dof < ranges[:, 0]) | (dof > ranges[:, 1])).mean(axis=0)
        over = np.maximum(over, frac)
        ok += 1
      except Exception as error:  # noqa: BLE001
        print(f"FAIL {path}: {type(error).__name__}: {error}", flush=True)
        fail += 1
    worst = int(over.argmax())
    print(
      f"[{dataset}] ok={ok} fail={fail} src_fps={SOURCE_FPS[dataset] or 'per-file'} "
      f"| worst joint out-of-range: {names[worst]} {over[worst]*100:.1f}%",
      flush=True,
    )
    grand_ok += ok
    grand_fail += fail
  print(f"\nTOTAL ok={grand_ok} fail={grand_fail} -> {out_dir}", flush=True)


if __name__ == "__main__":
  main()
