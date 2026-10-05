"""Show live raw hands, palm-local windows, and palm_local_v2 L21 predictions.

Two rows (left/right), three columns (raw image XY / palm-local / L21 FK).
Missing sides clear immediately; the robot panel waits for three new valid
frames. Darker palm skeletons are older frames from the actual model window.
Only display projections and zoom change here; model inputs remain untouched.

Run with --model-asset-path pointing to hand_landmarker.task. q/Esc quits,
1/2/3/4 selects YZ/XZ/XY/ISO, w toggles window traces, +/- zooms the aligned
panels, and s saves a PNG. --headless --frames N can verify rendering without a
window. No H5 recording, robot commands, smoothing, or identity tracking.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.kinematics import create_hand_kinematics
from retargeting.config import L21
from retargeting.mediapipe import (
    MediaPipeCameraAdapter,
    PALM_LOCAL_V2_CHECKPOINT,
    load_palm_local_retargeter,
)
from retargeting.simulation import angle18_to_nodes
from scripts.compare_zuobiaoxi_vs_origin import (
    COLORS, ROBOT_EDGES, ROBOT_TIPS, SOURCE_EDGES, draw_skeleton, project, text,
)


SIDES = ("left", "right")
WINDOW = "MediaPipe realtime: raw -> palm-local -> window -> L21"
WIDTH, HEIGHT, HEADER = 420, 320, 100
PALM_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    (0, 10), (10, 11), (11, 12), (12, 13), (13, 14),
    (0, 15), (15, 16), (16, 17), (17, 18), (18, 19),
    (0, 20), (20, 21), (21, 22), (22, 23), (23, 24),
)
PALM_TIPS = (4, 9, 14, 19, 24)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-asset-path", type=Path, default=Path("hand_landmarker.task"))
    parser.add_argument("--checkpoint", type=Path, default=PALM_LOCAL_V2_CHECKPOINT)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--view", choices=("yz", "xz", "xy", "iso"), default="iso")
    parser.add_argument("--palm-radius", type=float, default=0.22, help="fixed normalized-coordinate plot radius")
    parser.add_argument("--robot-radius", type=float, default=0.16, help="fixed robot-coordinate plot radius")
    parser.add_argument("--labels", action="store_true", help="show landmark/FK node indices")
    parser.add_argument("--frames", type=int, default=0, help="0: run until q/Esc; otherwise capture N frames")
    parser.add_argument("--headless", action="store_true", help="render without opening a GUI")
    parser.add_argument("--snapshot", type=Path, help="save the final displayed frame as a PNG")
    args = parser.parse_args(argv)
    if args.frames < 0 or (args.headless and args.frames == 0):
        parser.error("--frames must be nonnegative, and positive with --headless")
    for name in ("palm_radius", "robot_radius"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.snapshot is not None and args.snapshot.suffix.lower() != ".png":
        parser.error("--snapshot must end with .png")
    return args


def make_fk():
    """Create the existing, unchanged URDF FK once per side."""
    return {
        side: create_hand_kinematics(
            getattr(L21, f"{side}_urdf"), L21.hand_kinematics_config(), device="cpu",
            scale_factor=L21.training.robot_scale,
        )
        for side in SIDES
    }


def fk_positions(fk, angles):
    nodes = angle18_to_nodes(angles)
    with torch.no_grad():
        points = fk.forward(torch.from_numpy(nodes[None]))[2][0].cpu().numpy()
    if points.shape != (23, 3) or not np.isfinite(points).all():
        raise ValueError("L21 FK must produce finite (23,3) points")
    return points


def _panel(title, subtitle, color):
    panel = np.full((HEIGHT, WIDTH, 3), 18, dtype=np.uint8)
    text(panel, title, (12, 24), color, scale=0.52)
    text(panel, subtitle, (12, 45), scale=0.40)
    return panel


def _plot(panel, points, edges, tips, color, args, radius, labels=False):
    projected = project(points, args.view)
    # Fixed origin and isotropic scale; do not recenter/rescale each new frame.
    pixels = projected * np.array([1, -1]) * (185 / (2 * radius))
    pixels += np.array([WIDTH / 2, 105])
    pixels = np.rint(np.clip(pixels, -10000, 10000)).astype(np.int32)
    # Clip drawing to the plot area so zoom cannot obscure status text.
    plot_area = panel[60:250]
    cv2.line(plot_area, (12, 105), (WIDTH - 12, 105), (35, 35, 35), 1)
    cv2.line(plot_area, (WIDTH // 2, 5), (WIDTH // 2, 185), (35, 35, 35), 1)
    draw_skeleton(plot_area, pixels, edges, color, tips, labels)


def render(raw_frame, palms, payload, angles, robot_points, streaks, reasons,
           resets, args, frame_index, fps, traces=True, zoom=1.0):
    """Render current data only; no historical robot pose is cached here."""
    header = np.full((HEADER, WIDTH * 3, 3), 28, dtype=np.uint8)
    timestamp = raw_frame["timestamp"]
    text(header, f"frame={frame_index}  timestamp={timestamp} unix_ms  FPS={fps:.1f}  "
         f"view={args.view.upper()}  zoom={zoom:.2f}", (14, 22), scale=0.57)
    text(header, f"{args.checkpoint.parent.name}/{args.checkpoint.name} | palm_local_to_l21_v1 | "
         "identity tracking OFF | fixed display scales",
         (14, 45), scale=0.48)
    text(header, "q/Esc quit | 1 YZ  2 XZ  3 XY  4 ISO | w window traces | +/- zoom | s PNG", (14, 68), scale=0.48)
    text(header, "Raw: normalized image XY. Palm: normalized coordinates. Robot: URDF units. Screen distances differ.",
         (14, 89), scale=0.43)
    rows = [header]
    for side in SIDES:
        color = COLORS[side]
        raw, palm, prediction = raw_frame[side], palms[side], angles[side]
        window = None if payload is None else payload["hands"][side]
        status = "INVALID" if palm is None else f"WARMUP {streaks[side]}/3" if window is None else "VALID"
        raw_panel = _panel(f"{side.upper()} | RAW 21 | {'MISSING' if raw is None else 'DETECTED'}",
                           "Image X right, Y down; unmirrored", color)
        if raw is not None:
            width = raw_frame.get("metadata", {}).get("image_width", 640)
            height = raw_frame.get("metadata", {}).get("image_height", 480)
            scale = min((WIDTH - 40) / width, 185 / height)
            size = np.array([width, height]) * scale
            offset = np.array([(WIDTH - size[0]) / 2, 5 + (185 - size[1]) / 2])
            pixels = np.rint(np.clip(raw[:, :2] * size + offset, -10000, 10000)).astype(np.int32)
            draw_skeleton(raw_panel[60:250], pixels, SOURCE_EDGES, color, (4, 8, 12, 16, 20), args.labels)
            text(raw_panel, f"shape={raw.shape} dtype={raw.dtype} finite={bool(np.isfinite(raw).all())}",
                 (12, 285), scale=0.40)
        else:
            text(raw_panel, "No current hand", (110, 165), color)

        palm_panel = _panel(f"{side.upper()} | PALM-LOCAL 25 | {status}",
                            f"{args.view.upper()} | streak={streaks[side]}/3 | resets={resets[side]}", color)
        if palm is not None:
            if traces and window is not None:
                for age, alpha in ((0, 0.25), (1, 0.55)):
                    _plot(palm_panel, window[age], PALM_EDGES, (), tuple(int(c * alpha) for c in color),
                          args, args.palm_radius / zoom)
            _plot(palm_panel, palm, PALM_EDGES, PALM_TIPS, color, args, args.palm_radius / zoom, args.labels)
            text(palm_panel, f"shape={palm.shape} finite={bool(np.isfinite(palm).all())}", (12, 268), scale=0.42)
            text(palm_panel, f"window={None if window is None else window.shape}", (12, 289), scale=0.42)
            text(palm_panel, "traces: t-2 dark / t-1 medium / t current" if traces else "current frame only",
                 (12, 309), scale=0.38)
        else:
            text(palm_panel, "INVALID - side buffer cleared", (45, 165), color, scale=0.50)
            reason = reasons[side] or "missing"
            text(palm_panel, reason[:57], (12, 285), scale=0.37)
            if len(reason) > 57:
                text(palm_panel, reason[57:114], (12, 306), scale=0.37)

        robot_panel = _panel(f"{side.upper()} | L21 FK | {status}",
                             f"{args.view.upper()} | current 18D prediction only", color)
        if prediction is not None:
            _plot(robot_panel, robot_points[side], ROBOT_EDGES, ROBOT_TIPS, color, args,
                  args.robot_radius / zoom, args.labels)
            text(robot_panel, f"angles={prediction.shape} finite={bool(np.isfinite(prediction).all())}",
                 (12, 269), scale=0.42)
            text(robot_panel, f"range=[{prediction.min():.3f}, {prediction.max():.3f}] rad", (12, 291), scale=0.42)
        else:
            text(robot_panel, "No current prediction", (90, 155), color, scale=0.52)
            text(robot_panel, "Waiting for 3 consecutive valid frames", (25, 184), scale=0.44)
        rows.append(np.hstack((raw_panel, palm_panel, robot_panel)))
    return np.vstack(rows)


def save_png(path, canvas):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", canvas)
    if not success:
        raise RuntimeError("PNG encoding failed")
    path.write_bytes(encoded.tobytes())
    print(f"Saved {path.resolve()}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    canvas = None
    created_window = False
    completed = 0
    resets = {side: 0 for side in SIDES}
    outputs = {side: 0 for side in SIDES}
    zoom, traces = 1.0, True
    try:
        if args.device == "cpu":
            torch.set_num_threads(1)
        model = load_palm_local_retargeter(args.checkpoint, args.device)
        fk = make_fk()
        print(f"Checkpoint: {args.checkpoint}\nCamera: {args.camera_index}\n"
              "q/Esc quit; missing sides are cleared without holding old poses.", flush=True)
        with MediaPipeCameraAdapter(args.model_asset_path, args.camera_index) as camera:
            if not args.headless:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                created_window = True
                cv2.resizeWindow(WINDOW, 1150, 676)
            started = time.perf_counter()
            while args.frames == 0 or completed < args.frames:
                previous = dict(camera.processor.valid_streak)
                payload = camera.next_input()
                predictions = {side: None for side in SIDES} if payload is None else model.predict(payload, args.device)["hands"]
                positions = {side: None for side in SIDES}
                completed += 1
                for side in SIDES:
                    if previous[side] and camera.processor.valid_streak[side] == 0:
                        resets[side] += 1
                        print(f"frame={completed} {side}: RESET", flush=True)
                    if predictions[side] is not None:
                        positions[side] = fk_positions(fk[side], predictions[side])
                        outputs[side] += 1
                        if previous[side] == 2:
                            print(f"frame={completed} {side}: window ready after 3 valid frames", flush=True)
                fps = completed / max(time.perf_counter() - started, 1e-9)
                canvas = render(camera.last_raw_frame, camera.processor.current_hands, payload,
                                predictions, positions, camera.processor.valid_streak,
                                camera.processor.invalid_reasons, resets, args, completed, fps, traces, zoom)
                if not args.headless:
                    cv2.imshow(WINDOW, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key in (ord('1'), ord('2'), ord('3'), ord('4')):
                        args.view = {ord('1'): 'yz', ord('2'): 'xz', ord('3'): 'xy', ord('4'): 'iso'}[key]
                    elif key == ord('w'):
                        traces = not traces
                    elif key in (ord('+'), ord('=')):
                        zoom = min(zoom * 1.2, 8.0)
                    elif key == ord('-'):
                        zoom = max(zoom / 1.2, 0.125)
                    elif key == ord('s'):
                        save_png(args.snapshot or PROJECT_ROOT / "picture/mediapipe_realtime.png", canvas)
    except KeyboardInterrupt:
        print("Stopped.")
    except Exception as error:
        print(f"Realtime visualization failed: {error}", file=sys.stderr)
        return 1
    finally:
        if created_window:
            cv2.destroyAllWindows()
    if args.snapshot is not None and canvas is not None:
        save_png(args.snapshot, canvas)
    print(f"Frames={completed} outputs={outputs} resets={resets}; resources released", flush=True)
    if not any(outputs.values()):
        print("No real-hand model output observed during this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
