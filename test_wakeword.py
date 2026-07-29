"""Standalone OpenWakeWord microphone test for the official Hey Jarvis model."""

from __future__ import annotations

import queue
import sys
import time
from importlib import resources
from pathlib import Path

import numpy as np
import sounddevice as sd
from openwakeword.model import Model
from openwakeword.utils import download_models


SAMPLE_RATE = 16_000
BLOCK_SIZE = 1_280  # 80 ms at 16 kHz
THRESHOLD = 0.5
COOLDOWN_SECONDS = 2.0
SCORE_PRINT_INTERVAL = 0.5
SIGNIFICANT_SCORE_CHANGE = 0.05
MODEL_NAME = "hey_jarvis"


def find_hey_jarvis_onnx() -> Path | None:
    """Locate an actually existing official Hey Jarvis ONNX model."""
    package_root = Path(str(resources.files("openwakeword"))).resolve()
    models_dir = package_root / "resources" / "models"
    if not models_dir.exists():
        return None

    candidates = []
    for item in models_dir.iterdir():
        if item.name.lower().startswith(MODEL_NAME) and item.name.lower().endswith(".onnx"):
            candidate = item.resolve()
            if candidate.is_file():
                candidates.append(candidate)
    return sorted(candidates)[-1] if candidates else None


def ensure_model() -> Path:
    model_path = find_hey_jarvis_onnx()
    if model_path is None:
        print("Hey Jarvis ONNX model not found; downloading via OpenWakeWord...")
        download_models([MODEL_NAME])
        model_path = find_hey_jarvis_onnx()

    models_dir = Path(str(resources.files("openwakeword"))).resolve() / "resources" / "models"
    missing = []
    if model_path is None or not model_path.is_file():
        missing.append("hey_jarvis*.onnx")
    for required_name in ("embedding_model.onnx", "melspectrogram.onnx"):
        if not (models_dir / required_name).is_file():
            missing.append(required_name)

    if missing:
        raise FileNotFoundError(
            "OpenWakeWord model download completed, but required ONNX file(s) "
            f"are missing: {', '.join(missing)}"
        )
    return model_path


def main() -> int:
    model_path = ensure_model()
    print(f"Model: {model_path}")

    wakeword = Model(
        wakeword_models=[str(model_path)],
        inference_framework="onnx",
    )

    device_index = sd.default.device[0]
    if device_index is None or int(device_index) < 0:
        raise RuntimeError("No default input microphone is configured.")
    device_index = int(device_index)
    device_info = sd.query_devices(device_index, "input")
    print(f"Microphone: index={device_index}, name={device_info['name']}")
    print(
        f"Listening at {SAMPLE_RATE} Hz, mono int16, blocksize={BLOCK_SIZE}; "
        f"threshold={THRESHOLD:.2f}. Press Ctrl+C to stop."
    )

    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
    callback_errors: queue.Queue[str] = queue.Queue()

    def audio_callback(indata, frames, time_info, status) -> None:
        del time_info
        try:
            if status:
                try:
                    callback_errors.put_nowait(f"PortAudio status: {status}")
                except queue.Full:
                    pass
            if frames != BLOCK_SIZE:
                try:
                    callback_errors.put_nowait(
                        f"Unexpected audio block size: {frames} (expected {BLOCK_SIZE})"
                    )
                except queue.Full:
                    pass

            block = indata[:, 0].copy()
            try:
                audio_queue.put_nowait(block)
            except queue.Full:
                # Keep live audio flowing: discard the oldest stale block, then retry.
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    audio_queue.put_nowait(block)
                except queue.Full:
                    pass
        except Exception as exc:  # Never let an exception escape the PortAudio callback.
            try:
                callback_errors.put_nowait(f"Audio callback error: {exc}")
            except queue.Full:
                pass

    last_print_time = 0.0
    last_printed_score: float | None = None
    last_detection_time = -COOLDOWN_SECONDS

    try:
        with sd.InputStream(
            device=device_index,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=BLOCK_SIZE,
            callback=audio_callback,
        ):
            while True:
                while True:
                    try:
                        print(callback_errors.get_nowait(), file=sys.stderr)
                    except queue.Empty:
                        break

                try:
                    audio = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                predictions = wakeword.predict(audio)
                score = max(float(value) for value in predictions.values())
                now = time.monotonic()

                changed = (
                    last_printed_score is None
                    or abs(score - last_printed_score) >= SIGNIFICANT_SCORE_CHANGE
                )
                if changed or now - last_print_time >= SCORE_PRINT_INTERVAL:
                    print(f"Confidence: {score:.3f}")
                    last_printed_score = score
                    last_print_time = now

                if score >= THRESHOLD and now - last_detection_time >= COOLDOWN_SECONDS:
                    print("[WAKE WORD] Hey Jarvis detected")
                    last_detection_time = now
    except KeyboardInterrupt:
        print("\nStopping; microphone closed safely.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
