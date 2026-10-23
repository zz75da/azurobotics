from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

import video_privacy


class FakeResponse:
    def __init__(self, payload: bytes, url: str = video_privacy.YUNET_MODEL_URL) -> None:
        self.payload = payload
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def geturl(self) -> str:
        return self.url

    def read(self, size: int = -1) -> bytes:
        chunk, self.payload = self.payload[:size], self.payload[size:]
        return chunk


def test_yunet_model_is_downloaded_once_and_hash_verified(tmp_path: Path, monkeypatch):
    payload = b"x" * video_privacy.YUNET_MODEL_MIN_BYTES
    monkeypatch.setattr(
        video_privacy,
        "YUNET_MODEL_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    calls = []

    def fake_urlopen(*_args, **_kwargs):
        calls.append(True)
        return FakeResponse(payload)

    monkeypatch.setattr(video_privacy, "urlopen", fake_urlopen)

    first_path = video_privacy.ensure_yunet_model(tmp_path)
    second_path = video_privacy.ensure_yunet_model(tmp_path)

    assert first_path == second_path
    assert first_path.read_bytes() == payload
    assert len(calls) == 1
    assert not first_path.with_suffix(".onnx.part").exists()


def test_yunet_model_rejects_redirect_to_untrusted_host(tmp_path: Path, monkeypatch):
    payload = b"x" * video_privacy.YUNET_MODEL_MIN_BYTES
    monkeypatch.setattr(video_privacy, "YUNET_MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        video_privacy,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(payload, "https://example.com/model.onnx"),
    )

    with pytest.raises(RuntimeError, match="hôte inattendu"):
        video_privacy.ensure_yunet_model(tmp_path)

    assert not (tmp_path / video_privacy.YUNET_MODEL_NAME).exists()
    assert not (tmp_path / (video_privacy.YUNET_MODEL_NAME + ".part")).exists()


def test_pixelation_expands_face_box_and_changes_only_its_roi():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[:, :, 0] = np.arange(100, dtype=np.uint8)
    original = frame.copy()

    video_privacy._pixelate_face(
        frame,
        np.array([40.0, 25.0, 20.0, 20.0]),
        width=100,
        height=80,
    )

    assert not np.array_equal(frame[15:58, 32:68], original[15:58, 32:68])
    assert np.array_equal(frame[:10], original[:10])
    assert np.array_equal(frame[:, :20], original[:, :20])


def test_face_mode_uses_yunet_and_anonymizes_each_frame(tmp_path: Path, monkeypatch):
    source_path = tmp_path / "synthetic.avi"
    video_privacy.create_sample_video(source_path, seconds=1, fps=2)
    source_bytes = source_path.read_bytes()
    model_path = tmp_path / video_privacy.YUNET_MODEL_NAME
    model_path.write_bytes(b"verified test model")
    monkeypatch.setattr(video_privacy, "ensure_yunet_model", lambda _directory: model_path)

    class FakeDetector:
        def __init__(self):
            self.calls = 0

        def detect(self, _frame):
            self.calls += 1
            if self.calls % 5 == 1:
                return 1, np.array([[100, 100, 30, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9]])
            return 1, None

    detector = FakeDetector()
    monkeypatch.setattr(
        cv2,
        "FaceDetectorYN_create",
        lambda *_args: detector,
    )

    output, frame_count, detections = video_privacy.anonymize_video(source_bytes, "faces")

    assert frame_count == 2
    assert detections == 2
    assert detector.calls == frame_count * 5
    assert output


def test_multiscale_detection_finds_small_faces_and_suppresses_duplicates():
    class FakeDetector:
        def __init__(self):
            self.calls = 0

        def detect(self, _frame):
            self.calls += 1
            if self.calls == 1:
                return 1, np.array([[60, 60, 24, 24, *([0] * 10), 0.9]])
            if self.calls == 2:
                return 1, np.array([[100, 100, 40, 40, *([0] * 10), 0.95]])
            return 1, None

    detector = FakeDetector()
    faces = video_privacy._detect_faces_multiscale(
        detector,
        np.zeros((360, 640, 3), dtype=np.uint8),
    )

    assert detector.calls == 5
    assert faces is not None
    assert len(faces) == 1
    assert np.allclose(faces[0][:4], [60, 60, 24, 24])


def test_invalid_blur_mode_is_rejected():
    with pytest.raises(ValueError, match="Mode de floutage inconnu"):
        video_privacy.anonymize_video(b"video", "unknown")
