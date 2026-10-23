from email.message import Message
from io import BytesIO
from pathlib import Path

import pytest

import download_video


class FakeResponse:
    def __init__(self, payload, content_type="video/mp4", url="https://media.example/test.mp4"):
        self._payload = BytesIO(payload)
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.headers["Content-Length"] = str(len(payload))
        self._url = url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, size=-1):
        return self._payload.read(size)

    def geturl(self):
        return self._url


def test_download_saves_direct_video_url(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        download_video,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(b"video-bytes"),
    )

    saved_path = download_video.download_video(
        "https://media.example/test.mp4",
        tmp_path / "downloads",
    )

    assert saved_path.read_bytes() == b"video-bytes"
    assert saved_path.name == "test.mp4"
    assert not saved_path.with_name(saved_path.name + ".part").exists()


def test_youtube_watch_page_is_rejected_without_network(tmp_path: Path):
    with pytest.raises(ValueError, match="pages YouTube"):
        download_video.download_video(
            "https://youtu.be/s00_s3O3kYU",
            tmp_path,
        )


def test_non_https_direct_video_url_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="HTTPS"):
        download_video.download_video(
            "http://media.example/test.mp4",
            tmp_path,
        )


def test_non_video_response_is_rejected_and_partial_file_removed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        download_video,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(b"<html>", content_type="text/html"),
    )

    with pytest.raises(ValueError, match="type vidéo"):
        download_video.download_video(
            "https://media.example/test.mp4",
            tmp_path,
        )

    assert not (tmp_path / "test.mp4.part").exists()


def test_redirect_to_youtube_is_rejected_and_partial_file_removed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        download_video,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            b"video-bytes",
            url="https://www.youtube.com/watch?v=test",
        ),
    )

    with pytest.raises(ValueError, match="redirigé"):
        download_video.download_video(
            "https://media.example/test.mp4",
            tmp_path,
        )

    assert not (tmp_path / "test.mp4.part").exists()
