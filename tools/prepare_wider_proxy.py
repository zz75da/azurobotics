"""Download a deterministic, tiny WIDER FACE proxy subset outside this repository."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import ZipFile

ANNOTATION_REVISION = "db171f1b7fedf4d3453e81297ff02f9915356d19"
IMAGE_REVISION = "04cab2dc3e18be03e2935b731c1dfa9eb29a31b5"
ANNOTATION_URL = ("https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/"
                  f"{ANNOTATION_REVISION}/data/wider_face_split.zip?download=true")
IMAGE_URL = ("https://huggingface.co/datasets/yuuMQ/Retinaface_WiderFaceDataset/resolve/"
             f"{IMAGE_REVISION}/widerface/train/images/{{}}?download=true")
MAX_IMAGE_BYTES = 20_000_000


def _download(url: str, limit: int) -> bytes:
    request = Request(url, headers={"User-Agent": "face-blur-proxy-benchmark/1.0"})
    with urlopen(request, timeout=40) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Download exceeds safety limit: {url}")
    return data


def _records(annotation_text: str):
    lines = annotation_text.splitlines()
    cursor = 0
    while cursor < len(lines):
        name = lines[cursor].strip(); cursor += 1
        if not name:
            continue
        count = int(lines[cursor].strip()); cursor += 1
        if count == 0:
            if cursor < len(lines) and len(lines[cursor].split()) >= 10:
                cursor += 1
            yield name, False
            continue
        has_face = False
        for _ in range(count):
            values = [int(value) for value in lines[cursor].split()]
            cursor += 1
            if len(values) >= 10 and values[2] > 0 and values[3] > 0 and values[7] != 1:
                has_face = True
        if has_face:
            yield name, True


def prepare(output_dir: Path, seed: int = 20261025, positive_count: int = 8) -> Path:
    output_dir = output_dir.expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if output_dir == repo or repo in output_dir.parents:
        raise ValueError("Dataset downloads must be outside the repository")
    annotation_zip = _download(ANNOTATION_URL, 10_000_000)
    with ZipFile(BytesIO(annotation_zip)) as archive:
        member = next(name for name in archive.namelist() if name.endswith("wider_face_train_bbx_gt.txt"))
        annotation_text = archive.read(member).decode("utf-8-sig")
    records = list(_records(annotation_text))
    positives = [name for name, has_face in records if has_face]
    negatives = [name for name, has_face in records if not has_face]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = output_dir / "wider_face_train_bbx_gt.txt"
    annotation_path.write_text(annotation_text, encoding="utf-8")
    image_root = output_dir / "images"
    downloaded = []

    def fetch(name: str, has_face: bool) -> bool:
        image_path = image_root / name
        try:
            image_path.parent.mkdir(parents=True, exist_ok=True)
            content = _download(IMAGE_URL.format(quote(name, safe="/")), MAX_IMAGE_BYTES)
            if not content.startswith(b"\xff\xd8"):
                raise ValueError("Image response is not a JPEG")
            image_path.write_bytes(content)
        except (OSError, URLError, ValueError):
            image_path.unlink(missing_ok=True)
            return False
        downloaded.append({"path": name, "sha256": hashlib.sha256(content).hexdigest(),
                           "has_face": has_face})
        return True

    face_count = 0
    for name in positives:
        face_count += int(fetch(name, True))
        if face_count >= positive_count:
            break
    if face_count < positive_count:
        raise RuntimeError(f"Only downloaded {face_count} labeled positive images; needed {positive_count}")
    negative_count = 0
    for name in negatives:
        negative_count += int(fetch(name, False))
    if negative_count != len(negatives):
        raise RuntimeError(f"Only downloaded {negative_count} of {len(negatives)} labeled negative images")
    manifest = {"proxy": True, "seed": seed, "annotation_revision": ANNOTATION_REVISION,
                "image_revision": IMAGE_REVISION,
                "annotation_sha256": hashlib.sha256(annotation_text.encode("utf-8")).hexdigest(),
                "annotation_archive_sha256": hashlib.sha256(annotation_zip).hexdigest(),
                "positive_images": face_count,
                "negative_images": negative_count,
                "images": downloaded}
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "face-blur-wider-proxy")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--positive-count", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps({"manifest": str(prepare(args.output_dir, args.seed, args.positive_count))}, indent=2))


if __name__ == "__main__":
    main()
