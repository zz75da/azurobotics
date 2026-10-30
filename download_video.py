from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

MAX_DOWNLOAD_BYTES = 250 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
BLOCKED_HOSTS = {
    "youtu.be",
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}


def download_video(url: str, destination: Path) -> Path:
    parsed = urlparse(url.strip())
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Utilisez une URL HTTPS directe vers un fichier vidéo.")

    hostname = parsed.hostname.lower().rstrip(".")
    if hostname in BLOCKED_HOSTS or hostname.endswith(".youtube.com"):
        raise ValueError(
            "Les pages YouTube ne sont pas des liens directs de fichier. "
            "Utilisez les options de téléchargement officielles de YouTube, "
            "ou fournissez un lien direct autorisé vers un fichier vidéo."
        )

    filename = _filename_from_url(parsed.path)
    if not filename:
        raise ValueError(
            "L'URL doit se terminer par une extension vidéo connue "
            "(.mp4, .webm, .mov, .avi, .mkv ou .m4v)."
        )

    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / filename
    temporary_path = output_path.with_name(output_path.name + ".part")
    request = Request(url.strip(), headers={"User-Agent": "RoboticsVideoDownloader/1.0"})

    try:
        with urlopen(request, timeout=30) as response:
            final_url = urlparse(response.geturl())
            final_hostname = (final_url.hostname or "").lower().rstrip(".")
            if final_url.scheme != "https" or final_hostname in BLOCKED_HOSTS or final_hostname.endswith(
                ".youtube.com"
            ):
                raise ValueError("Le serveur a redirigé vers une URL non prise en charge.")
            content_type = response.headers.get_content_type().lower()
            content_length = _content_length(response.headers.get("Content-Length"))
            if not content_type.startswith("video/") and content_type != "application/octet-stream":
                raise ValueError("Le serveur n'a pas fourni un contenu de type vidéo.")
            if content_length is not None and content_length > MAX_DOWNLOAD_BYTES:
                raise ValueError("La vidéo dépasse la limite de 250 Mo.")

            downloaded = 0
            with temporary_path.open("wb") as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > MAX_DOWNLOAD_BYTES:
                        raise ValueError("La vidéo dépasse la limite de 250 Mo.")
                    output.write(chunk)

        if downloaded == 0:
            raise ValueError("Le fichier téléchargé est vide.")
        temporary_path.replace(output_path)
        return output_path
    except (HTTPError, URLError, TimeoutError, OSError):
        temporary_path.unlink(missing_ok=True)
        raise
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _filename_from_url(path: str) -> Optional[str]:
    filename = unquote(Path(path).name)
    if Path(filename).suffix.lower() not in VIDEO_EXTENSIONS:
        return None
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(filename).stem).strip("._")
    if not stem:
        stem = "video"
    return stem + Path(filename).suffix.lower()


def _content_length(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        length = int(value)
    except ValueError as error:
        raise ValueError("Le serveur a fourni un Content-Length invalide.") from error
    if length < 0:
        raise ValueError("Le serveur a fourni un Content-Length invalide.")
    return length


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Télécharge un fichier vidéo HTTPS direct autorisé (250 Mo maximum)."
    )
    parser.add_argument(
        "--url",
        help="URL HTTPS directe vers un fichier vidéo (sinon, saisie interactive).",
    )
    args = parser.parse_args()
    print("Télécharge un fichier vidéo HTTPS direct autorisé (250 Mo maximum).")
    try:
        url = args.url or input("URL de la vidéo : ").strip()
    except EOFError:
        print("Aucune URL fournie.", file=sys.stderr)
        return 2
    try:
        saved_path = download_video(url, Path(__file__).resolve().parent / "downloads")
    except (HTTPError, URLError, TimeoutError) as error:
        print("Échec du téléchargement : {}".format(error), file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print("Impossible de télécharger cette vidéo : {}".format(error), file=sys.stderr)
        return 1
    print("Vidéo enregistrée dans : {}".format(saved_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
