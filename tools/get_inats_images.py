#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


API_URL = "https://api.inaturalist.org/v1/observations"

# Identify the application politely. Replace the contact information.
USER_AGENT = (
    "moth-photo-downloader/1.0 "
    "(research project; contact: your-email@example.com)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the first photo from the most recently created "
            "iNaturalist observations for a taxon."
        )
    )
    parser.add_argument(
        "taxon_id",
        type=int,
        help="iNaturalist taxon ID",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Maximum number of observations to retrieve (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Images root directory. Defaults to the Django project's "
            "MOTHS_IMAGE_DIR. Photos are stored in <output>/<taxon_id>/ and "
            "metadata in <output>/<taxon_id>_observations.json."
        ),
    )
    parser.add_argument(
        "--image-size",
        choices=("original", "large", "medium"),
        default="original",
        help="Requested iNaturalist image size (default: original)",
    )
    parser.add_argument(
        "--research-grade-only",
        action="store_true",
        help="Restrict results to research-grade observations",
    )
    return parser.parse_args()


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
    )
    return session


def request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any],
    attempts: int = 4,
) -> dict[str, Any]:
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt + 1 == attempts:
                raise

            time.sleep(2**attempt)

    raise RuntimeError("Unreachable")


def fetch_observations(
    session: requests.Session,
    taxon_id: int,
    count: int,
    research_grade_only: bool,
    created_d2: str | None = None,
    exclude_ids: set[Any] | None = None,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("--count must be greater than zero")

    exclude_ids = exclude_ids or set()
    observations: list[dict[str, Any]] = []
    page = 1

    # Fetch full pages (200 is the API max) and count only *new* observations
    # toward the target. When extending coverage backwards, the first page(s)
    # under a ``created_d2`` date are dominated by observations we already have
    # (same boundary date), so we must keep paging past them rather than filter
    # a single small page afterwards.
    page_size = 200
    while len(observations) < count:
        params: dict[str, Any] = {
            "taxon_id": taxon_id,
            "photos": "true",
            "order_by": "created_at",
            "order": "desc",
            "per_page": page_size,
            "page": page,
        }

        # Restrict to observations created on or before this date (YYYY-MM-DD)
        # so we can extend coverage backwards in time on repeat runs.
        if created_d2:
            params["created_d2"] = created_d2

        if research_grade_only:
            params["quality_grade"] = "research"

        payload = request_json(session, API_URL, params=params)
        results = payload.get("results", [])

        if not isinstance(results, list):
            raise RuntimeError("Unexpected iNaturalist API response")

        for observation in results:
            if observation.get("id") in exclude_ids:
                continue
            observations.append(observation)
            if len(observations) >= count:
                break

        if len(results) < page_size:
            break

        page += 1

    return observations[:count]


def select_first_photos(
    observations: list[dict[str, Any]],
    image_size: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    for observation in observations:
        observation_photos = observation.get("observation_photos") or []
        if not observation_photos:
            continue

        photo = observation_photos[0].get("photo") or {}
        image_url = photo.get("url")
        if not image_url:
            continue

        # API photo URLs normally contain a size component such as "square".
        # Request the desired rendition.
        image_url = image_url.replace("/square.", f"/{image_size}.")

        taxon = observation.get("taxon") or {}
        user = observation.get("user") or {}

        selected.append(
            {
                "observation_id": observation["id"],
                "observation_url": (
                    f"https://www.inaturalist.org/observations/"
                    f"{observation['id']}"
                ),
                "created_at": observation.get("created_at"),
                "observed_on": observation.get("observed_on"),
                "quality_grade": observation.get("quality_grade"),
                "taxon_id": taxon.get("id"),
                "scientific_name": taxon.get("name"),
                "common_name": taxon.get("preferred_common_name"),
                "observer_login": user.get("login"),
                "photo_id": photo.get("id"),
                "photo_position": observation_photos[0].get("position"),
                "image_url": image_url,
                "license_code": photo.get("license_code"),
                "attribution": photo.get("attribution"),
            }
        )

    return selected


def image_extension(response: requests.Response, url: str) -> str:
    content_type = response.headers.get("Content-Type", "")
    content_type = content_type.partition(";")[0].strip().lower()

    extension = mimetypes.guess_extension(content_type)
    if extension == ".jpe":
        extension = ".jpg"

    if extension:
        return extension

    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return suffix

    return ".jpg"


def download_photo(
    session: requests.Session,
    item: dict[str, Any],
    image_directory: Path,
    taxon_id: int,
) -> Path:
    url = item["image_url"]

    response = session.get(url, timeout=60, stream=True)
    response.raise_for_status()

    extension = image_extension(response, url)
    filename = (
        f"{taxon_id}_"
        f"observation_{item['observation_id']}_"
        f"photo_{item['photo_id']}"
        f"{extension}"
    )
    destination = image_directory / filename
    temporary = destination.with_suffix(destination.suffix + ".part")

    with temporary.open("wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)

    temporary.replace(destination)
    return destination


def resolve_images_dir(output: Path | None) -> Path:
    """Return the images root directory.

    Uses ``--output`` when provided, otherwise falls back to the Django
    project's ``MOTHS_IMAGE_DIR``. Django is only imported in the fallback case,
    so the script still runs standalone when ``--output`` is given.
    """
    if output is not None:
        return output

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "moths_list.settings")

    import django

    django.setup()

    from django.conf import settings

    return Path(settings.MOTHS_IMAGE_DIR)


def parse_created(item: dict[str, Any]) -> datetime | None:
    """Parse an item's ``created_at`` into an aware ``datetime``, or ``None``."""
    value = item.get("created_at")
    if not value:
        return None
    try:
        # iNaturalist uses ISO 8601; normalize a trailing "Z" for fromisoformat.
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def load_existing(metadata_path: Path) -> list[dict[str, Any]]:
    """Load previously stored observations, or an empty list."""
    if not metadata_path.is_file():
        return []
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    return data if isinstance(data, list) else []


def sort_key(item: dict[str, Any]) -> float:
    """Sort key placing newest observations first (missing dates last)."""
    created = parse_created(item)
    return created.timestamp() if created else float("-inf")


def main() -> None:
    args = parse_args()

    # Store photos in a per-tax_id subfolder of the images directory, and the
    # observation metadata alongside it at the images root.
    images_dir = resolve_images_dir(args.output)
    image_directory = images_dir / str(args.taxon_id)
    image_directory.mkdir(parents=True, exist_ok=True)

    metadata_path = images_dir / f"{args.taxon_id}_observations.json"

    # Incremental mode: if we already have observations for this tax, only fetch
    # ones created before the earliest we've seen, to extend coverage backwards.
    existing_items = load_existing(metadata_path)
    existing_ids = {
        item.get("observation_id")
        for item in existing_items
        if item.get("observation_id") is not None
    }
    created_d2: str | None = None
    if existing_items:
        earliest = min(
            (d for d in (parse_created(i) for i in existing_items) if d),
            default=None,
        )
        if earliest is not None:
            created_d2 = earliest.date().isoformat()
            print(
                f"{len(existing_items)} existing observations; "
                f"fetching older than {earliest.isoformat()} "
                f"(created on/before {created_d2})"
            )

    session = create_session()

    observations = fetch_observations(
        session=session,
        taxon_id=args.taxon_id,
        count=args.count,
        research_grade_only=args.research_grade_only,
        created_d2=created_d2,
        exclude_ids=existing_ids,
    )

    items = select_first_photos(observations, args.image_size)

    # ``fetch_observations`` already excluded known ids; this guards against any
    # slipping through (e.g. duplicate ids within a page).
    new_items = [i for i in items if i["observation_id"] not in existing_ids]

    print(f"Found {len(observations)} new observations")
    print(f"Selected {len(new_items)} first photos")
    print(f"Metadata: {metadata_path}")

    # Merge with what we already had and persist (newest first) up front, so the
    # file stays valid even if downloading is interrupted.
    merged = existing_items + new_items
    merged.sort(key=sort_key, reverse=True)
    metadata_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    failures = 0

    for index, item in enumerate(new_items, start=1):
        try:
            destination = download_photo(
                session=session,
                item=item,
                image_directory=image_directory,
                taxon_id=args.taxon_id,
            )
            print(
                f"[{index:03d}/{len(new_items):03d}] "
                f"observation {item['observation_id']} -> "
                f"{destination.name}"
            )
        except requests.RequestException as error:
            failures += 1
            item["download_error"] = str(error)
            print(
                f"[{index:03d}/{len(new_items):03d}] "
                f"observation {item['observation_id']} failed: {error}"
            )

        # Avoid making rapid consecutive requests to the image servers.
        time.sleep(0.25)

    # Rewrite metadata to capture any download errors.
    metadata_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Downloaded: {len(new_items) - failures}")
    print(f"Failed: {failures}")
    print(f"Total observations on file: {len(merged)}")


if __name__ == "__main__":
    main()
