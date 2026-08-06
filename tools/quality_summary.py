#!/usr/bin/env python3

"""Print the Django index table as a human-readable Markdown report.

This reproduces the ``index`` view's per-``tax_id`` table (names, class-stage
counts, box counts, and the harvest "source data" tallies) but as a Markdown
table written to stdout (or ``--output``), so it can be pasted into a doc or an
issue. It reads the same per-tax summary cache the web app uses
(``<tax>_summary.json`` under ``MOTHS_LABEL_DIR``), building any that are
missing exactly like the index page does.

All paths come from the Django settings/environment (the ``MOTHS_*`` variables
the web app uses), so run it with the same environment configured — nothing is
passed on the command line for the primary dataset.

Two things differ from the web table on purpose:

* The "Source data check" section lists the **full** set of audit-CSV statuses
  (the web view collapses several away), and the finishing state is written as
  the full termination word (``done`` / ``no_more_observations`` /
  ``reached_scan_limit`` / ``corrupted``) rather than a single letter.
* An extra ``present`` column is added: a check mark when the tax_id also exists
  in a second images directory passed as an argument — detected by the presence
  of its ``<tax_id>/`` subfolder there.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical audit-CSV statuses, mirroring tools/harvest_top_images.py. The
# per-observation outcomes become their own columns (in this order); the
# terminal markers are surfaced in the single "finish" column instead.
OBSERVATION_STATUSES = [
    "taken",
    "no_photo",
    "rejected_non_cc",
    "rejected_stage_no_class",
    "rejected_stage_quota_full",
    "download_failed",
    "rejected_no_pose",
    "rejected_not_top_down",
    "rejected_low_score",
]
TERMINATION_STATUSES = [
    "done",
    "no_more_observations",
    "reached_scan_limit",
    "corrupted",
]

PRESENT_MARK = "\u2713"  # check mark
ABSENT_MARK = ""
DASH = "\u2014"  # em dash: source data unavailable (no CSV)


def bootstrap_django():
    """Set up Django from the ambient environment; return ``moths.utils``.

    Unlike the harvest/predict tools, this reads the ``MOTHS_*`` paths straight
    from the environment (never overriding them), so it reports on whatever
    dataset the web app is configured for.
    """
    sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "moths_list.settings")

    import django

    django.setup()

    from moths import utils as moth_utils  # noqa: E402  (after django.setup)

    return moth_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print the Django index table (per-tax_id counts + harvest source "
            "data) as a Markdown report, with a column marking which tax_ids "
            "also exist in a second images directory."
        )
    )
    parser.add_argument(
        "other_images",
        type=Path,
        help=(
            "Path to a second images directory. The 'present' column is a check "
            "mark when a <tax_id>/ subfolder exists there."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write the Markdown to this file instead of stdout.",
    )
    return parser.parse_args()


def _blank_last_key(row: dict) -> tuple:
    """Sort key matching the index view: by family, then species, blanks last."""
    family = row.get("family") or ""
    species = row.get("species") or ""
    return (family == "", family.lower(), species == "", species.lower())


def collect_rows(moth_utils, other_images: Path):
    """Return ``(rows, extra_statuses)`` for every tax_id, index-view style.

    Each row is a dict of already-stringified cell values keyed by column id.
    ``extra_statuses`` holds any audit statuses seen in the data that are not in
    the canonical :data:`OBSERVATION_STATUSES` / :data:`TERMINATION_STATUSES`
    lists, so the caller can still give them columns.
    """
    from moths.utils import SOURCE_STAGES

    rows: list[dict] = []
    extra_statuses: set[str] = set()

    for tax_id in moth_utils.list_tax_ids():
        summary = moth_utils.load_summary(tax_id)
        if summary is None:
            summary = moth_utils.build_summary(tax_id)

        names = summary.get("names", {})
        counts = summary.get("counts", {})
        source = summary.get("source_data")
        has_source = source is not None
        src_stages = (source or {}).get("stages", {})
        src_statuses = (source or {}).get("statuses", {})

        extra_statuses.update(
            status
            for status in src_statuses
            if status not in OBSERVATION_STATUSES
            and status not in TERMINATION_STATUSES
        )

        def src_count(value_map, key):
            return DASH if not has_source else str(value_map.get(key, 0))

        finish = DASH
        if has_source:
            finish = next(
                (m for m in TERMINATION_STATUSES if src_statuses.get(m)), ""
            )

        present = (
            PRESENT_MARK
            if (other_images / tax_id).is_dir()
            else ABSENT_MARK
        )

        row = {
            "tax_id": tax_id,
            "family": names.get("family", ""),
            "species": names.get("species", ""),
            "total": str(counts.get("total", 0)),
            "finish": finish,
            "present": present,
            "_family_raw": names.get("family", ""),
            "_species_raw": names.get("species", ""),
        }
        for stage in SOURCE_STAGES:
            row[f"src:{stage}"] = src_count(src_stages, stage)
        for status in OBSERVATION_STATUSES:
            row[f"st:{status}"] = src_count(src_statuses, status)

        rows.append(row)

    rows.sort(
        key=lambda r: (
            r["_family_raw"] == "",
            r["_family_raw"].lower(),
            r["_species_raw"] == "",
            r["_species_raw"].lower(),
        )
    )
    return rows, sorted(extra_statuses)


def build_columns(moth_utils, extra_statuses):
    """Return the ordered column specs ``(key, header, align)`` for the table.

    The class-stage / box columns (Egg/Larva/Pupa/Adult/none/labeled/not
    labeled) are intentionally omitted — those live in the Django index view.
    ``taken`` is moved to the end of the status columns, right before ``finish``.
    """
    from moths.utils import SOURCE_STAGES

    columns = [
        ("tax_id", "tax_id", "l"),
        ("family", "family", "l"),
        ("species", "species", "l"),
        ("total", "total", "r"),
    ]
    columns += [(f"src:{s}", f"src:{s.lower()}", "r") for s in SOURCE_STAGES]
    status_order = [s for s in OBSERVATION_STATUSES if s != "taken"]
    status_order += list(extra_statuses)
    status_order.append("taken")  # moved to the end, before 'finish'
    for status in status_order:
        columns.append((f"st:{status}", status, "r"))
    columns.append(("finish", "finish", "l"))
    columns.append(("present", "present", "l"))
    return columns


def render_table(columns, rows) -> str:
    """Render an aligned Markdown table from column specs and stringified rows.

    Columns are padded to a common width so the raw text stays readable; the
    ``:`` positions in the separator encode left/right alignment.
    """
    def cell(row, key):
        return str(row.get(key, "")).replace("|", "\\|")

    widths = {}
    for key, header, _align in columns:
        widths[key] = max(len(header), *(len(cell(r, key)) for r in rows)) if rows else len(header)

    def pad(text, key, align):
        return text.rjust(widths[key]) if align == "r" else text.ljust(widths[key])

    header = "| " + " | ".join(pad(h, k, "l") for k, h, _a in columns) + " |"
    sep = (
        "| "
        + " | ".join(
            ("-" * (widths[k] - 1) + ":") if a == "r" else ("-" * widths[k])
            for k, _h, a in columns
        )
        + " |"
    )
    body = [
        "| " + " | ".join(pad(cell(r, k), k, a) for k, _h, a in columns) + " |"
        for r in rows
    ]
    return "\n".join([header, sep, *body])


def main() -> None:
    args = parse_args()
    moth_utils = bootstrap_django()

    other_images = args.other_images
    if not other_images.is_dir():
        print(
            f"Warning: comparison images dir does not exist: {other_images} "
            f"(all 'present' cells will be blank)",
            file=sys.stderr,
        )

    rows, extra_statuses = collect_rows(moth_utils, other_images)
    columns = build_columns(moth_utils, extra_statuses)
    table = render_table(columns, rows)

    present_count = sum(1 for r in rows if r["present"] == PRESENT_MARK)
    lines = [
        "# Moth dataset quality summary",
        "",
        f"- Image directory: `{moth_utils.get_image_dir()}`",
        f"- Labels directory: `{moth_utils.get_label_dir()}`",
        f"- Comparison images directory: `{other_images}`",
        f"- Taxa: {len(rows)} "
        f"({present_count} also present in the comparison directory)",
        "",
        "Legend: `total` = images in the tax · `src:*` = harvest source-data "
        "life stages · remaining columns = audit-CSV statuses · `finish` = "
        f"termination marker · `present` = `{PRESENT_MARK}` when the tax_id has "
        "an images subfolder in the comparison directory · `" + DASH + "` = no harvest "
        "CSV for that tax. (Class-stage and box-label counts live in the Django "
        "index view.)",
        "",
        table,
        "",
    ]
    output = "\n".join(lines)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
        print(output)


if __name__ == "__main__":
    main()
