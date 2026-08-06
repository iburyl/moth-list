"""Per-tax summary cache, the hierarchical taxon tree and edit/delete ops."""
from __future__ import annotations

import csv
import json

from pathlib import Path

from .paths import (
    _tax_subdir,
    _unlink_quiet,
    get_image_dir,
    get_image_path,
    get_thumbnail_dir,
    image_basename,
    scan_tax_images,
    tax_id_for_file,
)
from .names import (
    get_name_info,
    get_observations_path,
    load_names,
)
from .classes import (
    STAGES,
    get_class_path,
    load_starred,
)
from .annotations import (
    POSE_BOTTOM_UP,
    POSE_SIDE,
    POSE_TOP_DOWN,
    POSE_UNCLEAR,
    classify_annotation,
    get_class_and_flags_with_source,
    get_label_dir,
    get_label_path,
    get_prediction_class_path,
    get_prediction_path,
    load_pose_source,
)
from .normalize import clear_normalized
from .posedata import (
    _write_pose_data,
    choose_tax_thumbnail,
    load_pose_data_raw,
)


# --- Per-tax index summary cache ---------------------------------------------

# Bump when the summary schema/counts change so old caches are ignored.
# v2: added "source_data" (harvest audit CSV life-stage / status tallies).
# v3: counts are plain totals (train/val split removed from the app).
# v4: class counts combine label+prediction (+ per-stage source split); the
#     "labels" group now tracks hand-annotation gaps (no_stage / no_box).
# v5: added per-pose "views" counts (+ per-pose source split) for the index.
# v6: names now include superfamily/subfamily (index columns + sort).
SUMMARY_VERSION = 6

# Bump when the hierarchical tax_summary.json schema changes so old files are
# ignored (a version mismatch makes soft updates a no-op until a full rebuild).
# v2: every node carries a representative ``thumbnail`` (best descendant image).
# v3: nodes also aggregate species_have_folder / no_stage / no_box / obs (for the
#     browse columns), and leaves cache those per-species from CSV + summaries.
# v4: an absent subfamily is keyed "-" ("not subdivided"), distinct from the
#     "(unknown)" bucket used for a genuinely missing higher rank.
# v5: nodes also aggregate ``images`` (total images under the node) for the
#     browse "images" column.
TAX_TREE_VERSION = 5

# Viewpoint buckets shown in the index "view" column group, in display order.
VIEW_POSES = [POSE_TOP_DOWN, POSE_SIDE, POSE_BOTTOM_UP, POSE_UNCLEAR]

# Life-stage buckets for the harvest "source data" summary, in display order.
# "Adult" also covers Teneral (a freshly emerged adult); anything else
# documented (Nymph, Juvenile, Subimago, numeric ids) or blank -> "Unknown".
SOURCE_STAGES = ["Egg", "Larva", "Pupa", "Adult", "Unknown"]


def get_summary_path(tax_id: str) -> Path:
    """Path to the cached index-summary JSON for a tax_id (at the labels root)."""
    return get_label_dir() / f"{tax_id}_summary.json"


def _compute_summary_counts(images) -> dict:
    """Tally the index counts (plain totals), cached in the summary.

    Per-stage counts (``stages``) combine hand labels and predictions: an
    image's stage is its hand class if set, otherwise its predicted class. For
    each stage, ``stage_sources`` records how many of those came from a hand
    label vs a prediction so the index can colour the number by provenance.

    The "view" group counts each image's viewpoint (top-down/side/bottom-up/
    unclear) from its first pose object, regardless of flags or stage, with a
    matching ``view_sources`` split (hand label vs prediction) for colouring.

    The "labels" group tracks hand-annotation gaps: ``no_stage`` (no hand stage
    class) and ``no_box`` (no hand label file).
    """
    stages = {stage: 0 for stage in STAGES}
    stage_sources = {stage: {"label": 0, "prediction": 0} for stage in STAGES}
    views = {pose: 0 for pose in VIEW_POSES}
    view_sources = {pose: {"label": 0, "prediction": 0} for pose in VIEW_POSES}
    no_stage = 0
    no_box = 0
    for image in images:
        filename = image.filename
        stage, _flags, source = get_class_and_flags_with_source(filename)
        if stage in stages:
            stages[stage] += 1
            stage_sources[stage]["label" if source == "class" else "prediction"] += 1
        # The "labels" group counts what still lacks a HAND annotation.
        if not (source == "class" and stage in stages):
            no_stage += 1
        if not get_label_path(filename).is_file():
            no_box += 1
        # The "view" group counts viewpoints from the pose keypoints (any stage).
        annotations, pose_source = load_pose_source(filename)
        if annotations:
            pose = classify_annotation(annotations[0])
            if pose in views:
                views[pose] += 1
                key = "label" if pose_source == "pose" else "prediction"
                view_sources[pose][key] += 1
    return {
        "total": len(images),
        "stages": stages,
        "stage_sources": stage_sources,
        "views": views,
        "view_sources": view_sources,
        "no_stage": no_stage,
        "no_box": no_box,
    }


def _source_stage_bucket(life_stage: str) -> str:
    """Map a documented iNaturalist life stage string to a SOURCE_STAGES bucket.

    ``life_stage`` may list several stages joined by ``/`` (as written by the
    harvest audit log). Adult/Teneral win, then Larva, Pupa, Egg; anything else
    (Nymph, Juvenile, Subimago, numeric ids, blank) falls into "Unknown".
    """
    parts = [p.strip() for p in (life_stage or "").split("/") if p.strip()]
    if any(p in ("Adult", "Teneral") for p in parts):
        return "Adult"
    for stage in ("Larva", "Pupa", "Egg"):
        if stage in parts:
            return stage
    return "Unknown"


def compute_source_data(tax_id: str) -> dict | None:
    """Summarize the harvest audit CSV ``images/<tax>_observations_list.csv``.

    Returns ``{"stages": {stage: count}, "statuses": {status: count}}`` where
    life-stage counts come from real observation rows (those with an
    ``observation_id``) and status counts cover every row's ``status`` flag
    (including terminal markers like ``done``/``corrupted``). Returns ``None``
    when the CSV is missing or unreadable, so the index can show a dash.
    """
    path = get_image_dir() / f"{tax_id}_observations_list.csv"
    if not path.is_file():
        return None
    stages = {stage: 0 for stage in SOURCE_STAGES}
    statuses: dict[str, int] = {}
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                status = (row.get("status") or "").strip()
                if status:
                    statuses[status] = statuses.get(status, 0) + 1
                # Rows with an observation_id are real observations; terminal
                # marker rows (done/corrupted/...) carry no life stage.
                if (row.get("observation_id") or "").strip():
                    bucket = _source_stage_bucket(row.get("life_stage") or "")
                    stages[bucket] += 1
    except OSError:
        return None
    return {"stages": stages, "statuses": statuses}


def _write_summary(tax_id: str, data: dict) -> None:
    """Write a summary dict to the tax_id's summary JSON (best effort)."""
    path = get_summary_path(tax_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError:
        pass


def build_summary(tax_id: str, images=None, update_tree: bool = True) -> dict:
    """Compute and cache the index summary (names + counts) for a tax_id.

    Reuses the previously stored representative-thumbnail choice when present;
    otherwise (e.g. after a schema bump wipes the old summary) it re-derives it
    from the cached pose data, so a rebuild doesn't drop the thumbnail.

    When ``update_tree`` is set (the default, used by the web app) the change is
    folded into the hierarchical ``labels/tax_summary.json`` via a cheap soft
    update. Bulk callers that rebuild many taxa (``tools/rebuild_poses.py``)
    pass ``update_tree=False`` and rebuild that aggregate once at the end.
    """
    if images is None:
        images = scan_tax_images(tax_id)
    info = get_name_info(tax_id)
    existing = load_summary(tax_id)
    thumbnail = (existing or {}).get("thumbnail") or choose_tax_thumbnail(tax_id)
    data = {
        "version": SUMMARY_VERSION,
        "names": {
            "superfamily": info["superfamily"],
            "family": info["family"],
            "subfamily": info["subfamily"],
            "species": info["species"],
            "name": info["name"],
        },
        "counts": _compute_summary_counts(images),
        "source_data": compute_source_data(tax_id),
        "thumbnail": thumbnail or None,
    }
    _write_summary(tax_id, data)
    if update_tree:
        soft_update_tax_summary(tax_id)
    return data


def load_summary(tax_id: str) -> dict | None:
    """Return the cached summary for a tax_id, or ``None`` if missing/stale."""
    path = get_summary_path(tax_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(data, dict) or data.get("version") != SUMMARY_VERSION:
        return None
    if not isinstance(data.get("counts"), dict):
        return None
    return data


def get_summary(tax_id: str, images=None) -> dict:
    """Return the cached summary for a tax_id, building it once if absent."""
    cached = load_summary(tax_id)
    if cached is not None:
        return cached
    return build_summary(tax_id, images)


def update_summary(tax_id: str | None) -> None:
    """Rebuild a tax_id's cached summary from current files (after an edit)."""
    if tax_id:
        build_summary(tax_id)


# --- Hierarchical taxonomy summary (labels/tax_summary.json) -----------------
#
# One aggregate file rolls the ~900 per-species summaries up the taxonomy so a
# future hierarchical index can render without walking every species. Structure
# (nested "children" maps): superfamily -> family -> subfamily -> genus ->
# species leaf. The "want to cover" universe is every row of the names CSV;
# genus is the first word of that row's ``species`` field. A species is
# considered "have" once its per-tax summary JSON exists, and it "has an image"
# when that summary reports ``counts.total > 0``.
#
# Every non-leaf node carries the same aggregate counts, all about its direct
# children ("next level") and the species beneath it:
#   want                - number of direct child entities we want to cover
#   have_any            - direct children with >=1 species summary
#   have_all            - direct children where every wanted species has a summary
#   species_want        - species we want to cover in this branch
#   species_have_image  - species with >=1 image
#   species_have_zero   - species with a summary but zero images
#   species_have_folder - species with an image subfolder on disk
#   no_stage / no_box   - summed hand-annotation gaps across those species
#   obs                 - summed iNaturalist observation counts (from the CSV)
# Genus nodes add ``species_want_ids`` / ``species_have_ids`` (the requested
# per-species iNaturalist id lists); their children are species leaves
# (``{species, name, have, has_image, has_folder, thumbnail, no_stage, no_box,
# obs}``). The root mirrors the aggregate counts across all superfamilies.
#
# Every node (and species leaf) also carries a ``thumbnail`` — the per-species
# representative image dict (``{filename, score, starred}``) chosen by
# :func:`choose_tax_thumbnail`. A non-leaf node inherits the best thumbnail of
# its descendants (starred first, then highest score), so the hierarchical index
# can show a picture at every level without touching the per-tax summaries.

_UNKNOWN_TAXON = "(unknown)"
# Subfamily key for species whose family is not subdivided into subfamilies.
# Distinct from ``_UNKNOWN_TAXON``: the rank is simply absent, not unknown. It
# is shown as "-" and travels in browse URLs as the shared "_" placeholder.
_NO_SUBFAMILY = "-"


def get_tax_summary_path() -> Path:
    """Path to the hierarchical taxonomy summary JSON (at the labels root)."""
    return get_label_dir() / "tax_summary.json"


def _tax_lineage_keys(info: dict) -> tuple[str, str, str, str]:
    """Return (superfamily, family, subfamily, genus) keys for a names row.

    Genus is the first whitespace-delimited token of the ``species`` field
    (assumed to be a binomial). A blank superfamily/family/genus collapses to
    ``(unknown)``; a blank subfamily collapses to ``-`` ("not subdivided") so a
    row with partial taxonomy still lands in a stable bucket.
    """
    species = (info.get("species") or "").strip()
    genus = species.split()[0] if species else ""
    return (
        info.get("superfamily") or _UNKNOWN_TAXON,
        info.get("family") or _UNKNOWN_TAXON,
        # An absent subfamily is "not subdivided" ("-"), not "(unknown)".
        info.get("subfamily") or _NO_SUBFAMILY,
        genus or _UNKNOWN_TAXON,
    )


def _species_summary_state(tax_id: str) -> dict:
    """Read a species' cached summary into the fields the tree aggregates.

    Returns ``{have, has_image, thumbnail, images, no_stage, no_box}``: ``have``
    is whether the summary JSON exists (version-agnostic), ``has_image`` reads
    ``counts.total`` (0 / unreadable -> False), ``images`` is that same total,
    ``thumbnail`` is the stored representative-image dict (or ``None``), and
    ``no_stage`` / ``no_box`` are the hand-annotation gap counts (0 when absent).
    """
    blank = {
        "have": False,
        "has_image": False,
        "thumbnail": None,
        "images": 0,
        "no_stage": 0,
        "no_box": 0,
    }
    path = get_summary_path(tax_id)
    if not path.is_file():
        return blank
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {**blank, "have": True}
    if not isinstance(raw, dict):
        return {**blank, "have": True}
    counts = raw.get("counts") if isinstance(raw.get("counts"), dict) else {}

    def _int(value):
        try:
            return int(value or 0)
        except (ValueError, TypeError):
            return 0

    thumbnail = raw.get("thumbnail")
    if not (isinstance(thumbnail, dict) and thumbnail.get("filename")):
        thumbnail = None
    return {
        "have": True,
        "has_image": _int(counts.get("total")) > 0,
        "thumbnail": thumbnail,
        "images": _int(counts.get("total")),
        "no_stage": _int(counts.get("no_stage")),
        "no_box": _int(counts.get("no_box")),
    }


def _parse_obs(value) -> int:
    """Parse the names-CSV ``obs`` field (iNat observation count) to an int."""
    try:
        return int(str(value).strip() or 0)
    except (ValueError, TypeError):
        return 0


def _build_species_leaf(tax_id: str, info: dict) -> dict:
    """Assemble a species leaf from the names CSV row + its cached summary.

    Centralises leaf construction so the full rebuild and the soft update store
    identical fields. ``has_folder`` reflects whether the tax has an image
    subfolder on disk (the "existing folders" count); ``obs`` comes from the CSV.
    """
    state = _species_summary_state(tax_id)
    return {
        "species": info.get("species") or "",
        "name": info.get("name") or "",
        "have": state["have"],
        "has_image": state["has_image"],
        "has_folder": (get_image_dir() / str(tax_id)).is_dir(),
        "thumbnail": state["thumbnail"],
        "images": state["images"],
        "no_stage": state["no_stage"],
        "no_box": state["no_box"],
        "obs": _parse_obs(info.get("obs")),
    }


def _pick_thumbnail(thumbs) -> dict | None:
    """Return the best thumbnail dict from an iterable (starred, then score)."""
    best = None
    best_key = None
    for thumb in thumbs:
        if not (isinstance(thumb, dict) and thumb.get("filename")):
            continue
        score = thumb.get("score")
        key = (bool(thumb.get("starred")), score if score is not None else float("-inf"))
        if best is None or key > best_key:
            best, best_key = thumb, key
    return best


def _genus_counts(species_map: dict) -> dict:
    """Roll a genus's species leaves into its count fields (incl. id lists)."""
    want_ids = sorted(species_map)
    have_ids = sorted(t for t, leaf in species_map.items() if leaf.get("have"))
    have_image = sum(1 for leaf in species_map.values() if leaf.get("has_image"))
    have_zero = sum(
        1
        for leaf in species_map.values()
        if leaf.get("have") and not leaf.get("has_image")
    )
    have_folder = sum(1 for leaf in species_map.values() if leaf.get("has_folder"))
    images = sum(leaf.get("images", 0) for leaf in species_map.values())
    no_stage = sum(leaf.get("no_stage", 0) for leaf in species_map.values())
    no_box = sum(leaf.get("no_box", 0) for leaf in species_map.values())
    obs = sum(leaf.get("obs", 0) for leaf in species_map.values())
    want = len(species_map)
    have = len(have_ids)
    return {
        # For a genus the "next level entities" are the species themselves, so
        # have_any / have_all both equal the number of species with a summary.
        "want": want,
        "have_any": have,
        "have_all": have,
        "species_want": want,
        "species_have_image": have_image,
        "species_have_zero": have_zero,
        "species_have_folder": have_folder,
        "images": images,
        "no_stage": no_stage,
        "no_box": no_box,
        "obs": obs,
        "species_want_ids": want_ids,
        "species_have_ids": have_ids,
    }


def _aggregate_children(children: dict) -> dict:
    """Roll a set of child nodes up into a parent's aggregate count fields."""
    want = len(children)
    have_any = have_all = 0
    species_want = species_have_image = species_have_zero = 0
    species_have_folder = images = no_stage = no_box = obs = 0
    for child in children.values():
        child_have = child["species_have_image"] + child["species_have_zero"]
        if child_have > 0:
            have_any += 1
        if child["species_want"] > 0 and child_have == child["species_want"]:
            have_all += 1
        species_want += child["species_want"]
        species_have_image += child["species_have_image"]
        species_have_zero += child["species_have_zero"]
        species_have_folder += child.get("species_have_folder", 0)
        images += child.get("images", 0)
        no_stage += child.get("no_stage", 0)
        no_box += child.get("no_box", 0)
        obs += child.get("obs", 0)
    return {
        "want": want,
        "have_any": have_any,
        "have_all": have_all,
        "species_want": species_want,
        "species_have_image": species_have_image,
        "species_have_zero": species_have_zero,
        "species_have_folder": species_have_folder,
        "images": images,
        "no_stage": no_stage,
        "no_box": no_box,
        "obs": obs,
    }


def _genus_node_from(species_map: dict) -> dict:
    """Assemble a genus node (counts + id lists + thumbnail) from its species."""
    return {
        **_genus_counts(species_map),
        "thumbnail": _pick_thumbnail(
            leaf.get("thumbnail") for leaf in species_map.values()
        ),
        "children": species_map,
    }


def _parent_node_from(children: dict) -> dict:
    """Assemble a superfamily/family/subfamily node from its child nodes."""
    return {
        **_aggregate_children(children),
        "thumbnail": _pick_thumbnail(
            child.get("thumbnail") for child in children.values()
        ),
        "children": children,
    }


def _refresh_genus_node(node: dict) -> None:
    """Recompute a genus node's counts + thumbnail from its species leaves."""
    node.update(_genus_counts(node["children"]))
    node["thumbnail"] = _pick_thumbnail(
        leaf.get("thumbnail") for leaf in node["children"].values()
    )


def _refresh_parent_node(node: dict) -> None:
    """Recompute a parent node's counts + thumbnail from its child nodes."""
    node.update(_aggregate_children(node["children"]))
    node["thumbnail"] = _pick_thumbnail(
        child.get("thumbnail") for child in node["children"].values()
    )


def _empty_parent_node() -> dict:
    return {
        "want": 0,
        "have_any": 0,
        "have_all": 0,
        "species_want": 0,
        "species_have_image": 0,
        "species_have_zero": 0,
        "species_have_folder": 0,
        "images": 0,
        "no_stage": 0,
        "no_box": 0,
        "obs": 0,
        "thumbnail": None,
        "children": {},
    }


def _empty_genus_node() -> dict:
    return {**_genus_counts({}), "thumbnail": None, "children": {}}


def load_tax_summary() -> dict | None:
    """Return the cached hierarchical summary, or ``None`` if missing/stale."""
    path = get_tax_summary_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(data, dict) or data.get("version") != TAX_TREE_VERSION:
        return None
    return data


def _write_tax_summary(data: dict) -> None:
    """Write the hierarchical summary JSON (best effort)."""
    path = get_tax_summary_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError:
        pass


def build_tax_summary() -> dict:
    """Rebuild ``labels/tax_summary.json`` from the names CSV + per-tax summaries.

    The full ("complete") rebuild path: every names-CSV row is placed in the
    superfamily/family/subfamily/genus/species tree and its ``have`` /
    ``has_image`` state is read from ``labels/{id}_summary.json``. Used by
    ``tools/rebuild_poses.py``.
    """
    names = load_names()
    tree: dict = {}
    for tax_id, info in names.items():
        sf, fam, subf, genus = _tax_lineage_keys(info)
        leaf = _build_species_leaf(tax_id, info)
        genus_map = (
            tree.setdefault(sf, {})
            .setdefault(fam, {})
            .setdefault(subf, {})
            .setdefault(genus, {})
        )
        genus_map[str(tax_id)] = leaf

    superfamilies: dict = {}
    for sf, fams in tree.items():
        fam_nodes: dict = {}
        for fam, subfs in fams.items():
            subf_nodes: dict = {}
            for subf, genera in subfs.items():
                genus_nodes: dict = {}
                for genus, species_map in genera.items():
                    genus_nodes[genus] = _genus_node_from(species_map)
                subf_nodes[subf] = _parent_node_from(genus_nodes)
            fam_nodes[fam] = _parent_node_from(subf_nodes)
        superfamilies[sf] = _parent_node_from(fam_nodes)

    data = {
        "version": TAX_TREE_VERSION,
        **_aggregate_children(superfamilies),
        "thumbnail": _pick_thumbnail(
            node.get("thumbnail") for node in superfamilies.values()
        ),
        "superfamilies": superfamilies,
    }
    _write_tax_summary(data)
    return data


def soft_update_tax_summary(tax_id: str) -> None:
    """Fold one species' state into the cached hierarchical summary in place.

    Cheap counterpart to :func:`build_tax_summary` used by the web app: it reads
    the aggregate once, rebuilds just this species' leaf from its (freshly
    written) summary + the CSV, and re-rolls only its superfamily/family/
    subfamily/genus lineage (plus the root), re-picking each level's
    representative thumbnail on the way up. No-ops when the aggregate is
    missing/stale (a full rebuild will regenerate it) or the tax_id is absent
    from the names CSV, and skips the write when nothing changed.
    """
    data = load_tax_summary()
    if data is None:
        return
    info = load_names().get(str(tax_id))
    if not info:
        return

    sf, fam, subf, genus = _tax_lineage_keys(info)
    superfamilies = data.setdefault("superfamilies", {})
    sf_node = superfamilies.setdefault(sf, _empty_parent_node())
    fam_node = sf_node["children"].setdefault(fam, _empty_parent_node())
    subf_node = fam_node["children"].setdefault(subf, _empty_parent_node())
    genus_node = subf_node["children"].setdefault(genus, _empty_genus_node())

    new_leaf = _build_species_leaf(tax_id, info)
    if genus_node["children"].get(str(tax_id)) == new_leaf:
        return
    genus_node["children"][str(tax_id)] = new_leaf

    _refresh_genus_node(genus_node)
    _refresh_parent_node(subf_node)
    _refresh_parent_node(fam_node)
    _refresh_parent_node(sf_node)
    data.update(_aggregate_children(superfamilies))
    data["thumbnail"] = _pick_thumbnail(
        node.get("thumbnail") for node in superfamilies.values()
    )
    _write_tax_summary(data)


# --- Image deletion ----------------------------------------------------------


def delete_image_files(image_filename: str) -> bool:
    """Physically delete an image and every file derived from it.

    Removes the source image, its hand label (``.txt``) and stage class
    (``.class``), the model prediction (``.txt`` + ``.class``) in the
    prediction/test directory, and all cached derivatives (the plain thumbnail
    and the normalized crop + thumbnail, every version). Best-effort: missing
    files are ignored. Returns ``True`` if the source image existed.
    """
    name = image_basename(image_filename)
    existed = get_image_path(name).is_file()
    _unlink_quiet(get_image_path(name))
    _unlink_quiet(get_label_path(name))
    _unlink_quiet(get_class_path(name))
    _unlink_quiet(get_prediction_path(name))
    _unlink_quiet(get_prediction_class_path(name))
    # Cached derivatives: the plain list-view thumbnail + the normalized crops.
    _unlink_quiet(_tax_subdir(get_thumbnail_dir().resolve(), name) / name)
    clear_normalized(name)
    return existed


def prune_pose_data(tax_id: str, removed_names) -> None:
    """Drop entries for ``removed_names`` from the tax's cached pose data JSON."""
    removed = set(removed_names)
    if not removed:
        return
    data = load_pose_data_raw(tax_id)
    if not data:
        return
    images = data.get("images", {})
    kept = {name: row for name, row in images.items() if name not in removed}
    if len(kept) != len(images):
        data["images"] = kept
        _write_pose_data(tax_id, data)


def prune_observations(tax_id: str) -> None:
    """Drop observation entries with no remaining image on disk for the tax.

    An observation can have several images; its metadata entry is kept as long
    as at least one of its images still exists (only fully-removed observations
    are dropped). Rewrites ``images/<tax>_observations.json`` in place.
    """
    path = get_observations_path(tax_id)
    if not path.is_file():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return
    if not isinstance(raw, list):
        return
    present = {image.obs_id for image in scan_tax_images(tax_id)}
    kept = [
        item
        for item in raw
        if isinstance(item, dict)
        and str(item.get("observation_id")) in present
    ]
    if len(kept) != len(raw):
        try:
            path.write_text(json.dumps(kept, indent=1), encoding="utf-8")
        except OSError:
            pass


def delete_images(tax_id: str, filenames) -> dict:
    """Delete images (and all their files) for a tax, skipping starred ones.

    For each requested image that belongs to ``tax_id`` and is *not* starred,
    physically removes the image plus its label/class/prediction/cache files
    (see :func:`delete_image_files`). Afterwards the observation and pose-data
    caches are pruned, the representative thumbnail is re-chosen if it pointed
    at a deleted image, and the summary is rebuilt. Returns
    ``{"deleted": int, "skipped_starred": int}``.
    """
    starred = load_starred(tax_id)
    removed: list[str] = []
    deleted = 0
    skipped_starred = 0
    for filename in filenames:
        name = image_basename(filename)
        if tax_id_for_file(name) != tax_id:
            continue
        if name in starred:
            skipped_starred += 1
            continue
        if delete_image_files(name):
            deleted += 1
        removed.append(name)

    if not removed:
        return {"deleted": 0, "skipped_starred": skipped_starred}

    prune_pose_data(tax_id, removed)
    prune_observations(tax_id)

    # If the cached representative thumbnail was one of the deleted images,
    # re-choose it (possibly None) before rebuilding, so the index never points
    # at a now-missing file.
    removed_set = set(removed)
    summary = load_summary(tax_id)
    if summary and (summary.get("thumbnail") or {}).get("filename") in removed_set:
        summary["thumbnail"] = choose_tax_thumbnail(tax_id)
        _write_summary(tax_id, summary)

    build_summary(tax_id)
    return {"deleted": deleted, "skipped_starred": skipped_starred}


def _fmt(value: float) -> str:
    return f"{float(value):.6f}"


def save_annotations(image_filename: str, objects: list[dict]) -> None:
    """Write annotations to the image's YOLO-pose label file.

    ``objects`` is a list of dicts with keys ``class_id``, ``cx``, ``cy``,
    ``w``, ``h`` and ``keypoints`` (a list of ``{"x", "y", "v"}``). If the list
    is empty, the label file is removed.
    """
    label_path = get_label_path(image_filename)
    if not objects:
        if label_path.is_file():
            label_path.unlink()
        return

    lines = []
    for obj in objects:
        values = [
            str(int(obj.get("class_id", 0))),
            _fmt(obj["cx"]),
            _fmt(obj["cy"]),
            _fmt(obj["w"]),
            _fmt(obj["h"]),
        ]
        for kp in obj.get("keypoints", []):
            values += [_fmt(kp["x"]), _fmt(kp["y"]), str(int(kp["v"]))]
        lines.append(" ".join(values))

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
