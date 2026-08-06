import json

from django.http import (
    FileResponse,
    Http404,
    JsonResponse,
)
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from .utils import (
    _NO_SUBFAMILY,
    _UNKNOWN_TAXON,
    FLAG_TABLE,
    KEYPOINT_LABELS,
    POSE_BOTTOM_UP,
    flag_applies_to_stage,
    flags_for_stage,
    flags_suppress_normalization,
    POSE_NONE,
    POSE_SIDE,
    POSE_TOP_DOWN,
    POSE_UNCLEAR,
    STAGES,
    VIEW_POSES,
    build_pose_data,
    build_summary,
    classify_pose,
    compute_normalization,
    clear_image_class,
    delete_images,
    find_image,
    get_class_and_flags,
    get_class_and_flags_with_source,
    get_image_class,
    get_image_flags,
    get_image_dir,
    get_image_path,
    get_label_path,
    get_name_info,
    get_image_details,
    get_observation_info,
    get_or_create_normalized,
    get_or_create_thumbnail,
    get_predicted_class_and_flags,
    get_predicted_pose_class,
    get_side_wing_stats_path,
    get_tax_thumbnail,
    get_wing_stats_path,
    group_by_tax_id,
    is_image_starred,
    list_tax_ids,
    load_annotations,
    load_pose_data,
    load_pose_data_raw,
    load_predictions,
    load_starred,
    load_summary,
    load_tax_summary,
    mark_pose_row_stale,
    pose_data_version_ok,
    refresh_tax_thumbnail,
    save_annotations,
    score_components,
    set_image_class,
    set_image_details,
    set_image_flags,
    set_image_starred,
    set_pose_row_flags,
    tax_id_for_file,
    update_summary,
    verify_pose_row,
)

# Unified stage/pose/flag groups. A flagged image lands in a flag subsection per
# flag it carries (so it can appear in more than one and is removed from its
# pose/stage subsection); unflagged images fall into their stage/pose subsection.
# Adults are split by predicted pose; other stages have a single base group;
# images without a stage class fall into "unknown".
GROUP_ADULT_TOP_DOWN = "adult_top_down"
GROUP_ADULT_SIDE = "adult_side"
GROUP_ADULT_BOTTOM_UP = "adult_bottom_up"
GROUP_ADULT_UNCLEAR = "adult_unclear"
GROUP_ADULT_NONE = "adult_none"
GROUP_UNKNOWN = "unknown"

# Adult pose -> unified group id (for UNFLAGGED adults only).
ADULT_POSE_GROUP = {
    POSE_TOP_DOWN: GROUP_ADULT_TOP_DOWN,
    POSE_SIDE: GROUP_ADULT_SIDE,
    POSE_BOTTOM_UP: GROUP_ADULT_BOTTOM_UP,
    POSE_UNCLEAR: GROUP_ADULT_UNCLEAR,
    POSE_NONE: GROUP_ADULT_NONE,
}

# Poses whose images carry keypoints: metrics, normalized-view link, score sort.
KEYPOINT_POSES = {POSE_TOP_DOWN, POSE_SIDE, POSE_BOTTOM_UP, POSE_UNCLEAR}
# Poses that have a normalized crop -> use the normalized thumbnail.
NORM_POSES = {POSE_TOP_DOWN, POSE_SIDE}

# Stage keys that get their own base/flag subsections (anything else -> unknown).
STAGE_KEYS = ("Adult", "Pupa", "Larva", "Egg")


def _flag_gid(stage_key, flag):
    """Group id for a stage's flag subsection, e.g. ``adult_Pinned``."""
    return f"{stage_key}_{flag}"


# Adult base subsections, in display order: the "real" pose groups come first,
# then (spliced by :func:`_unified_group_defs`) the Adult flag subsections, then
# the degenerate pose groups.
_ADULT_POSE_HEAD = [
    (GROUP_ADULT_TOP_DOWN, "Adult: top-down view"),
    (GROUP_ADULT_SIDE, "Adult: side view"),
    (GROUP_ADULT_BOTTOM_UP, "Adult: bottom-up view"),
]
_ADULT_POSE_TAIL = [
    (GROUP_ADULT_UNCLEAR, "Adult: unclear pose"),
    (GROUP_ADULT_NONE, "Adult: no keypoints"),
]
# Display noun per stage bucket (Adult is handled separately by pose).
_STAGE_PLURAL = {"Pupa": "Pupa", "Larva": "Larvae", "Egg": "Egg"}


def _flag_subsection_defs(stage_key, plural):
    """``(group_id, label, False)`` for each flag offered for ``stage_key``.

    Driven entirely by the flag table (:func:`flags_for_stage`), so which flag
    subsections a stage gets — and their order — follows the table, with no flag
    name hardcoded here.
    """
    return [
        (_flag_gid(stage_key, flag), f"{plural}: {flag.lower()}", False)
        for flag in flags_for_stage(stage_key)
    ]


def _unified_group_defs():
    """Ordered ``(group_id, label, is_unknown_base)`` specs for all subsections.

    For adults the flag subsections sit between the real pose groups (top-down/
    side/bottom-up) and the degenerate ones (unclear/no-keypoints); other stages
    get their flag subsections right after the base group. Each stage only gets
    the flag subsections the table says apply to it. ``is_unknown_base`` marks
    the single unflagged "Unknown stage" group (the only one with bulk buttons).
    """
    defs = [(gid, label, False) for gid, label in _ADULT_POSE_HEAD]
    defs += _flag_subsection_defs("Adult", "Adult")
    defs += [(gid, label, False) for gid, label in _ADULT_POSE_TAIL]
    for stage_key, plural in _STAGE_PLURAL.items():
        defs.append((stage_key, plural, False))
        defs += _flag_subsection_defs(stage_key, plural)
    defs.append((GROUP_UNKNOWN, "Unknown stage", True))
    defs += _flag_subsection_defs(GROUP_UNKNOWN, "Unknown stage")
    return defs


def _target_group_ids(stage, flags, pose):
    """Group id(s) an image belongs to: one per applicable flag, else its group.

    Only flags the table says apply to the image's stage steer grouping; a flag
    that doesn't apply (e.g. a leftover Traces on an Adult) is ignored here so
    the image still lands in its stage/pose subsection rather than vanishing.
    """
    stage_key = stage if stage in STAGE_KEYS else "unknown"
    applicable = [f for f in flags if flag_applies_to_stage(f, stage_key)]
    if applicable:
        return [_flag_gid(stage_key, flag) for flag in applicable]
    if stage_key == "Adult":
        return [ADULT_POSE_GROUP.get(pose, GROUP_ADULT_NONE)]
    return [stage_key]

# Colors per keypoint visibility flag (0 unlabeled, 1 occluded, 2 visible).
VISIBILITY_COLORS = {0: "#9ca3af", 1: "#f59e0b", 2: "#22c55e"}


# Endpoints for the class-source colour blend on the index page: a stage count
# is green when every image's stage comes from a hand label, blue when it comes
# entirely from predictions, and a proportional blend in between.
_CLASS_LABEL_RGB = (0x16, 0xA3, 0x4A)
_CLASS_PRED_RGB = (0x25, 0x63, 0xEB)


# Display labels for the index "view" (pose) column group.
VIEW_POSE_LABELS = {
    POSE_TOP_DOWN: "top-down",
    POSE_SIDE: "side",
    POSE_BOTTOM_UP: "bottom-up",
    POSE_UNCLEAR: "unclear",
}


def _class_source_color(label_count: int, pred_count: int) -> str | None:
    """Blend green (all labels) .. blue (all predictions); ``None`` if empty."""
    total = label_count + pred_count
    if total <= 0:
        return None
    frac_pred = pred_count / total
    rgb = tuple(
        round(lo + (hi - lo) * frac_pred)
        for lo, hi in zip(_CLASS_LABEL_RGB, _CLASS_PRED_RGB)
    )
    return "#%02x%02x%02x" % rgb


def index(request):
    """Landing page: the taxonomy browser at its root (the superfamily list).

    The old flat per-tax_id table got too slow at ~900 species; the index is now
    the top level of the hierarchical browser backed by ``tax_summary.json`` (see
    :func:`browse`), which renders from the pre-aggregated file without scanning.
    """
    return browse(request, lineage="")


def _index_row(tax_id: str) -> dict:
    """Build one flat index-table row for a tax_id from its cached summary.

    Read-only: never builds here (that is the heavy path). A tax with no cached
    summary yet renders with "—" placeholders; its summary is built when the
    poses view is entered. Names/obs come straight from the (mtime-cached) names
    CSV, so they show even before a summary exists. Reused by the legacy flat
    index and by the genus-level browse page (its species listing).
    """
    summary = load_summary(tax_id)
    name_info = get_name_info(tax_id)

    counts = (summary or {}).get("counts", {})
    stages_map = counts.get("stages", {})
    stage_sources = counts.get("stage_sources", {})
    views_map = counts.get("views", {})
    view_sources = counts.get("view_sources", {})
    thumbnail = (summary or {}).get("thumbnail") or {}
    has_summary = summary is not None

    def _stage_cell(stage):
        if not has_summary:
            return {"stage": stage, "count": None, "color": None}
        return {
            "stage": stage,
            "count": stages_map.get(stage, 0),
            "color": _class_source_color(
                stage_sources.get(stage, {}).get("label", 0),
                stage_sources.get(stage, {}).get("prediction", 0),
            ),
        }

    def _view_cell(pose):
        if not has_summary:
            return {"pose": pose, "label": VIEW_POSE_LABELS[pose], "count": None, "color": None}
        return {
            "pose": pose,
            "label": VIEW_POSE_LABELS[pose],
            "count": views_map.get(pose, 0),
            "color": _class_source_color(
                view_sources.get(pose, {}).get("label", 0),
                view_sources.get(pose, {}).get("prediction", 0),
            ),
        }

    return {
        "tax_id": tax_id,
        "superfamily": name_info.get("superfamily", ""),
        "family": name_info.get("family", ""),
        "subfamily": name_info.get("subfamily", ""),
        "species": name_info.get("species", ""),
        "name": name_info.get("name", ""),
        "obs": name_info.get("obs", ""),
        "thumbnail": thumbnail.get("filename") or None,
        # None counts render as "—" (no summary yet); real counts link.
        "total": counts.get("total") if has_summary else None,
        "stage_cells": [_stage_cell(stage) for stage in STAGES],
        "view_cells": [_view_cell(pose) for pose in VIEW_POSES],
        "no_stage": counts.get("no_stage", 0) if has_summary else None,
        "no_box": counts.get("no_box", 0) if has_summary else None,
    }


def _legacy_index(request):
    """Former flat per-tax_id table (kept for reference; no longer routed)."""
    rows = [_index_row(tax_id) for tax_id in list_tax_ids()]
    # Sort by superfamily, family, subfamily, then species; unknown (blank)
    # names fall to the end at each level.
    rows.sort(
        key=lambda r: (
            r["superfamily"] == "",
            r["superfamily"].lower(),
            r["family"] == "",
            r["family"].lower(),
            r["subfamily"] == "",
            r["subfamily"].lower(),
            r["species"] == "",
            r["species"].lower(),
        )
    )
    context = {
        "rows": rows,
        "stages": STAGES,
        "total": len(rows),
        "image_dir": str(get_image_dir()),
    }
    return render(request, "moths/index.html", context)


# Taxonomy levels by descent depth: depth 0 lists superfamilies, ... depth 4
# lists the species leaves under a genus. Also the label of the children shown.
BROWSE_LEVELS = ["superfamily", "family", "subfamily", "genus", "species"]


def _browse_seg_to_key(seg: str, index: int) -> str:
    """URL segment -> tree key.

    ``"_"`` is the shared placeholder for a bucketed rank; at the subfamily
    position (index 2) it decodes to the "not subdivided" key ("-"), elsewhere
    to the "(unknown)" bucket.
    """
    if seg == "_":
        return _NO_SUBFAMILY if index == 2 else _UNKNOWN_TAXON
    return seg


def _browse_key_to_seg(key: str) -> str:
    """Tree key -> URL segment (bucket keys travel as a bare "_")."""
    return "_" if key in (_UNKNOWN_TAXON, _NO_SUBFAMILY) else key


def _browse_thumb_filename(thumbnail) -> str | None:
    """Extract the image filename from a node's stored thumbnail dict."""
    if isinstance(thumbnail, dict):
        return thumbnail.get("filename") or None
    return None


_BROWSE_COUNT_KEYS = (
    "want",
    "have_any",
    "have_all",
    "species_want",
    "species_have_image",
    "species_have_zero",
)

# Fields copied onto each taxon browse row for the table columns: next-level
# total (want), species present-vs-expected (folders of want), the total image
# count under the node, the hand-label gaps, and the summed iNaturalist
# observation count.
_BROWSE_ROW_KEYS = (
    "want",
    "species_want",
    "species_have_folder",
    "images",
    "no_stage",
    "no_box",
    "obs",
)


def browse(request, lineage=""):
    """Hierarchical taxonomy browser backed by ``labels/tax_summary.json``.

    The URL path mirrors the lineage, one taxon per segment
    (``browse/Noctuoidea/Erebidae/Arctiinae/Anathix``), with ``_`` standing in
    for the "(unknown)" bucket. Walking is a plain descent through the aggregate
    file's nested ``children`` maps — O(depth) lookups, no file scan. Each page
    lists the current node's direct children with their coverage counts; the
    species leaves under a genus link straight to the poses view (there is no
    browse page for a single species).
    """
    data = load_tax_summary()
    segs = [s for s in lineage.split("/") if s]
    keys = [_browse_seg_to_key(s, i) for i, s in enumerate(segs)]

    if data is None:
        return render(request, "moths/browse.html", {"missing": True})

    if len(keys) > 4:
        raise Http404("No browse page for a single species")

    # Descend the tree by key: the root's children live under "superfamilies",
    # every deeper node's under "children".
    node = data
    for key in keys:
        container = node["superfamilies"] if node is data else node["children"]
        child = container.get(key)
        if child is None:
            raise Http404(f"Unknown taxon path: {lineage!r}")
        node = child

    depth = len(keys)
    container = node["superfamilies"] if node is data else node["children"]

    # Collapse the subfamily level when a family isn't subdivided: if the only
    # subfamily is the "not subdivided" bucket ("-"), list its genera directly
    # (a lone "-" subfamily page would just be a pointless extra click). The
    # child URLs then carry the "_" subfamily placeholder so genus links resolve.
    url_prefix = list(segs)
    if depth == 2 and set(container) == {_NO_SUBFAMILY}:
        container = container[_NO_SUBFAMILY]["children"]
        url_prefix = segs + [_browse_key_to_seg(_NO_SUBFAMILY)]
        depth = 3

    child_level = BROWSE_LEVELS[depth]
    child_is_species = depth == 4

    children = []
    index_rows = []
    if child_is_species:
        # Species leaves render as the full flat index table (per-stage / view /
        # labels stats), reusing the legacy index row builder and template.
        index_rows = [_index_row(tax_id) for tax_id in container]
        index_rows.sort(key=lambda r: (r["species"] == "", r["species"].lower(), r["tax_id"]))
    else:
        for key in sorted(container):
            child = container[key]
            child_lineage = "/".join(url_prefix + [_browse_key_to_seg(key)])
            row = {
                "label": key,
                "url": reverse("moths:browse", args=[child_lineage]),
                "thumbnail": _browse_thumb_filename(child.get("thumbnail")),
            }
            row.update({ck: child.get(ck, 0) for ck in _BROWSE_ROW_KEYS})
            children.append(row)

    crumbs = [{"label": "All", "url": reverse("moths:browse")}]
    for i, key in enumerate(keys):
        # Skip the subfamily crumb (level index 2) when the family isn't
        # subdivided (the "-" bucket): there is no meaningful subfamily to show.
        if i == 2 and key == _NO_SUBFAMILY:
            continue
        crumbs.append(
            {
                "label": key,
                "url": reverse("moths:browse", args=["/".join(segs[: i + 1])]),
            }
        )

    # Counts for the summary line: species_* describe the current node; the
    # next-level trio (want/have_any/have_all) describes the children actually
    # listed (which, after a collapse, are genera rather than subfamilies).
    node_counts = {
        "species_want": node.get("species_want", 0),
        "species_have_image": node.get("species_have_image", 0),
        "species_have_zero": node.get("species_have_zero", 0),
        "want": len(container),
        "have_any": 0,
        "have_all": 0,
    }
    if not child_is_species:
        for child in container.values():
            have = child["species_have_image"] + child["species_have_zero"]
            if have > 0:
                node_counts["have_any"] += 1
            if child["species_want"] > 0 and have == child["species_want"]:
                node_counts["have_all"] += 1

    context = {
        "missing": False,
        "crumbs": crumbs,
        "children": children,
        "index_rows": index_rows,
        "stages": STAGES,
        "child_level": child_level,
        "child_is_species": child_is_species,
        # Label for the next-next level (the grandchildren the "want/with data/
        # complete" columns summarise); species children have no deeper level.
        "next_level": None if child_is_species else BROWSE_LEVELS[depth + 1],
        "node_counts": node_counts,
        "level_label": BROWSE_LEVELS[depth - 1] if depth else "",
    }
    return render(request, "moths/browse.html", context)


def observation_lookup(request):
    """Client-side iNaturalist observation lookup page.

    The page fetches the observation and its photos from the iNaturalist API in
    the browser (nothing is uploaded to our data), and asks
    :func:`species_info` whether we already track that taxon.
    """
    return render(request, "moths/observation_lookup.html", {})


def species_info(request, tax_id):
    """JSON: whether we track ``tax_id`` plus its starred reference image(s).

    ``starred`` lists the tax's starred images ordered like the poses view
    (highest cumulative score first); ``reference`` is the first of those (or
    the recorded tax thumbnail as a fallback when nothing is starred yet).
    """
    exists = (get_image_dir() / tax_id).is_dir()
    info = get_name_info(tax_id)

    def entry(filename):
        return {
            "filename": filename,
            "thumbnail_url": reverse("moths:serve_norm_thumbnail", args=[filename]),
            "image_url": reverse("moths:serve_norm_image", args=[filename]),
        }

    starred_entries = []
    reference = None
    if exists:
        data = load_pose_data(tax_id)
        per_image = data.get("images", {}) if data else {}
        starred = load_starred(tax_id)
        ordered = sorted(
            starred,
            key=lambda fn: (per_image.get(fn, {}).get("score") is not None,
                            per_image.get(fn, {}).get("score") or 0.0),
            reverse=True,
        )
        starred_entries = [entry(fn) for fn in ordered]
        if starred_entries:
            reference = starred_entries[0]
        else:
            thumbnail = get_tax_thumbnail(tax_id)
            if thumbnail:
                reference = entry(thumbnail)
                starred_entries = [reference]

    return JsonResponse(
        {
            "exists": exists,
            "tax_id": tax_id,
            "family": info["family"],
            "species": info["species"],
            "name": info["name"],
            "reference": reference,
            "starred": starred_entries,
            "poses_url": (
                reverse("moths:pose_view", args=[tax_id]) if exists else None
            ),
        }
    )


def _parse_filters(request):
    """Extract (stage, labeled, pose) filters from the query string."""
    return (
        request.GET.get("stage") or None,
        request.GET.get("labeled") or None,
        request.GET.get("pose") or None,
    )


def _filter_desc(stage_filter, labeled_filter, pose_filter):
    """Human-readable description of the active filters (or None)."""
    parts = []
    if stage_filter == "none":
        parts.append("unclassified")
    elif stage_filter:
        parts.append(f"stage {stage_filter}")
    if labeled_filter == "yes":
        parts.append("with labels")
    elif labeled_filter == "no":
        parts.append("without labels")
    if pose_filter:
        parts.append(f"view {VIEW_POSE_LABELS.get(pose_filter, pose_filter)}")
    return ", ".join(parts) if parts else None


def _filter_images(images, stage_filter, labeled_filter, pose_filter):
    """Apply stage/label/pose filters to a list of images."""
    result = []
    for image in images:
        if stage_filter:
            stage = get_image_class(image.filename)
            if stage_filter == "none":
                if stage in STAGES:
                    continue
            elif stage != stage_filter:
                continue
        if labeled_filter:
            has_label = get_label_path(image.filename).is_file()
            if labeled_filter == "yes" and not has_label:
                continue
            if labeled_filter == "no" and has_label:
                continue
        if pose_filter:
            if classify_pose(image.filename) != pose_filter:
                continue
        result.append(image)
    return result


def tax_detail(request, tax_id):
    """Legacy list route: now folded into the unified poses view.

    Kept so existing links (e.g. the index page's clickable counts) keep
    working; it redirects to :func:`pose_view`, preserving any active filters.
    """
    url = reverse("moths:pose_view", args=[tax_id])
    query = request.GET.urlencode()
    if query:
        url = f"{url}?{query}"
    return redirect(url)


def _pose_row_sort_key(row):
    """Default pose ordering within a group: starred first, then score desc.

    Missing scores (non-top-down groups) sort last.
    """
    score = row["score"]
    return (not row["starred"], -(score if score is not None else float("-inf")))


def _make_row(image, data, version_ok, starred, is_adult, flags=(), class_from_prediction=False):
    """Build a template row for one image from its cached pose ``data``.

    Keypoint treatment (metrics/scores, normalized thumbnail + normalized-view
    link, and score-sorting of the group) applies only to **Adult** images with
    a keypoint pose (top-down/side/bottom-up/unclear). Non-adult stages get a
    plain thumbnail and the edit link even when their only pose source is a
    leftover adult prediction — otherwise a larva/pupa would be score-sorted and
    reshuffle as its cached row is flagged for rebuild. Pose info travels on the
    row so a single flag subsection can mix images of different poses.

    A ``NO_NORM_FLAGS`` flag (e.g. "Mating") also drops the keypoint treatment:
    the image is never normalized, so it renders as a plain thumbnail linking to
    the edit view, and its subsection carries no score column.

    ``class_from_prediction`` is set when the image's stage/flags themselves came
    from the prediction folder (no hand ``.class``); it forces the blue "from
    prediction" marker on regardless of the keypoint source.
    """
    pose = data.get("pose", POSE_NONE)
    no_norm = flags_suppress_normalization(flags)
    has_keypoints = is_adult and pose in KEYPOINT_POSES and not no_norm
    row = {
        "image": image,
        "starred": image.filename in starred,
        "has_keypoints": has_keypoints,
        "is_norm": is_adult and pose in NORM_POSES and not no_norm,
        "is_top_down": is_adult and pose == POSE_TOP_DOWN and not no_norm,
        "from_prediction": class_from_prediction,
        "needs_rebuild": False,
        "metric": None,
        "sym_score": None,
        "pixel_span": None,
        "pixel_score": None,
        "sharpness": None,
        "sharpness_m": None,
        "sharp_score": None,
        "score": None,
    }
    if has_keypoints:
        needs_rebuild = (not version_ok) or bool(data.get("needs_rebuild"))
        symmetry = None if needs_rebuild else data.get("symmetry")
        pixel_span = None if needs_rebuild else data.get("pixel_span")
        sharpness = None if needs_rebuild else data.get("sharpness")
        sym_score, pixel_score, sharp_score = score_components(
            symmetry, pixel_span, sharpness
        )
        row.update(
            {
                "metric": symmetry,
                "sym_score": sym_score,
                "pixel_span": pixel_span,
                "pixel_score": pixel_score,
                "sharpness": sharpness,
                "sharpness_m": None if sharpness is None else sharpness / 1_000_000,
                "sharp_score": sharp_score,
                "score": None if needs_rebuild else data.get("score"),
                "from_prediction": class_from_prediction or (
                    not needs_rebuild and data.get("source") == "prediction"
                ),
                "needs_rebuild": needs_rebuild,
            }
        )
    return row


def _build_unified_groups(tax_id, images, image_list):
    """Return the ordered, non-empty unified groups for ``image_list``.

    Each image is placed in its flag subsection(s) when flagged, otherwise in its
    stage/pose subsection. A group's ``has_keypoints``/``is_top_down`` reflect
    whether *any* row qualifies (so mixed flag subsections still show the sort
    bar/score column). Keypoint-bearing groups sort starred-first then by score.
    Empty groups are dropped.
    """
    cached = load_pose_data_raw(tax_id)
    if cached is None and images:
        cached = build_pose_data(tax_id, images)
    version_ok = pose_data_version_ok(cached) if cached else False
    per_image = cached.get("images", {}) if cached else {}
    starred = load_starred(tax_id)

    buckets = {gid: [] for gid, _label, _unknown in _unified_group_defs()}
    for image in image_list:
        stage, flags, class_source = get_class_and_flags_with_source(image.filename)
        data = per_image.get(image.filename) or {}
        pose = data.get("pose", POSE_NONE)
        row = _make_row(
            image, data, version_ok, starred, stage == "Adult", flags,
            class_source == "prediction",
        )
        for gid in _target_group_ids(stage, flags, pose):
            if gid in buckets:
                buckets[gid].append(row)

    groups = []
    for gid, label, is_unknown_base in _unified_group_defs():
        rows = buckets[gid]
        if not rows:
            continue
        group_has_keypoints = any(r["has_keypoints"] for r in rows)
        if group_has_keypoints:
            rows = sorted(rows, key=_pose_row_sort_key)
        groups.append(
            {
                "id": gid,
                "label": label,
                "rows": rows,
                "has_keypoints": group_has_keypoints,
                # Controls the sharpness/score sort buttons + rebuild caption.
                "is_top_down": any(r["is_top_down"] for r in rows),
                "is_unknown": is_unknown_base,
            }
        )
    return groups


def pose_view(request, tax_id):
    """Unified per-tax view: images grouped by stage, adults split by pose.

    Groups (in order): Adult top-down/side/bottom-up/unclear/no-keypoints, then
    Pupa, Larvae, Egg, and finally images with no stage class ("Unknown
    stage"); empty groups are hidden. Adult-with-keypoints images link to the
    normalized view (top-down uses normalized thumbnails and shows metrics);
    all others link to the edit view. Honors the stage/labeled filters.
    If the tax_id folder exists but has no images, the page still renders with a
    "no data" note.
    """
    images = group_by_tax_id().get(tax_id) or []
    if not images and not (get_image_dir() / tax_id).is_dir():
        raise Http404(f"No images found for tax_id {tax_id!r}")

    stage_filter, labeled_filter, pose_filter = _parse_filters(request)
    filtered = _filter_images(images, stage_filter, labeled_filter, pose_filter)

    # Consistency check: refresh the index summary cache against actual files.
    build_summary(tax_id, images)

    groups = _build_unified_groups(tax_id, images, filtered)

    context = {
        "tax_id": tax_id,
        "groups": groups,
        "total": len(filtered),
        "total_all": len(images),
        "has_images": bool(images),
        "wing_stats_available": get_wing_stats_path(tax_id).is_file(),
        "side_wing_stats_available": get_side_wing_stats_path(tax_id).is_file(),
        "filter_desc": _filter_desc(stage_filter, labeled_filter, pose_filter),
        "is_filtered": bool(stage_filter or labeled_filter or pose_filter),
        "filter_qs": request.GET.urlencode(),
    }
    return render(request, "moths/tax_poses.html", context)


@require_POST
def rebuild_poses(request, tax_id):
    """Recompute and cache the pose data for a tax_id, then return to the view.

    Preserves the active filters (carried in the query string) on redirect.
    """
    images = group_by_tax_id().get(tax_id)
    if not images:
        raise Http404(f"No images found for tax_id {tax_id!r}")

    build_pose_data(tax_id, images)

    url = reverse("moths:pose_view", args=[tax_id])
    query = request.GET.urlencode()
    if query:
        url = f"{url}?{query}"
    return redirect(url)


def _ordered_nav(request, image):
    """Return (position, total, prev_filename, next_filename) for an image.

    Both single-image views (edit and normalized) navigate in the exact order
    the poses page lays images out: by unified group, then starred-first, then
    score descending. Honors the active stage/labeled filters, falling back to
    the unfiltered order if the current image is filtered out. Sharing this one
    helper keeps edit/normalized/poses navigation consistent.
    """
    images = group_by_tax_id().get(image.tax_id, [])
    stage_filter, labeled_filter, pose_filter = _parse_filters(request)
    filtered = _filter_images(images, stage_filter, labeled_filter, pose_filter)

    def ordered(image_list):
        groups = _build_unified_groups(image.tax_id, images, image_list)
        # A flagged image can appear in several flag subsections; keep the first
        # occurrence so navigation visits each image once.
        seen = set()
        out = []
        for group in groups:
            for row in group["rows"]:
                fn = row["image"].filename
                if fn not in seen:
                    seen.add(fn)
                    out.append(fn)
        return out

    filenames = ordered(filtered)
    if image.filename not in filenames:
        filenames = ordered(images)

    try:
        index = filenames.index(image.filename)
    except ValueError:
        index = None

    prev_filename = filenames[index - 1] if index not in (None, 0) else None
    next_filename = (
        filenames[index + 1]
        if index is not None and index < len(filenames) - 1
        else None
    )
    position = index + 1 if index is not None else None
    return position, len(filenames), prev_filename, next_filename


def image_edit(request, filename):
    """Editable single-image view: annotations and stage (full controls)."""
    image = find_image(filename)
    if image is None:
        raise Http404("Image not found")

    annotations = load_annotations(filename)

    # Plain data for the client-side editor (normalized coordinates).
    objects_data = [
        {
            "class_id": annotation.class_id,
            "cx": annotation.cx,
            "cy": annotation.cy,
            "w": annotation.width,
            "h": annotation.height,
            "keypoints": [
                {"x": kp.x, "y": kp.y, "v": kp.visibility}
                for kp in annotation.keypoints
            ],
        }
        for annotation in annotations
    ]
    keypoint_slots = [KEYPOINT_LABELS.get(i + 1, str(i + 1)) for i in range(4)]

    # Read-only reference predictions (from MOTHS_PREDICTION_DIR), overlaid for
    # comparison. Never editable.
    prediction_data = [
        {
            "class_id": annotation.class_id,
            "cx": annotation.cx,
            "cy": annotation.cy,
            "w": annotation.width,
            "h": annotation.height,
            "keypoints": [
                {"x": kp.x, "y": kp.y, "v": kp.visibility}
                for kp in annotation.keypoints
            ],
        }
        for annotation in load_predictions(filename)
    ]

    position, total, prev_filename, next_filename = _ordered_nav(request, image)

    predicted_stage, predicted_flags = get_predicted_class_and_flags(filename)

    context = {
        "image": image,
        "has_label": get_label_path(filename).is_file(),
        "label_name": get_label_path(filename).name,
        "stages": STAGES,
        "current_stage": get_image_class(filename),
        "predicted_stage": predicted_stage,
        "predicted_pose_class": get_predicted_pose_class(filename),
        # Flag options with the stages each applies to, so the picker can hide
        # flags that don't apply to the image's current stage (see the table).
        "flag_specs": [
            {"name": flag.name, "stages": list(flag.stages) or list(STAGES)}
            for flag in FLAG_TABLE
        ],
        "current_flags": get_image_flags(filename),
        "predicted_flags": predicted_flags,
        "position": position,
        "total": total,
        "prev_filename": prev_filename,
        "next_filename": next_filename,
        "objects_data": objects_data,
        "prediction_data": prediction_data,
        "has_prediction": bool(prediction_data),
        "keypoint_slots": keypoint_slots,
        "norm_available": compute_normalization(filename) is not None,
        "starred": is_image_starred(filename),
        "filter_qs": request.GET.urlencode(),
    }
    return render(request, "moths/image_edit.html", context)


def image_normalized(request, filename):
    """Read-only pose-normalized view with a star/unstar control.

    Redirects to the edit view when the image has no normalized crop (missing
    prediction keypoints), so the back/tabs never dead-end.
    """
    image = find_image(filename)
    if image is None:
        raise Http404("Image not found")

    # Verify the keypoints against the live pose source and recompute this
    # image's normalized crop + scores if they changed. This is the only place
    # that recompute is triggered (the poses view reads the cache as-is).
    metrics = verify_pose_row(image.tax_id, filename)
    if metrics:
        sharpness = metrics.get("sharpness")
        sym_score, pixel_score, sharp_score = score_components(
            metrics.get("symmetry"), metrics.get("pixel_span"), sharpness
        )
        metrics = {
            **metrics,
            "sym_score": sym_score,
            "pixel_score": pixel_score,
            "sharp_score": sharp_score,
            "sharpness_m": None if sharpness is None else sharpness / 1_000_000,
        }

    normalization = compute_normalization(filename)
    if normalization is None:
        url = reverse("moths:image_edit", args=[filename])
        query = request.GET.urlencode()
        return redirect(f"{url}?{query}" if query else url)

    norm_keypoints = [
        {
            "x": kp["x"],
            "y": kp["y"],
            "v": kp["v"],
            "label": kp["label"],
            "color": VISIBILITY_COLORS.get(kp["v"], "#3b82f6"),
        }
        for kp in normalization["keypoints"]
    ]

    position, total, prev_filename, next_filename = _ordered_nav(request, image)

    observation = get_observation_info(filename) or {}
    license_code = observation.get("license_code")
    quality_grade = observation.get("quality_grade")

    # Reference-circle geometry (as % of the crop) matching the pose's scaling:
    # 90% for top-down/bottom-up, 80% for side view.
    circle_radius = normalization.get("circle_radius", 0.45)
    circle_pct = circle_radius * 2 * 100
    circle_offset_pct = (0.5 - circle_radius) * 100

    context = {
        "image": image,
        "position": position,
        "total": total,
        "prev_filename": prev_filename,
        "next_filename": next_filename,
        "norm_keypoints": norm_keypoints,
        "circle_pct": circle_pct,
        "circle_offset_pct": circle_offset_pct,
        "starred": is_image_starred(filename),
        "details": get_image_details(filename),
        "details_range": [1, 2, 3, 4, 5],
        "metrics": metrics,
        "license_code": license_code,
        "license_is_cc": bool(license_code) and license_code.lower().startswith("cc"),
        "quality_grade": quality_grade,
        "is_research": quality_grade == "research",
        "filter_qs": request.GET.urlencode(),
    }
    return render(request, "moths/image_normalized.html", context)


@require_POST
def set_selection_stage(request, tax_id):
    """Classify an explicit set of a tax_id's images as ``stage``.

    Body is JSON ``{"stage": "Adult", "filenames": [...]}``. Every listed image
    that belongs to this tax_id is (re)labelled with the stage, regardless of any
    current class. Backs the poses view's selection mode. Returns how many were
    updated.
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "invalid JSON"}, status=400)

    stage = (payload.get("stage") or "").strip()
    if stage not in STAGES:
        return JsonResponse({"error": "invalid stage"}, status=400)

    filenames = payload.get("filenames") or []
    if not isinstance(filenames, list):
        return JsonResponse({"error": "filenames must be a list"}, status=400)

    count = 0
    for filename in filenames:
        if not isinstance(filename, str):
            continue
        if tax_id_for_file(filename) != tax_id or find_image(filename) is None:
            continue
        set_image_class(filename, stage)
        count += 1
    if count:
        update_summary(tax_id)
    return JsonResponse({"ok": True, "count": count, "stage": stage})


@require_POST
def delete_selection(request, tax_id):
    """Physically delete an explicit set of a tax_id's images (skips starred).

    Body is JSON ``{"filenames": [...]}``. Each listed image that belongs to
    this tax_id and is not starred is deleted along with its label, class,
    prediction and cache files; the observation and pose-data caches are pruned
    and the summary rebuilt (see :func:`moths.utils.delete_images`). Starred
    images are skipped. Returns how many were deleted and how many starred
    images were skipped.
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "invalid JSON"}, status=400)

    filenames = payload.get("filenames") or []
    if not isinstance(filenames, list):
        return JsonResponse({"error": "filenames must be a list"}, status=400)

    result = delete_images(
        tax_id, [name for name in filenames if isinstance(name, str)]
    )
    return JsonResponse({"ok": True, **result})


@require_POST
def save_label(request, filename):
    """Persist edited YOLO-pose annotations to the image's label file.

    Body is JSON: ``{"objects": [{class_id, cx, cy, w, h, keypoints:[{x,y,v}]}]}``.
    An empty list removes the label file.
    """
    if find_image(filename) is None:
        raise Http404("Image not found")
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "invalid JSON"}, status=400)

    objects = payload.get("objects", [])
    try:
        save_annotations(filename, objects)
    except (KeyError, TypeError, ValueError) as exc:
        return JsonResponse({"error": f"invalid data: {exc}"}, status=400)
    tax_id = tax_id_for_file(filename)
    update_summary(tax_id)
    # Keypoints just changed: flag the cached pose row for rebuild (deferred;
    # the poses view shows "to be rebuild" until the next rebuild).
    mark_pose_row_stale(tax_id, filename)
    return JsonResponse({"ok": True, "count": len(objects)})


def serve_image(request, filename):
    """Serve a single image file from the (external) image directory.

    Images live in a per-``tax_id`` subfolder derived from the filename. Guards
    against path traversal by resolving the requested path and ensuring it stays
    inside the configured image directory.
    """
    image_dir = get_image_dir().resolve()
    file_path = get_image_path(filename).resolve()
    if image_dir not in file_path.parents:
        raise Http404("Invalid image path")
    if not file_path.is_file():
        raise Http404("Image not found")
    return FileResponse(open(file_path, "rb"))


@require_POST
def set_stage(request, filename):
    """Store or clear the stage classification for an image.

    Writes ``<name>.class`` containing the stage name; an empty/missing stage
    removes the file. Returns the resulting stage as JSON.
    """
    if find_image(filename) is None:
        raise Http404("Image not found")

    stage = (request.POST.get("stage") or "").strip()
    if not stage:
        clear_image_class(filename)
        update_summary(tax_id_for_file(filename))
        return JsonResponse({"stage": None})
    if stage not in STAGES:
        return JsonResponse({"error": "invalid stage"}, status=400)
    set_image_class(filename, stage)
    update_summary(tax_id_for_file(filename))
    return JsonResponse({"stage": stage})


@require_POST
def set_flags(request, filename):
    """Store the optional flags (Pinned/Macro/Damaged) for an image.

    Accepts a JSON body ``{"flags": [...]}``; unknown names are ignored. Flags
    live in the ``.class`` file alongside the stage and are mirrored into the
    cached pose row. Returns the resulting flag list.
    """
    if find_image(filename) is None:
        raise Http404("Image not found")
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "invalid JSON"}, status=400)

    requested = payload.get("flags", [])
    if not isinstance(requested, list):
        return JsonResponse({"error": "flags must be a list"}, status=400)
    flags = set_image_flags(filename, requested)
    set_pose_row_flags(tax_id_for_file(filename), filename, flags)
    return JsonResponse({"flags": flags})


@require_POST
def set_details(request, filename):
    """Store (1-5) or clear the throwaway hand ``details`` rating for an image.

    POST ``rating`` = ``1``..``5`` sets it; an empty/other value clears it. The
    rating is a ground-truth annotation for tuning the sharpness score: it lives
    only in the ``.class`` file and is deliberately kept out of the pose-data and
    summary caches, so there is nothing to re-summarise here. Returns the stored
    value as JSON (``null`` when cleared).
    """
    if find_image(filename) is None:
        raise Http404("Image not found")

    raw = (request.POST.get("rating") or "").strip()
    try:
        rating = int(raw)
    except ValueError:
        rating = None
    return JsonResponse({"details": set_image_details(filename, rating)})


@require_POST
def set_star(request, filename):
    """Star or unstar an observation (image). Returns the new state as JSON.

    A truthy ``starred`` POST value stars it; anything else unstars it.
    """
    if find_image(filename) is None:
        raise Http404("Image not found")

    starred = (request.POST.get("starred") or "").strip() in ("1", "true", "on", "yes")
    set_image_starred(filename, starred)
    # Starring can change the representative thumbnail (starred images win).
    refresh_tax_thumbnail(tax_id_for_file(filename))
    return JsonResponse({"starred": starred})


def serve_thumbnail(request, filename):
    """Serve a cached thumbnail, generating it on first request.

    Falls back to the full image if a thumbnail can't be produced.
    """
    thumb = get_or_create_thumbnail(filename)
    if thumb is None or not thumb.is_file():
        return serve_image(request, filename)
    return FileResponse(open(thumb, "rb"))


def serve_norm_thumbnail(request, filename):
    """Serve the pose-normalized thumbnail, generating it on first request.

    Falls back to the regular thumbnail if the normalized crop can't be built
    (e.g. missing prediction keypoints).
    """
    result = get_or_create_normalized(filename)
    if result is None or not result[1].is_file():
        return serve_thumbnail(request, filename)
    return FileResponse(open(result[1], "rb"))


def serve_norm_image(request, filename):
    """Serve the full-resolution pose-normalized crop, generating if needed."""
    result = get_or_create_normalized(filename)
    if result is None or not result[0].is_file():
        return serve_image(request, filename)
    return FileResponse(open(result[0], "rb"))


def serve_wing_stats(request, tax_id):
    """Serve the pre-generated top-down wing-position scatter PNG for a tax_id."""
    path = get_wing_stats_path(tax_id)
    if not path.is_file():
        raise Http404("No wing-stats image for this tax_id.")
    return FileResponse(open(path, "rb"))


def serve_side_wing_stats(request, tax_id):
    """Serve the pre-generated side-view wing-position scatter PNG for a tax_id."""
    path = get_side_wing_stats_path(tax_id)
    if not path.is_file():
        raise Http404("No side wing-stats image for this tax_id.")
    return FileResponse(open(path, "rb"))
