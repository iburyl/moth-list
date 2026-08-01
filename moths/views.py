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
    KEYPOINT_LABELS,
    POSE_BOTTOM_UP,
    FLAGS,
    POSE_NONE,
    POSE_SIDE,
    POSE_TOP_DOWN,
    POSE_UNCLEAR,
    SOURCE_STAGES,
    STAGES,
    build_pose_data,
    build_summary,
    compute_normalization,
    clear_image_class,
    find_image,
    get_class_and_flags,
    get_image_class,
    get_image_flags,
    get_image_dir,
    get_image_path,
    get_label_path,
    get_name_info,
    get_observation_info,
    get_or_create_normalized,
    get_or_create_thumbnail,
    get_tax_thumbnail,
    group_by_tax_id,
    is_image_starred,
    list_tax_ids,
    load_annotations,
    load_pose_data,
    load_pose_data_raw,
    load_predictions,
    load_starred,
    load_summary,
    mark_pose_row_stale,
    pose_data_version_ok,
    refresh_tax_thumbnail,
    save_annotations,
    scan_tax_images,
    score_components,
    set_image_class,
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


def _unified_group_defs():
    """Ordered ``(group_id, label, is_unknown_base)`` specs for all subsections.

    Flag subsections are woven into the requested order: for adults, ``pinned``
    follows top-down and ``macro``/``damaged`` follow bottom-up; other stages get
    their flag subsections right after the base group. ``is_unknown_base`` marks
    the single unflagged "Unknown stage" group (the only one with bulk buttons).
    """
    defs = [
        (GROUP_ADULT_TOP_DOWN, "Adult: top-down view", False),
        (_flag_gid("Adult", "Pinned"), "Adult: pinned", False),
        (GROUP_ADULT_SIDE, "Adult: side view", False),
        (GROUP_ADULT_BOTTOM_UP, "Adult: bottom-up view", False),
        (_flag_gid("Adult", "Macro"), "Adult: macro", False),
        (_flag_gid("Adult", "Damaged"), "Adult: damaged", False),
        (GROUP_ADULT_UNCLEAR, "Adult: unclear pose", False),
        (GROUP_ADULT_NONE, "Adult: no keypoints", False),
    ]
    for stage_key, plural in (("Pupa", "Pupa"), ("Larva", "Larvae"), ("Egg", "Egg")):
        defs.append((stage_key, plural, False))
        for flag in FLAGS:
            defs.append((_flag_gid(stage_key, flag), f"{plural}: {flag.lower()}", False))
    defs.append((GROUP_UNKNOWN, "Unknown stage", True))
    for flag in FLAGS:
        defs.append((_flag_gid("unknown", flag), f"Unknown stage: {flag.lower()}", False))
    return defs


def _target_group_ids(stage, flags, pose):
    """Group id(s) an image belongs to: one per flag, else its stage/pose group."""
    stage_key = stage if stage in STAGE_KEYS else "unknown"
    if flags:
        return [_flag_gid(stage_key, flag) for flag in flags]
    if stage_key == "Adult":
        return [ADULT_POSE_GROUP.get(pose, GROUP_ADULT_NONE)]
    return [stage_key]

# Colors per keypoint visibility flag (0 unlabeled, 1 occluded, 2 visible).
VISIBILITY_COLORS = {0: "#9ca3af", 1: "#f59e0b", 2: "#22c55e"}

# Header labels for the "Source data stage" columns, paired with the stage keys
# stored in the summary (SOURCE_STAGES). "unknown" already folds in every other
# documented/blank stage.
SOURCE_STAGE_LABELS = {
    "Egg": "egg",
    "Larva": "larvae",
    "Pupa": "pupa",
    "Adult": "adult",
    "Unknown": "unknown",
}

# "Source data check" columns: (header label, audit status key), in order.
SOURCE_CHECK_COLUMNS = [
    ("ok", "taken"),
    ("rej-license", "rejected_non_cc"),
    ("rej-full", "rejected_stage_quota_full"),
    ("rej-no pose", "rejected_no_pose"),
    ("rej-wrong pose", "rejected_not_top_down"),
    ("rej-score", "rejected_low_score"),
]

# Terminal marker -> single-letter "finish" state (priority order).
SOURCE_FINISH_LETTERS = [
    ("done", "D"),
    ("no_more_observations", "N"),
    ("reached_scan_limit", "Q"),
    ("corrupted", "C"),
]


def _source_display(summary):
    """Build the index "Source data" cells from a summary's ``source_data``.

    Returns a dict with ``stages`` (list of per-stage counts) and ``checks``
    (list of per-status counts), each aligned to a fixed column set, plus a
    ``finish`` letter. Counts are ``None`` (rendered as a dash) when the tax has
    no harvest audit CSV.
    """
    source = summary.get("source_data") if summary else None
    has = source is not None
    stages = (source or {}).get("stages") or {}
    statuses = (source or {}).get("statuses") or {}

    stage_cells = [
        stages.get(key, 0) if has else None for key in SOURCE_STAGES
    ]
    check_cells = [
        statuses.get(key, 0) if has else None for _, key in SOURCE_CHECK_COLUMNS
    ]
    finish = ""
    if has:
        for key, letter in SOURCE_FINISH_LETTERS:
            if statuses.get(key):
                finish = letter
                break
    return {"has": has, "stages": stage_cells, "checks": check_cells, "finish": finish}


def index(request):
    """Table of tax_ids with per-stage, unclassified and labeled image counts.

    Rows come from the image directory's per-``tax_id`` subfolders. Counts and
    names are read from a per-tax summary cache (``<tax_id>_summary.json``),
    which is built on first sight and refreshed by edit actions and the
    poses-view consistency check — so this view avoids scanning image files.
    """
    rows = []
    for tax_id in list_tax_ids():
        summary = load_summary(tax_id)
        if summary is None:
            summary = build_summary(tax_id, scan_tax_images(tax_id))

        counts = summary.get("counts", {})
        names = summary.get("names", {})
        stages_map = counts.get("stages", {})
        thumbnail = summary.get("thumbnail") or {}
        source = _source_display(summary)
        rows.append(
            {
                "tax_id": tax_id,
                "family": names.get("family", ""),
                "species": names.get("species", ""),
                "name": names.get("name", ""),
                "thumbnail": thumbnail.get("filename") or None,
                "total": counts.get("total", 0),
                "stage_cells": [
                    {"stage": stage, "count": stages_map.get(stage, 0)}
                    for stage in STAGES
                ],
                "unclassified": counts.get("unclassified", 0),
                "labeled": counts.get("labeled", 0),
                "not_labeled": counts.get("not_labeled", 0),
                "source": source,
            }
        )

    # Sort by family, then species; unknown (blank) names fall to the end.
    rows.sort(
        key=lambda r: (
            r["family"] == "",
            r["family"].lower(),
            r["species"] == "",
            r["species"].lower(),
        )
    )
    context = {
        "rows": rows,
        "stages": STAGES,
        "total": len(rows),
        "image_dir": str(get_image_dir()),
        "source_stage_labels": [SOURCE_STAGE_LABELS[s] for s in SOURCE_STAGES],
        "source_check_labels": [label for label, _ in SOURCE_CHECK_COLUMNS],
    }
    return render(request, "moths/index.html", context)


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
    """Extract (stage, labeled) filters from the query string."""
    return (
        request.GET.get("stage") or None,
        request.GET.get("labeled") or None,
    )


def _filter_desc(stage_filter, labeled_filter):
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
    return ", ".join(parts) if parts else None


def _filter_images(images, stage_filter, labeled_filter):
    """Apply stage/label filters to a list of images."""
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


def _make_row(image, data, version_ok, starred, is_adult):
    """Build a template row for one image from its cached pose ``data``.

    Keypoint treatment (metrics/scores, normalized thumbnail + normalized-view
    link, and score-sorting of the group) applies only to **Adult** images with
    a keypoint pose (top-down/side/bottom-up/unclear). Non-adult stages get a
    plain thumbnail and the edit link even when their only pose source is a
    leftover adult prediction — otherwise a larva/pupa would be score-sorted and
    reshuffle as its cached row is flagged for rebuild. Pose info travels on the
    row so a single flag subsection can mix images of different poses.
    """
    pose = data.get("pose", POSE_NONE)
    has_keypoints = is_adult and pose in KEYPOINT_POSES
    row = {
        "image": image,
        "starred": image.filename in starred,
        "has_keypoints": has_keypoints,
        "is_norm": is_adult and pose in NORM_POSES,
        "is_top_down": is_adult and pose == POSE_TOP_DOWN,
        "from_prediction": False,
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
                "from_prediction": (
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
        stage, flags = get_class_and_flags(image.filename)
        data = per_image.get(image.filename) or {}
        pose = data.get("pose", POSE_NONE)
        row = _make_row(image, data, version_ok, starred, stage == "Adult")
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

    stage_filter, labeled_filter = _parse_filters(request)
    filtered = _filter_images(images, stage_filter, labeled_filter)

    # Consistency check: refresh the index summary cache against actual files.
    build_summary(tax_id, images)

    groups = _build_unified_groups(tax_id, images, filtered)

    context = {
        "tax_id": tax_id,
        "groups": groups,
        "total": len(filtered),
        "total_all": len(images),
        "has_images": bool(images),
        "filter_desc": _filter_desc(stage_filter, labeled_filter),
        "is_filtered": bool(stage_filter or labeled_filter),
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
    stage_filter, labeled_filter = _parse_filters(request)
    filtered = _filter_images(images, stage_filter, labeled_filter)

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

    context = {
        "image": image,
        "has_label": get_label_path(filename).is_file(),
        "label_name": get_label_path(filename).name,
        "stages": STAGES,
        "current_stage": get_image_class(filename),
        "flags": FLAGS,
        "current_flags": get_image_flags(filename),
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

    context = {
        "image": image,
        "position": position,
        "total": total,
        "prev_filename": prev_filename,
        "next_filename": next_filename,
        "norm_keypoints": norm_keypoints,
        "starred": is_image_starred(filename),
        "metrics": metrics,
        "license_code": license_code,
        "license_is_cc": bool(license_code) and license_code.lower().startswith("cc"),
        "quality_grade": quality_grade,
        "is_research": quality_grade == "research",
        "filter_qs": request.GET.urlencode(),
    }
    return render(request, "moths/image_normalized.html", context)


@require_POST
def add_to_stage(request, tax_id, stage):
    """Classify a tax_id's currently-unclassified, filtered images as ``stage``.

    Scoped to images that have no stage class yet (the "Unknown stage" group),
    so it never re-labels images already assigned to a stage. Filters are read
    from the query string.
    """
    if stage not in STAGES:
        return JsonResponse({"error": "invalid stage"}, status=400)

    images = group_by_tax_id().get(tax_id)
    if not images:
        raise Http404(f"No images found for tax_id {tax_id!r}")

    stage_filter, labeled_filter = _parse_filters(request)
    filtered = _filter_images(images, stage_filter, labeled_filter)
    targets = [img for img in filtered if get_image_class(img.filename) not in STAGES]
    for image in targets:
        set_image_class(image.filename, stage)
    update_summary(tax_id)
    return JsonResponse({"ok": True, "count": len(targets), "stage": stage})


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
