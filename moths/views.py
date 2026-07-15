import json

from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .utils import (
    KEYPOINT_LABELS,
    SPLIT_FILES,
    STAGES,
    clear_image_class,
    find_image,
    get_image_class,
    get_image_dir,
    get_image_subset,
    get_label_path,
    get_or_create_thumbnail,
    group_by_tax_id,
    load_annotations,
    load_subset_map,
    save_annotations,
    set_image_class,
    set_image_subset,
    set_images_subset,
)

# Colors per keypoint visibility flag (0 unlabeled, 1 occluded, 2 visible).
VISIBILITY_COLORS = {0: "#9ca3af", 1: "#f59e0b", 2: "#22c55e"}


# Subset -> index within a [train, val, unlisted] count triple.
SUBSET_INDEX = {"train": 0, "val": 1}


def _cell(triple):
    """Turn a [train, val, unlisted] list into a template-friendly dict."""
    return {
        "train": triple[0],
        "val": triple[1],
        "none": triple[2],
        "total": sum(triple),
    }


def index(request):
    """Table of tax_ids with per-stage, unclassified and labeled image counts.

    Every count is broken down by subset as train + val + unlisted.
    """
    groups = group_by_tax_id()
    subset_map = load_subset_map()
    rows = []
    for tax_id, images in groups.items():
        stage_triples = {stage: [0, 0, 0] for stage in STAGES}
        unclassified = [0, 0, 0]
        labeled = [0, 0, 0]
        not_labeled = [0, 0, 0]
        for image in images:
            si = SUBSET_INDEX.get(subset_map.get(image.filename), 2)
            stage = get_image_class(image.filename)
            if stage in stage_triples:
                stage_triples[stage][si] += 1
            else:
                unclassified[si] += 1
            if get_label_path(image.filename).is_file():
                labeled[si] += 1
            else:
                not_labeled[si] += 1
        rows.append(
            {
                "tax_id": tax_id,
                "total": len(images),
                "stage_cells": [
                    dict(_cell(stage_triples[stage]), stage=stage) for stage in STAGES
                ],
                "unclassified": _cell(unclassified),
                "labeled": _cell(labeled),
                "not_labeled": _cell(not_labeled),
            }
        )
    context = {
        "rows": rows,
        "stages": STAGES,
        "total": len(rows),
        "image_dir": str(get_image_dir()),
    }
    return render(request, "moths/index.html", context)


def _parse_filters(request):
    """Extract (stage, labeled, subset) filters from the query string."""
    return (
        request.GET.get("stage") or None,
        request.GET.get("labeled") or None,
        request.GET.get("subset") or None,
    )


def _filter_images(images, stage_filter, labeled_filter, subset_filter, subset_map):
    """Apply stage/label/subset filters to a list of images."""
    result = []
    for image in images:
        if subset_filter:
            actual = subset_map.get(image.filename) or "none"
            if actual != subset_filter:
                continue
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
    """Show images for a tax_id, optionally filtered by stage/label/subset.

    Query params:
        stage=<Stage>|none   filter by classification ("none" = unclassified)
        labeled=yes|no        filter by presence of a label file
        subset=train|val|none filter by split membership ("none" = unlisted)
    """
    images = group_by_tax_id().get(tax_id)
    if not images:
        raise Http404(f"No images found for tax_id {tax_id!r}")

    stage_filter, labeled_filter, subset_filter = _parse_filters(request)
    subset_map = load_subset_map()
    filtered = _filter_images(
        images, stage_filter, labeled_filter, subset_filter, subset_map
    )

    # Build a human-readable description of the active filters.
    parts = []
    if stage_filter == "none":
        parts.append("unclassified")
    elif stage_filter:
        parts.append(f"stage {stage_filter}")
    if labeled_filter == "yes":
        parts.append("with labels")
    elif labeled_filter == "no":
        parts.append("without labels")
    if subset_filter == "none":
        parts.append("not in train/val")
    elif subset_filter:
        parts.append(f"{subset_filter} subset")

    context = {
        "tax_id": tax_id,
        "images": filtered,
        "total_all": len(images),
        "filter_desc": ", ".join(parts) if parts else None,
        "is_filtered": bool(stage_filter or labeled_filter or subset_filter),
        "filter_qs": request.GET.urlencode(),
    }
    return render(request, "moths/tax_detail.html", context)


def image_detail(request, filename):
    """Show a single image with its YOLO-pose labels overlaid."""
    image = find_image(filename)
    if image is None:
        raise Http404("Image not found")

    annotations = load_annotations(filename)
    # Enrich keypoints with a display index and color for the template.
    overlays = []
    for annotation in annotations:
        keypoints = [
            {
                "index": i + 1,
                "label": KEYPOINT_LABELS.get(i + 1, str(i + 1)),
                "x_pct": kp.x * 100,
                "y_pct": kp.y * 100,
                "visibility": kp.visibility,
                "color": VISIBILITY_COLORS.get(kp.visibility, "#3b82f6"),
            }
            for i, kp in enumerate(annotation.keypoints)
        ]
        overlays.append({"annotation": annotation, "keypoints": keypoints})

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

    # Navigation within the current tax_id's image list (n of m), honoring the
    # same filters that were active on the tax_detail page.
    siblings = group_by_tax_id().get(image.tax_id, [])
    stage_filter, labeled_filter, subset_filter = _parse_filters(request)
    filtered_siblings = _filter_images(
        siblings, stage_filter, labeled_filter, subset_filter, load_subset_map()
    )
    filenames = [img.filename for img in filtered_siblings]
    if image.filename not in filenames:
        # Current image is outside the filter (e.g. direct link) — fall back.
        filenames = [img.filename for img in siblings]
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

    context = {
        "image": image,
        "overlays": overlays,
        "has_label": get_label_path(filename).is_file(),
        "label_name": get_label_path(filename).name,
        "subset": get_image_subset(filename),
        "stages": STAGES,
        "current_stage": get_image_class(filename),
        "position": index + 1 if index is not None else None,
        "total": len(filenames),
        "prev_filename": prev_filename,
        "next_filename": next_filename,
        "objects_data": objects_data,
        "keypoint_slots": keypoint_slots,
        "filter_qs": request.GET.urlencode(),
    }
    return render(request, "moths/image_detail.html", context)


@require_POST
def add_to_train(request, tax_id):
    """Add all currently-filtered images of a tax_id to the train subset.

    Filters are read from the query string, matching the tax_detail view.
    """
    images = group_by_tax_id().get(tax_id)
    if not images:
        raise Http404(f"No images found for tax_id {tax_id!r}")

    stage_filter, labeled_filter, subset_filter = _parse_filters(request)
    filtered = _filter_images(
        images, stage_filter, labeled_filter, subset_filter, load_subset_map()
    )
    filenames = [image.filename for image in filtered]
    set_images_subset(filenames, "train")
    return JsonResponse({"ok": True, "count": len(filenames)})


@require_POST
def set_subset(request, filename):
    """Assign the image to a subset (train/val) or remove it (empty/none).

    Updates train.txt / val.txt, keeping them sorted. Returns the new subset.
    """
    if find_image(filename) is None:
        raise Http404("Image not found")

    subset = (request.POST.get("subset") or "").strip()
    if not subset or subset == "none":
        set_image_subset(filename, None)
        return JsonResponse({"subset": None})
    if subset not in SPLIT_FILES:
        return JsonResponse({"error": "invalid subset"}, status=400)
    set_image_subset(filename, subset)
    return JsonResponse({"subset": subset})


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
    return JsonResponse({"ok": True, "count": len(objects)})


def serve_image(request, filename):
    """Serve a single image file from the (external) image directory.

    Guards against path traversal by resolving the requested path and
    ensuring it stays inside the configured image directory.
    """
    image_dir = get_image_dir().resolve()
    file_path = (image_dir / filename).resolve()
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
        return JsonResponse({"stage": None})
    if stage not in STAGES:
        return JsonResponse({"error": "invalid stage"}, status=400)
    set_image_class(filename, stage)
    return JsonResponse({"stage": stage})


def serve_thumbnail(request, filename):
    """Serve a cached thumbnail, generating it on first request.

    Falls back to the full image if a thumbnail can't be produced.
    """
    thumb = get_or_create_thumbnail(filename)
    if thumb is None or not thumb.is_file():
        return serve_image(request, filename)
    return FileResponse(open(thumb, "rb"))
