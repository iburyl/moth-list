import json

from django.http import (
    FileResponse,
    Http404,
    JsonResponse,
)
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from .permissions import editor_required
from .utils import (
    _NO_SUBFAMILY,
    _UNKNOWN_TAXON,
    FLAG_TABLE,
    KEYPOINT_LABELS,
    POSE_BOTTOM_UP,
    flags_suppress_normalization,
    POSE_NONE,
    POSE_SIDE,
    POSE_TOP_DOWN,
    POSE_UNCLEAR,
    STAGES,
    VIEW_POSES,
    build_pose_data,
    build_summary,
    build_integration_groups,
    classify_pose,
    compute_normalization,
    clear_image_class,
    clear_normalized,
    delete_images,
    find_image,
    get_class_and_flags,
    get_class_and_flags_with_source,
    get_image_class,
    get_image_flags,
    get_image_dir,
    get_image_path,
    get_label_path,
    get_prediction_path,
    get_name_info,
    get_image_details,
    get_observation_info,
    get_or_create_normalized,
    get_or_create_thumbnail,
    get_predicted_class_and_flags,
    get_predicted_pose_class,
    get_side_wing_stats_path,
    get_tax_summary_path,
    get_tax_thumbnail,
    get_wing_stats_path,
    integration_layout,
    scan_tax_images,
    is_image_starred,
    list_tax_ids,
    load_annotations,
    load_names,
    load_pose_data,
    load_pose_data_raw,
    load_predictions,
    load_starred,
    load_summary,
    load_tax_summary,
    load_data_tax_summary,
    mark_pose_row_stale,
    pose_data_version_ok,
    pose_row_needs_rebuild,
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
    flat_taxonomy_rows,
    load_data_summary,
    load_species_data,
)

# Unified stage/pose/flag group definitions now live in moths.utils.groups (a
# single source of truth shared with the thumbnail chooser). Imported with the
# historical private names so the rest of this module reads unchanged.
from .utils.groups import KEYPOINT_POSES  # noqa: E402
from .utils.groups import target_group_ids as _target_group_ids  # noqa: E402
from .utils.groups import unified_group_defs as _unified_group_defs  # noqa: E402

# Colors per keypoint visibility flag (0 unlabeled, 1 occluded, 2 visible).
VISIBILITY_COLORS = {0: "#9ca3af", 1: "#f59e0b", 2: "#22c55e"}


# Display labels for the index "view" (pose) column group.
VIEW_POSE_LABELS = {
    POSE_TOP_DOWN: "top-down",
    POSE_SIDE: "side",
    POSE_BOTTOM_UP: "bottom-up",
    POSE_UNCLEAR: "unclear",
}


def index(request):
    """Landing page: the taxonomy browser at its root (the superfamily list).

    The old flat per-tax_id table got too slow at ~900 species; the index is now
    the top level of the hierarchical browser backed by ``tax_summary.json`` (see
    :func:`browse`), which renders from the pre-aggregated file without scanning.
    """
    return browse(request, lineage="")


def flat_taxonomy(request):
    """All species in taxonomic order with cross-source scientific-name cells."""
    data = load_data_summary()
    rows = flat_taxonomy_rows(data)
    species_count = sum(1 for row in rows if row.get("kind") == "species")
    return render(
        request,
        "moths/flat_taxonomy.html",
        {
            "rows": rows,
            "integration_flat": integration_layout("flat"),
            "species_count": species_count,
            "missing": data is None,
        },
    )


def _index_row(tax_id: str) -> dict:
    """Build one flat index-table row for a tax_id from its cached summary.

    Read-only: never builds here (that is the heavy path). A tax with no cached
    summary yet renders with "—" placeholders; its summary is built when the
    species view is entered. Names/obs come from ``data_summary.json`` via
    :func:`get_name_info`. Reused by the legacy flat index and by the
    genus-level browse page (its species listing).
    """
    summary = load_summary(tax_id)
    name_info = get_name_info(tax_id)

    counts = (summary or {}).get("counts", {})
    stages_map = counts.get("stages", {})
    views_map = counts.get("views", {})
    thumbnail = (summary or {}).get("thumbnail") or {}
    has_summary = summary is not None

    def _stage_cell(stage):
        return {
            "stage": stage,
            "count": stages_map.get(stage, 0) if has_summary else None,
        }

    def _view_cell(pose):
        return {
            "pose": pose,
            "label": VIEW_POSE_LABELS[pose],
            "count": views_map.get(pose, 0) if has_summary else None,
        }

    species_data = load_species_data(tax_id)
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
        "integrations": build_integration_groups(
            "species-list", species_data, context={"tax_id": tax_id}
        ),
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
        "integration_species_list": integration_layout("species-list"),
        "total": len(rows),
        "image_dir": str(get_image_dir()),
    }
    return render(request, "moths/index.html", context)


# Taxonomy levels by descent depth: depth 0 lists superfamilies, ... depth 4
# lists the species leaves under a genus. Also the label of the children shown.
BROWSE_LEVELS = ["superfamily", "family", "subfamily", "genus", "species"]
# Plurals for column headers / count phrasing (English plurals here are
# irregular, so spell them out rather than naively appending "s").
BROWSE_LEVELS_PLURAL = {
    "superfamily": "superfamilies",
    "family": "families",
    "subfamily": "subfamilies",
    "genus": "genera",
    "species": "species",
}


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
# count under the node, and the hand-label gaps. External source columns come
# from ``data/tax_summary.json`` through the integration schema, not labels.
_BROWSE_ROW_KEYS = (
    "want",
    "species_want",
    "species_have_folder",
    "images",
    "no_stage",
    "no_box",
)


def browse(request, lineage=""):
    """Hierarchical taxonomy browser backed by ``labels/tax_summary.json``.

    The URL path mirrors the lineage, one taxon per segment
    (``browse/Noctuoidea/Erebidae/Arctiinae/Anathix``), with ``_`` standing in
    for the "(unknown)" bucket. Walking is a plain descent through the aggregate
    file's nested ``children`` maps — O(depth) lookups, no file scan. Each page
    lists the current node's direct children with their coverage counts; the
    species leaves under a genus link straight to the species view (there is no
    browse page for a single species).
    """
    data = load_tax_summary()
    integration_tree = load_data_tax_summary()
    segs = [s for s in lineage.split("/") if s]
    keys = [_browse_seg_to_key(s, i) for i, s in enumerate(segs)]

    if data is None:
        return render(request, "moths/browse.html", {"missing": True})

    # An over-long path (a species tax_id appended to a genus) or a segment that
    # isn't in the tree is an unknown taxon path -> stub, not a 404.
    if len(keys) > 4:
        return render(request, "moths/browse.html", {"unknown": True, "lineage": lineage})

    # Descend the tree by key: the root's children live under "superfamilies",
    # every deeper node's under "children".
    node = data
    integration_node = integration_tree
    for key in keys:
        container = node["superfamilies"] if node is data else node["children"]
        child = container.get(key)
        if child is None:
            return render(request, "moths/browse.html", {"unknown": True, "lineage": lineage})
        node = child
        # External coverage is optional; keep descending while its tree exists.
        if integration_node is not None:
            integration_container = (
                integration_node.get("superfamilies")
                if integration_node is integration_tree
                else integration_node.get("children")
            )
            integration_node = (
                integration_container.get(key)
                if isinstance(integration_container, dict)
                else None
            )

    depth = len(keys)
    container = node["superfamilies"] if node is data else node["children"]
    integration_container = None
    if integration_node is not None:
        integration_container = (
            integration_node.get("superfamilies")
            if integration_node is integration_tree
            else integration_node.get("children")
        )
        if not isinstance(integration_container, dict):
            integration_container = None

    # Collapse the subfamily level when a family isn't subdivided: if the only
    # subfamily is the "not subdivided" bucket ("-"), list its genera directly
    # (a lone "-" subfamily page would just be a pointless extra click). The
    # child URLs then carry the "_" subfamily placeholder so genus links resolve.
    url_prefix = list(segs)
    if depth == 2 and set(container) == {_NO_SUBFAMILY}:
        container = container[_NO_SUBFAMILY]["children"]
        if integration_container and _NO_SUBFAMILY in integration_container:
            integration_container = integration_container[_NO_SUBFAMILY].get("children")
            if not isinstance(integration_container, dict):
                integration_container = None
        else:
            integration_container = None
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
            integration_child = (
                integration_container.get(key) if integration_container else None
            )
            source_rollups = (
                integration_child.get("sources")
                if isinstance(integration_child, dict)
                else None
            )
            row["integrations"] = build_integration_groups(
                "browse", source_rollups
            )
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
        "unknown": False,
        "crumbs": crumbs,
        "children": children,
        "index_rows": index_rows,
        "integration_browse": integration_layout("browse"),
        "integration_species_list": integration_layout("species-list"),
        "stages": STAGES,
        "child_level": child_level,
        "child_level_plural": BROWSE_LEVELS_PLURAL[child_level],
        "child_is_species": child_is_species,
        # Label for the next-next level (the grandchildren the "want/with data/
        # complete" columns summarise); species children have no deeper level.
        "next_level": None if child_is_species else BROWSE_LEVELS[depth + 1],
        "next_level_plural": (
            None if child_is_species else BROWSE_LEVELS_PLURAL[BROWSE_LEVELS[depth + 1]]
        ),
        "node_counts": node_counts,
        "level_label": BROWSE_LEVELS[depth - 1] if depth else "",
    }
    return render(request, "moths/browse.html", context)


# --- Global taxon search (header autocomplete) -------------------------------

# Taxonomic ranks by descent depth; lower depth = higher (broader) rank, which
# is what wins ties per "no exact match -> the highest rank is chosen".
_SEARCH_RANKS = ("superfamily", "family", "subfamily", "genus", "species")
# Parsed flat search index, rebuilt when tax_summary.json changes.
_SEARCH_INDEX_CACHE: dict = {"mtime": None, "entries": []}


def _build_search_entries() -> list[dict]:
    """Flatten the hierarchical summary into a searchable taxon list.

    Every superfamily/family/subfamily/genus node becomes a "higher" entry that
    links to its browse page; every *present* species leaf (an image folder or
    cached images) becomes a "species" entry linking to its species view. Bucket
    keys ("(unknown)" / "-") are traversed but never themselves searchable.
    Each entry carries the lowercased strings it can be matched on plus its
    rank depth for ordering.
    """
    data = load_tax_summary()
    if not data:
        return []
    entries: list[dict] = []

    def add_higher(key, depth, segs):
        if key in (_UNKNOWN_TAXON, _NO_SUBFAMILY):
            return
        entries.append({
            "label": key,
            "detail": _SEARCH_RANKS[depth],
            "kind": "higher",
            "depth": depth,
            "url": reverse("moths:browse", args=["/".join(segs)]),
            "search": [key.lower()],
        })

    for sf_key, sf in data.get("superfamilies", {}).items():
        sf_seg = _browse_key_to_seg(sf_key)
        add_higher(sf_key, 0, [sf_seg])
        for fam_key, fam in sf.get("children", {}).items():
            fam_seg = _browse_key_to_seg(fam_key)
            add_higher(fam_key, 1, [sf_seg, fam_seg])
            for subf_key, subf in fam.get("children", {}).items():
                subf_seg = _browse_key_to_seg(subf_key)
                add_higher(subf_key, 2, [sf_seg, fam_seg, subf_seg])
                for gen_key, gen in subf.get("children", {}).items():
                    gen_seg = _browse_key_to_seg(gen_key)
                    add_higher(gen_key, 3, [sf_seg, fam_seg, subf_seg, gen_seg])
                    for tax_id, leaf in gen.get("children", {}).items():
                        if not (leaf.get("has_folder") or leaf.get("has_image")):
                            continue
                        species = (leaf.get("species") or "").strip()
                        common = (leaf.get("name") or "").strip()
                        entries.append({
                            "label": species or common or str(tax_id),
                            "detail": common if (species and common) else "species",
                            "kind": "species",
                            "depth": 4,
                            "url": reverse("moths:species_view", args=[str(tax_id)]),
                            "search": [
                                s.lower()
                                for s in (species, common, str(tax_id))
                                if s
                            ],
                        })
    return entries


def _search_entries() -> list[dict]:
    """Return the taxon search index, rebuilt only when the summary changes."""
    try:
        mtime = get_tax_summary_path().stat().st_mtime
    except OSError:
        mtime = None
    if _SEARCH_INDEX_CACHE["mtime"] != mtime:
        _SEARCH_INDEX_CACHE["entries"] = _build_search_entries()
        _SEARCH_INDEX_CACHE["mtime"] = mtime
    return _SEARCH_INDEX_CACHE["entries"]


def _search_tier(entry: dict, q: str) -> int | None:
    """Best match tier of ``entry`` for query ``q``: 0 exact, 1 prefix, 2 sub.

    ``None`` when nothing matches. Tiers rank match quality; ties break on the
    taxonomic rank (broader first) so a partial query resolves to the highest
    rank, per the spec.
    """
    tier = None
    for text in entry["search"]:
        if text == q:
            return 0
        if text.startswith(q):
            tier = 1 if tier is None else min(tier, 1)
        elif q in text:
            tier = 2 if tier is None else min(tier, 2)
    return tier


def taxon_search(request):
    """JSON autocomplete for the header search box.

    ``?q=`` is matched (case-insensitive) against superfamily/family/subfamily/
    genus names and species (scientific + common name + id). Results are ordered
    by match quality (exact, prefix, substring), then by broader taxonomic rank,
    then alphabetically, and capped to five. Each result carries the target
    ``url`` (browse page for a rank, species view for a species) so the client can
    navigate on Enter/selection.
    """
    q = (request.GET.get("q") or "").strip().lower()
    results: list[dict] = []
    if q:
        scored = []
        for entry in _search_entries():
            tier = _search_tier(entry, q)
            if tier is not None:
                scored.append((tier, entry["depth"], entry["label"].lower(), entry))
        scored.sort(key=lambda item: item[:3])
        results = [
            {
                "label": e["label"],
                "detail": e["detail"],
                "kind": e["kind"],
                "url": e["url"],
            }
            for _tier, _depth, _label, e in scored[:5]
        ]
    return JsonResponse({"results": results})


def observation_lookup(request):
    """Client-side iNaturalist observation lookup page.

    The page fetches the observation and its photos from the iNaturalist API in
    the browser (nothing is uploaded to our data), and asks
    :func:`species_info` whether we already track that taxon.
    """
    return render(request, "moths/observation_lookup.html", {})


def species_info(request, tax_id):
    """JSON: whether we track ``tax_id`` plus its starred reference image(s).

    ``starred`` lists the tax's starred images ordered like the species view
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
                reverse("moths:species_view", args=[tax_id]) if exists else None
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
    """Legacy list route: now folded into the unified species view.

    Kept so existing links (e.g. the index page's clickable counts) keep
    working; it redirects to :func:`species_view`, preserving any active filters.
    """
    url = reverse("moths:species_view", args=[tax_id])
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


def _make_row(image, data, starred, is_adult, flags=(), class_from_prediction=False,
              file_versions=None, box_from_prediction=False, needs_box=False):
    """Build a template row for one image from its cached pose ``data``.

    Any image with at least one computable metric (symmetry / pixels /
    sharpness) gets per-metric sub-scores and a general ``score`` = the minimum
    across the *available* sub-scores; every category then sorts by that score
    (starred first). Images with no metric at all fall back to an obs/photo
    caption and sort last. ``has_keypoints`` still marks the full keypoint adults
    (top-down/side/vertical), used only to describe the pose crop.

    The normalized crop (``is_norm``) drives both the thumbnail and the click
    target: any image with a normalized crop shows it and links to the
    normalized view; images without one show the plain thumbnail and link to the
    edit view. Keypoint adults get their pose crop, and every other boxed image
    (non-adult, or an image whose flag suppresses pose normalization) gets the
    simplified bounding-box crop.

    Metric values are shown as cached, even if the cache is a stale metric
    version or the keypoints changed since (a rebuild, or the normalized view,
    refreshes them) — a slightly stale number is more useful than an em dash.

    Per-plate "Todo" hints are derived here, each shown only when actually
    needed. ``review_stage`` (``class_from_prediction``: the stage/flags came
    from prediction, no hand ``.class``) and ``review_box``
    (``box_from_prediction``: the box/keypoints come from prediction, no hand
    label yet) drive a blue "review stage" / "review box" / "review stage, box"
    note.     ``needs_rebuild`` (via :func:`pose_row_needs_rebuild`, using
    ``file_versions`` — the file-level ``metric_versions`` of the cache — to
    judge staleness) drives an orange "rebuild" note. ``needs_box`` (computed by
    the caller from the live label/prediction folders) drives a red "needs a
    box" note: the image is normalizable (its flags don't suppress it) but has
    no box or keypoints from either source yet. None of these draw a border.

    Both review hints are read from the *live* label folders (not the cached
    ``source``), so they clear as soon as a hand label exists — decoupled from
    the metric rebuild, which the "rebuild" hint tracks separately.
    """
    pose = data.get("pose", POSE_NONE)
    no_norm = flags_suppress_normalization(flags)
    has_keypoints = is_adult and pose in KEYPOINT_POSES and not no_norm
    # Every image with a normalized crop shows its normalized thumbnail. Adults
    # with a keypoint pose (top-down/side/vertical) always have one; any other
    # image that has a bounding box gets the simplified bbox crop instead. A
    # cached ``sharpness`` (computed from that box) is the cheap "has a box"
    # signal, matching exactly when ``compute_normalization`` returns a crop.
    is_norm = has_keypoints or data.get("sharpness") is not None

    symmetry = data.get("symmetry")
    pixel_span = data.get("pixel_span")
    sharpness = data.get("sharpness")
    sym_score, pixel_score, sharp_score = score_components(symmetry, pixel_span, sharpness)
    # General score = min across whatever sub-scores exist (weakest link over the
    # available metrics). Computed live so it's right even when the cached
    # ``score`` predates a metric/formula change.
    available = [s for s in (sym_score, pixel_score, sharp_score) if s is not None]
    score = min(available) if available else None

    return {
        "image": image,
        "starred": image.filename in starred,
        "has_keypoints": has_keypoints,
        "has_metrics": bool(available),
        "is_norm": is_norm,
        "is_top_down": is_adult and pose == POSE_TOP_DOWN and not no_norm,
        "review_stage": bool(class_from_prediction),
        "review_box": bool(box_from_prediction),
        "needs_box": bool(needs_box),
        "needs_rebuild": pose_row_needs_rebuild(data, file_versions),
        "metric": symmetry,
        "sym_score": sym_score,
        "pixel_span": pixel_span,
        "pixel_score": pixel_score,
        "sharpness": sharpness,
        "sharpness_m": None if sharpness is None else sharpness / 1_000_000,
        "sharp_score": sharp_score,
        "score": score,
    }


def _build_unified_groups(tax_id, image_list, cached):
    """Return the ordered, non-empty unified groups for ``image_list``.

    Reads the pre-loaded ``cached`` pose data *without ever rebuilding it*: an
    image with no cached row (missing/stale cache) is classified live from its
    label file so it still lands in the right group. Cached metric values are
    shown as-is (even if their version is stale or the keypoints changed) —
    :func:`_make_row` never blanks them; a rebuild refreshes them. This keeps
    the species view a pure read — the caller shows a "needs rebuild" banner and
    the explicit Rebuild button regenerates the cache.

    Each image is placed in its flag subsection(s) when flagged, otherwise in its
    stage/pose subsection. A group's ``has_keypoints``/``is_top_down`` reflect
    whether *any* row qualifies (so mixed flag subsections still show the sort
    bar/score column). Keypoint-bearing groups sort starred-first then by score.
    Empty groups are dropped.
    """
    per_image = cached.get("images", {}) if cached else {}
    file_versions = cached.get("metric_versions") if cached else None
    starred = load_starred(tax_id)

    buckets = {gid: [] for gid, _label, _unknown in _unified_group_defs()}
    for image in image_list:
        stage, flags, class_source = get_class_and_flags_with_source(image.filename)
        # No cached row (missing/stale cache): classify live so grouping is
        # correct; the row then has no metrics (they show as em dashes) until a
        # rebuild populates them.
        data = per_image.get(image.filename)
        if data is None:
            data = {"pose": classify_pose(image.filename)}
        pose = data.get("pose", POSE_NONE)
        # Live box source (two cheap stats, cache-independent): the box comes
        # from prediction only while there's a prediction label and no hand
        # label yet, so "review box" clears the moment a hand box is drawn.
        has_label = get_label_path(image.filename).is_file()
        has_prediction = get_prediction_path(image.filename).is_file()
        box_from_prediction = has_prediction and not has_label
        # "Needs a box": the image could be normalized (its flags don't opt it
        # out) but there is no box or keypoints from either source yet, so a
        # human has to draw one before it can be cropped/normalized.
        needs_box = (
            not has_label
            and not has_prediction
            and not flags_suppress_normalization(flags)
        )
        row = _make_row(
            image, data, starred, stage == "Adult", flags,
            class_source == "prediction", file_versions, box_from_prediction,
            needs_box,
        )
        for gid in _target_group_ids(stage, flags, pose):
            if gid in buckets:
                buckets[gid].append(row)

    groups = []
    for gid, label, is_unknown_base in _unified_group_defs():
        rows = buckets[gid]
        if not rows:
            continue
        # Every category sorts by the general score (starred first, score desc);
        # rows without any metric (score None) fall to the end.
        rows = sorted(rows, key=_pose_row_sort_key)
        groups.append(
            {
                "id": gid,
                "label": label,
                "rows": rows,
                "has_keypoints": any(r["has_keypoints"] for r in rows),
                "is_top_down": any(r["is_top_down"] for r in rows),
                "is_unknown": is_unknown_base,
            }
        )
    return groups


def _tax_stub(request, tax_id, unknown):
    """Render the stub page for a taxon with nothing to show.

    ``unknown`` distinguishes a tax_id absent from ``data_summary.json`` (an
    unknown taxon) from a known one that simply has no data on disk yet.
    """
    info = get_name_info(tax_id)
    context = {
        "tax_id": tax_id,
        "unknown": unknown,
        "name": info.get("name") or "",
        "species": info.get("species") or "",
    }
    return render(request, "moths/tax_stub.html", context)


def species_view(request, tax_id):
    """Unified per-tax species view: images grouped by stage, adults by pose.

    This view is a **pure read** — it never rebuilds a cache in the background.
    The states it distinguishes:

    * ``tax_id`` not in ``data_summary.json`` -> the "unknown taxon" stub.
    * known but with no observations *and* no summary yet -> the "no data" stub.
    * known with a summary but no observations -> the normal page's "no data".
    * known with observations but a missing/stale summary or pose cache -> the
      images are shown (grouped live from their label files, metrics blank) with
      a "needs rebuild" banner; the explicit Rebuild button regenerates caches.
    * known with observations and current caches -> the full view.

    Groups (in order): Adult top-down/side/bottom-up/unclear/no-keypoints, then
    Pupa, Larvae, Egg, and finally images with no stage class ("Unknown
    stage"); empty groups are hidden. Any image with a normalized crop shows its
    normalized thumbnail and links to the normalized view; images without one
    (no bounding box) show the plain thumbnail and link to the edit view.
    Keypoint adults additionally show metrics and are score-sorted. Honors the
    stage/labeled filters.
    """
    # Unknown from data_summary's perspective -> stub, regardless of any stray
    # files. Guard on a non-empty index so a missing summary degrades to the old
    # behaviour instead of flagging every taxon as unknown.
    names = load_names()
    if names and str(tax_id) not in names:
        return _tax_stub(request, tax_id, unknown=True)

    images = scan_tax_images(tax_id)
    summary = load_summary(tax_id)  # None when missing OR at a stale version

    if not images:
        # Known but nothing on disk: a valid summary means "covered, no images"
        # (render the normal no-data page); otherwise there is nothing to show.
        if summary is None:
            return _tax_stub(request, tax_id, unknown=False)

    stage_filter, labeled_filter, pose_filter = _parse_filters(request)
    filtered = _filter_images(images, stage_filter, labeled_filter, pose_filter)

    # Read the pose cache as-is (never rebuilt here). A missing/stale summary or
    # pose cache, when there are images to show, drives the "needs rebuild"
    # banner; nothing is written until the user hits Rebuild.
    cached = load_pose_data_raw(tax_id)
    version_ok = pose_data_version_ok(cached)
    needs_rebuild = bool(images) and (summary is None or not version_ok)

    groups = _build_unified_groups(tax_id, filtered, cached)

    species_data = load_species_data(tax_id)
    context = {
        "tax_id": tax_id,
        "groups": groups,
        "total": len(filtered),
        "total_all": len(images),
        "has_images": bool(images),
        "needs_rebuild": needs_rebuild,
        "wing_stats_available": get_wing_stats_path(tax_id).is_file(),
        "side_wing_stats_available": get_side_wing_stats_path(tax_id).is_file(),
        "filter_desc": _filter_desc(stage_filter, labeled_filter, pose_filter),
        "is_filtered": bool(stage_filter or labeled_filter or pose_filter),
        "filter_qs": request.GET.urlencode(),
        "integration_sections": build_integration_groups(
            "species", species_data, context={"tax_id": tax_id}
        ),
    }
    return render(request, "moths/species_view.html", context)


@editor_required
@require_POST
def rebuild_poses(request, tax_id):
    """Recompute and cache the pose data for a tax_id, then return to the view.

    Preserves the active filters (carried in the query string) on redirect.
    """
    images = scan_tax_images(tax_id)
    if not images:
        raise Http404(f"No images found for tax_id {tax_id!r}")

    # Explicit, user-triggered rebuild: refresh both the pose cache and the
    # index summary (+ soft tree update) so the "needs rebuild" banner clears.
    build_pose_data(tax_id, images)
    build_summary(tax_id, images)

    url = reverse("moths:species_view", args=[tax_id])
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
    images = scan_tax_images(image.tax_id)
    stage_filter, labeled_filter, pose_filter = _parse_filters(request)
    filtered = _filter_images(images, stage_filter, labeled_filter, pose_filter)

    # Read the pose cache as-is (navigation never rebuilds); grouping falls back
    # to live classification for uncached images, matching the species view.
    cached = load_pose_data_raw(image.tax_id)

    def ordered(image_list):
        groups = _build_unified_groups(image.tax_id, image_list, cached)
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


def image_original(request, filename):
    """Original single-image view: the raw photo with editing controls.

    Editors get the full annotation/stage/flags controls; anonymous viewers see
    just the photo. The "Normalized" tab is shown only when the current
    annotation can produce a normalized crop (``norm_available``).
    """
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
    return render(request, "moths/image_original.html", context)


def image_normalized(request, filename):
    """Read-only normalized view with a star/unstar control.

    Shows the pose-normalized crop (side / vertical F→B) for adults with F&B
    keypoints, or the simplified bounding-box crop otherwise. Redirects to the
    original view only when the image has no annotation/box at all (no crop to
    show), so the back/tabs never dead-end.
    """
    image = find_image(filename)
    if image is None:
        raise Http404("Image not found")

    # Verify the keypoints against the live pose source and recompute this
    # image's normalized crop + scores if they changed. This is the only place
    # that recompute is triggered (the species view reads the cache as-is).
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
        url = reverse("moths:image_original", args=[filename])
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
    # 80% for both pose layouts. The bounding-box layout has no reference circle
    # (``circle_radius`` is ``None``).
    circle_radius = normalization.get("circle_radius")
    if circle_radius:
        circle_pct = circle_radius * 2 * 100
        circle_offset_pct = (0.5 - circle_radius) * 100
    else:
        circle_pct = None
        circle_offset_pct = None

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


@editor_required
@require_POST
def set_selection_stage(request, tax_id):
    """Classify an explicit set of a tax_id's images as ``stage``.

    Body is JSON ``{"stage": "Adult", "filenames": [...]}``. Every listed image
    that belongs to this tax_id is (re)labelled with the stage, regardless of any
    current class. Backs the species view's selection mode. Returns how many were
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
        # Stage drives the normalization layout, so drop the cached crop.
        clear_normalized(filename)
        count += 1
    if count:
        update_summary(tax_id)
    return JsonResponse({"ok": True, "count": count, "stage": stage})


@editor_required
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


@editor_required
@require_POST
def confirm_selection_prediction(request, tax_id):
    """Adopt the model prediction as the hand label for a set of images.

    Body is JSON ``{"filenames": [...]}``. For each listed image of this tax_id
    that has a predicted ``.class`` (stage and/or flags), those predicted values
    are written as the hand stage/flags — i.e. the prediction is "confirmed".
    Images without any prediction are skipped. Backs the species view's selection
    mode. Returns how many were confirmed.
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "invalid JSON"}, status=400)

    filenames = payload.get("filenames") or []
    if not isinstance(filenames, list):
        return JsonResponse({"error": "filenames must be a list"}, status=400)

    count = 0
    for filename in filenames:
        if not isinstance(filename, str):
            continue
        if tax_id_for_file(filename) != tax_id or find_image(filename) is None:
            continue
        pred_stage, pred_flags = get_predicted_class_and_flags(filename)
        if not pred_stage and not pred_flags:
            continue  # nothing predicted to confirm
        if pred_stage:
            set_image_class(filename, pred_stage)
        flags = set_image_flags(filename, pred_flags)
        set_pose_row_flags(tax_id, filename, flags)
        # Stage/flags can change the normalization layout, so drop the crop.
        clear_normalized(filename)
        count += 1
    if count:
        update_summary(tax_id)
    return JsonResponse({"ok": True, "count": count})


@editor_required
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
    # the species view shows "to be rebuild" until the next rebuild).
    mark_pose_row_stale(tax_id, filename)
    return JsonResponse({
        "ok": True,
        "count": len(objects),
        "normalized_available": compute_normalization(filename) is not None,
    })


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


@editor_required
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
        # Stage drives the normalization layout, so drop the cached crop.
        clear_normalized(filename)
        update_summary(tax_id_for_file(filename))
        return JsonResponse({
            "stage": None,
            "normalized_available": compute_normalization(filename) is not None,
        })
    if stage not in STAGES:
        return JsonResponse({"error": "invalid stage"}, status=400)
    set_image_class(filename, stage)
    clear_normalized(filename)
    update_summary(tax_id_for_file(filename))
    return JsonResponse({
        "stage": stage,
        "normalized_available": compute_normalization(filename) is not None,
    })


@editor_required
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
    tax_id = tax_id_for_file(filename)
    set_pose_row_flags(tax_id, filename, flags)
    # Flags move the image between poses-page groups, so the representative
    # thumbnail may change (a now-flagged image can win/lose its group).
    refresh_tax_thumbnail(tax_id)
    return JsonResponse({
        "flags": flags,
        "normalized_available": compute_normalization(filename) is not None,
    })


@editor_required
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


@editor_required
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
