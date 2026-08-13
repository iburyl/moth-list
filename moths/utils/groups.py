"""Unified stage/pose/flag group definitions for the species view.

A single source of truth for *how images are grouped and in what order* on a
species' poses page. It lives here (rather than in ``views``) so non-view code
— notably the representative-thumbnail chooser in :mod:`moths.utils.posedata` —
can reproduce the exact same group order without importing the view layer.

A flagged image lands in a flag subsection per flag it carries (so it can
appear in more than one and is removed from its pose/stage subsection);
unflagged images fall into their stage/pose subsection. Adults are split by
predicted pose; other stages have a single base group; images without a stage
class fall into "unknown".
"""

from .annotations import (
    POSE_BOTTOM_UP,
    POSE_NONE,
    POSE_SIDE,
    POSE_TOP_DOWN,
    POSE_UNCLEAR,
)
from .classes import flag_applies_to_stage, flags_for_stage

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

# Stage keys that get their own base/flag subsections (anything else -> unknown).
STAGE_KEYS = ("Adult", "Pupa", "Larva", "Egg")


def _flag_gid(stage_key, flag):
    """Group id for a stage's flag subsection, e.g. ``adult_Pinned``."""
    return f"{stage_key}_{flag}"


# Adult base subsections, in display order: the "real" pose groups come first,
# then (spliced by :func:`unified_group_defs`) the Adult flag subsections, then
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


def unified_group_defs():
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


def unified_group_order():
    """Just the group ids from :func:`unified_group_defs`, in display order."""
    return [gid for gid, _label, _unknown in unified_group_defs()]


def target_group_ids(stage, flags, pose):
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
