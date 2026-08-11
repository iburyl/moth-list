from django.urls import path

from . import views

app_name = "moths"

urlpatterns = [
    path("", views.index, name="index"),
    path("search/", views.taxon_search, name="taxon_search"),
    path("browse/", views.browse, name="browse"),
    path("browse/<path:lineage>/", views.browse, name="browse"),
    path("observation/", views.observation_lookup, name="observation_lookup"),
    path(
        "observation/species/<str:tax_id>/",
        views.species_info,
        name="species_info",
    ),
    path("tax/<str:tax_id>/", views.tax_detail, name="tax_detail"),
    path("tax/<str:tax_id>/poses/", views.pose_view, name="pose_view"),
    path(
        "tax/<str:tax_id>/poses/rebuild",
        views.rebuild_poses,
        name="rebuild_poses",
    ),
    path(
        "tax/<str:tax_id>/wing-stats.png",
        views.serve_wing_stats,
        name="serve_wing_stats",
    ),
    path(
        "tax/<str:tax_id>/side-wing-stats.png",
        views.serve_side_wing_stats,
        name="serve_side_wing_stats",
    ),
    path(
        "tax/<str:tax_id>/set-selection-stage",
        views.set_selection_stage,
        name="set_selection_stage",
    ),
    path(
        "tax/<str:tax_id>/delete-selection",
        views.delete_selection,
        name="delete_selection",
    ),
    path("image/<path:filename>/", views.image_edit, name="image_edit"),
    path(
        "image/<path:filename>/normalized",
        views.image_normalized,
        name="image_normalized",
    ),
    path("stage/<path:filename>", views.set_stage, name="set_stage"),
    path("flags/<path:filename>", views.set_flags, name="set_flags"),
    path("details/<path:filename>", views.set_details, name="set_details"),
    path("star/<path:filename>", views.set_star, name="set_star"),
    path("label/<path:filename>", views.save_label, name="save_label"),
    path("images/<path:filename>", views.serve_image, name="serve_image"),
    path("thumbnails/<path:filename>", views.serve_thumbnail, name="serve_thumbnail"),
    path(
        "norm-thumbnails/<path:filename>",
        views.serve_norm_thumbnail,
        name="serve_norm_thumbnail",
    ),
    path("norm-images/<path:filename>", views.serve_norm_image, name="serve_norm_image"),
]
