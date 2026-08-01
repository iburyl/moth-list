from django.urls import path

from . import views

app_name = "moths"

urlpatterns = [
    path("", views.index, name="index"),
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
        "tax/<str:tax_id>/add-to-stage/<str:stage>",
        views.add_to_stage,
        name="add_to_stage",
    ),
    path("image/<path:filename>/", views.image_edit, name="image_edit"),
    path(
        "image/<path:filename>/normalized",
        views.image_normalized,
        name="image_normalized",
    ),
    path("stage/<path:filename>", views.set_stage, name="set_stage"),
    path("flags/<path:filename>", views.set_flags, name="set_flags"),
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
