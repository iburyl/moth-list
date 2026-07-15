from django.urls import path

from . import views

app_name = "moths"

urlpatterns = [
    path("", views.index, name="index"),
    path("tax/<str:tax_id>/", views.tax_detail, name="tax_detail"),
    path("tax/<str:tax_id>/add-to-train", views.add_to_train, name="add_to_train"),
    path("image/<path:filename>/", views.image_detail, name="image_detail"),
    path("stage/<path:filename>", views.set_stage, name="set_stage"),
    path("subset/<path:filename>", views.set_subset, name="set_subset"),
    path("label/<path:filename>", views.save_label, name="save_label"),
    path("images/<path:filename>", views.serve_image, name="serve_image"),
    path("thumbnails/<path:filename>", views.serve_thumbnail, name="serve_thumbnail"),
]
