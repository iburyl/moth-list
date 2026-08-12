"""
URL configuration for moths_list project.

The `urlpatterns` list routes URLs to views. For more information see:
https://docs.djangoproject.com/en/5.1/topics/http/urls/
"""

from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import include, path

urlpatterns = [
    # Django admin, deliberately not on the well-known /admin/ path.
    # disable admin for now, until the use case is clear
    # path("rule/", admin.site.urls),

    # Login / logout for the edit split (viewing is public; editing requires a
    # logged-in user). Logout is POST-only in Django 5.
    path("accounts/login/", auth_views.LoginView.as_view(), name="login"),
    path("accounts/logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("", include("moths.urls")),
]
