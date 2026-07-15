"""
URL configuration for moths_list project.

The `urlpatterns` list routes URLs to views. For more information see:
https://docs.djangoproject.com/en/5.1/topics/http/urls/
"""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("moths.urls")),
]
