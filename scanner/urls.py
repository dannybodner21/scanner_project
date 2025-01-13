from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path('five-min-update/', views.five_min_update, name='five-min-update'),

    path('test/', views.test, name='test'),
]
