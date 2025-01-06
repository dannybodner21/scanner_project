from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path('setup-scheduler/', views.setup_scheduler, name='setup-scheduler'),
    path('five-min-update/', views.five_min_update, name='five-min-update'),
]
