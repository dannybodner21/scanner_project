from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path('five-min-update/', views.five_min_update, name='five-min-update'),
    path('daily-update/', views.daily_update, name='daily-update'),
    path('thirty-min-pattern-check/', views.thirty_min_pattern_check, name='thirty-min-pattern-check'),
    path('five-min-pattern-check/', views.five_min_pattern_check, name='five-min-pattern-check'),
    path('support-resistance/', views.support_resistance, name='support-resistance'),
    path('check-support-resistance/', views.check_support_resistance, name='check-support-resistance'),
    path('finn/', views.finn, name='finn'),

    path('post-metrics-to-bot/', views.post_metrics_to_bot, name='post-metrics-to-bot'),
    path('run-metrics-and-scan/', views.run_metrics_and_scan, name='run-metrics-and-scan'),


    path('download/<str:filename>/', views.download_file, name='download_file'),
]
