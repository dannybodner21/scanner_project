from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("index/", views.index, name="indexold"),
    path("", views.index_view, name="index"),

    path('daily-update/', views.daily_update, name='daily-update'),
    path('thirty-min-pattern-check/', views.thirty_min_pattern_check, name='thirty-min-pattern-check'),
    path('five-min-pattern-check/', views.five_min_pattern_check, name='five-min-pattern-check'),
    path('support-resistance/', views.support_resistance, name='support-resistance'),
    path('check-support-resistance/', views.check_support_resistance, name='check-support-resistance'),

    path('download/<str:filename>/', views.download_file, name='download_file'),
    path('download-parquet/<str:filename>/', views.download_parquet_file, name='download_parquet_file'),

    # RickisScanner URLs
    path("api/ricki/latest/", views.get_hod_movers, name="get-hod-movers"),
    path("short-intervals/", views.short_interval_summary, name="short-intervals"),

    path('five-min-update/', views.five_min_update, name='five-min-update'),
    path("api/open-trades/", views.get_open_trades, name="open-trades"),
    path("api/closed-trades/", views.get_closed_trades, name="closed-trades"),
    path("api/model-results/", views.get_model_results, name="model-results"),
    path("api/memory-trades/", views.get_memory_trades, name="memory-trades"),
    path("chart-image/<str:coin>/", views.serve_chart_image, name="serve-chart-image"),

    path("api/patterns/", views.get_patterns, name="patterns"),
    path('update-patterns/', views.run_update_patterns, name='update_patterns'),

    path("api/metrics-health-daily/", views.daily_metrics_health, name="daily-metrics-health"),

    path("run-live-pipeline/", views.run_live_pipeline_view, name="run_live_pipeline"),

    path('trades/', views.open_trades_view, name='open_trades'),
    path('api/live-trades/', views.live_trades, name='live-trades'),

    # binance price pull url
    path("internal/import-candles/", views.import_candles, name="import-candles"),

]
