# Generated by Django 4.2.8 on 2024-12-23 06:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("scanner", "0002_metrics_price_change_1hr_metrics_price_change_7d"),
    ]

    operations = [
        migrations.AddField(
            model_name="metrics",
            name="twenty_min_relative_volume",
            field=models.FloatField(blank=True, null=True),
        ),
    ]