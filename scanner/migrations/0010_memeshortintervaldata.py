# Generated by Django 4.2.8 on 2025-01-07 21:21

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("scanner", "0009_memecoin_mememetric"),
    ]

    operations = [
        migrations.CreateModel(
            name="MemeShortIntervalData",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("timestamp", models.DateTimeField()),
                ("price", models.DecimalField(decimal_places=8, max_digits=20)),
                ("volume_5min", models.DecimalField(decimal_places=2, max_digits=20)),
                (
                    "circulating_supply",
                    models.DecimalField(
                        blank=True, decimal_places=2, max_digits=20, null=True
                    ),
                ),
                (
                    "coin",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="meme_short_interval_data",
                        to="scanner.memecoin",
                    ),
                ),
            ],
        ),
    ]