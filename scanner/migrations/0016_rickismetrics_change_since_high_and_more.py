# Generated by Django 5.1.4 on 2025-05-03 21:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('scanner', '0015_rickismetrics_market_sentiment_label_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='rickismetrics',
            name='change_since_high',
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name='rickismetrics',
            name='change_since_low',
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name='rickismetrics',
            name='volume_mc_ratio',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='rickismetrics',
            name='avg_volume_1h',
            field=models.DecimalField(decimal_places=2, max_digits=30, null=True),
        ),
        migrations.AlterField(
            model_name='rickismetrics',
            name='change_1h',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='rickismetrics',
            name='change_24h',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='rickismetrics',
            name='change_5m',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='rickismetrics',
            name='high_24h',
            field=models.DecimalField(decimal_places=10, max_digits=20, null=True),
        ),
        migrations.AlterField(
            model_name='rickismetrics',
            name='low_24h',
            field=models.DecimalField(decimal_places=10, max_digits=20, null=True),
        ),
    ]
