# Generated by Django 5.2.3 on 2025-07-23 19:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('scanner', '0036_memorytrade'),
    ]

    operations = [
        migrations.RenameField(
            model_name='memorytrade',
            old_name='chart_image',
            new_name='chart_image_5min',
        ),
        migrations.AddField(
            model_name='memorytrade',
            name='chart_image_30min',
            field=models.ImageField(blank=True, null=True, upload_to='memory_charts/'),
        ),
        migrations.AlterField(
            model_name='memorytrade',
            name='sl_percent',
            field=models.FloatField(default=1.0),
        ),
        migrations.AlterField(
            model_name='memorytrade',
            name='tp_percent',
            field=models.FloatField(default=2.0),
        ),
    ]
