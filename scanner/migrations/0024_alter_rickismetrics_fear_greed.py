# Generated by Django 5.1.4 on 2025-06-06 19:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('scanner', '0023_rickismetrics_fear_greed'),
    ]

    operations = [
        migrations.AlterField(
            model_name='rickismetrics',
            name='fear_greed',
            field=models.FloatField(null=True),
        ),
    ]
