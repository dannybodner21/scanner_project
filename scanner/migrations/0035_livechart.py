# Generated by Django 5.2.3 on 2025-07-20 23:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('scanner', '0034_livepricesnapshot'),
    ]

    operations = [
        migrations.CreateModel(
            name='LiveChart',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('coin', models.CharField(max_length=20, unique=True)),
                ('timestamp', models.DateTimeField()),
                ('image', models.ImageField(upload_to='live_charts/')),
            ],
        ),
    ]
