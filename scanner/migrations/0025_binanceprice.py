# Generated by Django 5.1.4 on 2025-06-11 13:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('scanner', '0024_alter_rickismetrics_fear_greed'),
    ]

    operations = [
        migrations.CreateModel(
            name='BinancePrice',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('coin', models.CharField(max_length=20)),
                ('timestamp', models.DateTimeField(unique=True)),
                ('open', models.DecimalField(decimal_places=10, max_digits=20, null=True)),
                ('high', models.DecimalField(decimal_places=10, max_digits=20, null=True)),
                ('low', models.DecimalField(decimal_places=10, max_digits=20, null=True)),
                ('close', models.DecimalField(decimal_places=10, max_digits=20, null=True)),
                ('volume', models.DecimalField(decimal_places=15, max_digits=30, null=True)),
            ],
            options={
                'indexes': [models.Index(fields=['coin', 'timestamp'], name='scanner_bin_coin_89a0b7_idx')],
                'unique_together': {('coin', 'timestamp')},
            },
        ),
    ]
