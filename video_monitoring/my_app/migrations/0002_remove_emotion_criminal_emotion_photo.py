# Generated by Django 4.0.1 on 2024-02-07 10:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('my_app', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='emotion',
            name='CRIMINAL',
        ),
        migrations.AddField(
            model_name='emotion',
            name='photo',
            field=models.CharField(default=1, max_length=250),
            preserve_default=False,
        ),
    ]
