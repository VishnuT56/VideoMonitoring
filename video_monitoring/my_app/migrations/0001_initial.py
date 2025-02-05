# Generated by Django 4.0.1 on 2024-02-07 05:41

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Criminal',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('criminal_name', models.CharField(max_length=100)),
                ('date_of_birth', models.DateField()),
                ('gender', models.CharField(max_length=100)),
                ('photo1', models.CharField(max_length=250)),
                ('photo2', models.CharField(max_length=250)),
                ('case', models.CharField(max_length=100)),
                ('identification_mark1', models.CharField(max_length=100)),
                ('identification_mark2', models.CharField(max_length=100)),
                ('place', models.CharField(max_length=100)),
                ('post', models.CharField(max_length=100)),
                ('pin', models.IntegerField()),
                ('district', models.CharField(max_length=100)),
                ('state', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Login',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_name', models.CharField(max_length=100)),
                ('password', models.CharField(max_length=100)),
                ('type', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='SocialAlert',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('alert', models.CharField(max_length=100)),
                ('date', models.DateField()),
                ('distance', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Emotion',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('emotion', models.CharField(max_length=100)),
                ('date', models.DateField()),
                ('time', models.CharField(max_length=100)),
                ('CRIMINAL', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='my_app.criminal')),
            ],
        ),
        migrations.CreateModel(
            name='CriminalLogs',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('photo', models.CharField(max_length=250)),
                ('CRIMINAL', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='my_app.criminal')),
            ],
        ),
    ]
