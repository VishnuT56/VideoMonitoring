from django.db import models

# Create your models here.
class Login(models.Model):
    user_name=models.CharField(max_length=100)
    password = models.CharField(max_length=100)
    type = models.CharField(max_length=100)

class Criminal(models.Model):
    criminal_name=models.CharField(max_length=100)
    date_of_birth=models.DateField()
    gender=models.CharField(max_length=100)
    photo1=models.CharField(max_length=250)
    photo2=models.CharField(max_length=250)
    case= models.CharField(max_length=100)
    identification_mark1= models.CharField(max_length=100)
    identification_mark2 = models.CharField(max_length=100)
    place = models.CharField(max_length=100)
    post = models.CharField(max_length=100)
    pin = models.IntegerField()
    district = models.CharField(max_length=100)
    state = models.CharField(max_length=100)

class Emotion (models.Model):
    photo = models.CharField(max_length=250)
    emotion = models.CharField(max_length=100)
    date = models.DateField()
    time = models.CharField(max_length=100)

class CriminalLogs(models.Model):
    CRIMINAL=models.ForeignKey(Criminal,on_delete=models.CASCADE)
    date = models.DateField()
    time = models.TimeField()
    photo=models.CharField(max_length=250)

class SocialAlert(models.Model):
    alert=models.CharField(max_length=100)
    date = models.DateField()
    time = models.TimeField()
    photo = models.CharField(max_length=250)
    violations_count=models.CharField(max_length=100)







