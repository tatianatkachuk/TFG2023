from django.db import models

# Create your models here.
class Marker(models.Model):
    name = models.CharField(max_length=100, blank=False, default='')
    markerLat = models.FloatField()
    markerLong = models.FloatField()