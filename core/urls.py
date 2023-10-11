from django.contrib import admin
from django.urls import path, include, re_path 
from .views import HomeView
# from deepMeteorology.views import add_Marker, getMarkers

urlpatterns = [
    path('admin/', admin.site.urls),
    re_path(r'^', include('deepMeteorology.urls')),
   
]
