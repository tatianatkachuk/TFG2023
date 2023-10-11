# from django.conf.urls import url 
from django.urls import re_path, path
from django.views.generic import RedirectView
from deepMeteorology import views 
 
urlpatterns = [ 
    path('', RedirectView.as_view(url=r'markers/')),
    re_path(r'markers/', views.markers_list), 
    re_path(r'predict/', views.predict),
]