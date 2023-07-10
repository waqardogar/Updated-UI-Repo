from django.urls import path,include
from . import views
from pathplanning import views as v_p
urlpatterns=[
    path("",views.dashboard),
    path("pp/", v_p.pathplanning,name="pathplanning"),
    path("ppNFZ/", v_p.start_mission,name="start_mission"),
    path("video-upload/",views.VideoUpload,name="VideoUpload"),
    path("localize/",views.localization,name="localization"),
    path("doenload/",views.download_image,name="download")
    
]