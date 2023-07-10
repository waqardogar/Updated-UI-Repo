from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
from django.conf import settings
from django.http import HttpResponse
import json
from . models import video
import cv2
import numpy as np
import random
import os
from django.urls import reverse
import requests
from  .image_stitching import build_mosaic
def dashboard(request):
    return render (request,"videostraem/video.html")
def VideoUpload(request):
    if request.method == 'POST' and request.FILES['video']:
        video_file = request.FILES['video']
        area_name = request.POST.get('area-name')
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'videos'))
        print("FS->",fs)
        filename = fs.save(video_file.name, video_file)
        video_path = fs.url(filename)
        Video = video(AreaName = area_name , file_name = filename , file_path = video_path)
        Video.save()
        def frame_extractor(path,save_dir,current_frame,gap=50):
            temp=1
            vid = cv2.VideoCapture(path)
            while True:
                flag , frame = vid.read()
                if flag==False:
                    vid.release()
                    break
                else:
                    if(current_frame%gap==0):
                        cv2.imwrite(f"{save_dir}/{temp}.png",frame)
                        temp+=1
                current_frame+=1

        current_frame = 0
        path='.'+video_path+"/videos"
        segments = path.split('/')
        new_segments = [segments[0],  segments[1],segments[-1],segments[2]]
        new_path = '/'.join(new_segments)
        print(new_path)
        save_dir = os.path.join(settings.MEDIA_ROOT, 'dataset/'+area_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            save_dir = os.path.join(settings.MEDIA_ROOT, 'dataset/'+area_name)
        frame_extractor(new_path, save_dir, current_frame,gap=130)

        #---------------------------------
        raw_img_dir      = save_dir
        ftr_detector     = 'sift'
        num_keypoints    = 10000
        # save_mosaic_dir  = '/media/mosaic/'
        # save_mosaic_dir = "../media/mosaic"
        # save_mosaic_dir = os.path.join(settings.MEDIA_ROOT, 'static/googlemaps/mosaic/'+area_name)
        save_mosaic_dir="E:\Talha-Git-Rep-UI\FYP\dnda2\googlemaps\static\googlemaps\mosaic/"+area_name
        mosaic_base_name = area_name
        flag_save        = True
        #----------------------------------
        input_images = os.listdir(raw_img_dir)
        temp = [(n,int(i.split(".")[0])) for n,i in enumerate(input_images)]
        temp.sort(key=lambda x:x[1])
        input_images = [input_images[i[0]] for i in temp]
        input_images_paths = []
        for i in input_images:
            name = i.split('.')[0]
            ext = i.split('.')[1]
            path = f"{raw_img_dir}/{name}.{ext}"
            input_images_paths.append(path) 
        map_path = build_mosaic(input_images_paths,save_mosaic_dir,area_name,num_keypoints,flag_save)
        map_path2 = "googlemaps/mosaic/"+area_name+".png"
        return render(request,"googlemaps/map.html",{'success': True,"map_path":map_path2})
    return render(request,"googlemaps/map.html",{'success': 2})



def localization(request):
    import os
    import numpy
    import cv2 as cv
    def process_source_image(source_img):
        img_bgr = cv.imread(source_img)
        height,width=img_bgr.shape[:2]
        img_bgr=cv2.resize(img_bgr, (int(height/4),int(width/4)))
        img_map = cv2.imread(source_img)
        height,width=img_map.shape[:2]
        img_map=cv2.resize(img_map, (int(height/4),int(width/4)))
        img_map=cv2.cvtColor(img_map,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints_1, descriptors_1 = sift.detectAndCompute(img_map, None)
        img_draw_1 = cv2.drawKeypoints(img_map, keypoints_1, img_map, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return img_bgr, img_map, keypoints_1, descriptors_1
    def process_target_image(target_img, img_map, keypoints_1, descriptors_1):
        img_of_map = cv2.imread(target_img)
        height,width=img_of_map.shape[:2]
        img_of_map=cv2.resize(img_of_map, (int(height/4),int(width/4)))
        img_map=cv2.cvtColor(img_of_map,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints_2, descriptors_2 = sift.detectAndCompute(img_of_map, None)
        img_draw = cv2.drawKeypoints(img_of_map, keypoints_2, img_of_map,
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        bf = cv2.BFMatcher()
        matches_bf = bf.knnMatch(descriptors_1, descriptors_2, k=2)
        good_bf = []
        for m, n in matches_bf:
            if m.distance < 0.75 * n.distance:
                good_bf.append([m])
        img3_bf = cv2.drawMatchesKnn(img_map, keypoints_1, img_of_map, keypoints_2, good_bf, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img_of_map, keypoints_2, good_bf, img3_bf
    def Sift_Localization(source_img, target_img):
        img_bgr, img_map, keypoints_1, descriptors_1 = process_source_image(source_img)
        img_of_map, keypoints_2, good_bf, img3_bf = process_target_image(target_img, img_map, keypoints_1, descriptors_1)
        src_pts = np.float32([keypoints_1[m[0].queryIdx].pt for m in good_bf]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_2[m[0].trainIdx].pt for m in good_bf]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = img_of_map.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, np.linalg.inv(M))
        rect = cv2.boundingRect(dst)
        x, y, w, h = rect
        img_matched = cv2.rectangle(img_bgr.copy(), (x, y), (x + w, y + h), (0, 255, 0), 1)
        midpoint_x = int((x + x + w) / 2)
        midpoint_y = int((y + y + h) / 2)
        
        circle = cv2.circle(img_matched, (midpoint_x, midpoint_y), 6, (0, 0, 255), -1)
        circle2=cv2.resize(circle,(500,600))
        cv2.imshow("circle",circle2)
        cv2.waitKey(0)


    import os

    def get_file_names(directory):
        file_names = []
        for file_name in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file_name)):
                file_names.append(file_name)
        return file_names
    all_objects=video.objects.all()
    last_name = []
    for obj in all_objects:
        last_name.append((obj.AreaName))
    mosaic_path = "E:\Talha-Git-Rep-UI\FYP\dnda2\googlemaps\static\googlemaps\mosaic/"+last_name[-1]+".png"
    frame_path = "E:\Talha-Git-Rep-UI\FYP\dnda2\media\dataset/"+last_name[-1]+"/"
    file_names = get_file_names(frame_path)
    random_pic = random.choice(file_names)
    file_path =  frame_path+random_pic
    print("Mosaic Path -> ",mosaic_path)
    print("file path-> ", file_path)
    sift=Sift_Localization(mosaic_path, file_path)
    return render(request,"googlemaps/map.html",{'success': True})