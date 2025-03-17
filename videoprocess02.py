import cv2
import mediapipe as mp
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path="D:\\Python\\mediapipe\\squat.mp4"#path to your video
#This program will generate a csv file with 
#Human body points in current file path,

mp_pose=mp.solutions.pose
mp_drawing=mp.solutions.drawing_utils
#DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 10, 1)   #设定线条
pose=mp_pose.Pose(static_image_mode=False,
                  model_complexity=1,#2
                  smooth_landmarks=True,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5)

def Angle(p1x,p1y,p2x,p2y,p3x,p3y):
    angle=np.arccos( ((p2x-p1x)*(p2x-p3x)+(p2y-p1y)*(p2y-p3y) )/ (np.sqrt(((p2x-p1x)**2+(p2y-p1y)**2)) * np.sqrt((p2x-p3x)**2+(p2y-p3y)**2)))
    return angle*360/(2*np.pi)

def process_frame(img):
    start_time=time.time()
    h,w=img.shape[0],img.shape[1]
    img_RGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=pose.process(img_RGB)
    
    scaler=1
    point_list=[]
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
        for i in range(33):
            
            cx=int(results.pose_landmarks.landmark[i].x*w)
            cy=int(results.pose_landmarks.landmark[i].y*h)
            cz=results.pose_landmarks.landmark[i].z
            point_list.append([cx,cy,cz])

            
            radius=10
            if i==0:#鼻尖
                img=cv2.circle(img,(cx,cy),radius,(0,0,255),-1)
            elif i in [11,12]:#肩膀
                img=cv2.circle(img,(cx,cy),radius,(223,155,6),-1)
            elif i in [23,24]:#髋关节
                img=cv2.circle(img,(cx,cy),radius,(1,240,255),-1)
            elif i in [13,14]:#胳膊肘
                img=cv2.circle(img,(cx,cy),radius,(140,47,240),-1)
            elif i in [25,26]:#膝盖
                img=cv2.circle(img,(cx,cy),radius,(0,0,255),-1)
            elif i in [15,16,27,28]:#手腕脚腕
                img=cv2.circle(img,(cx,cy),radius,(223,155,60),-1)
            elif i in [17,19,21]:#左手
                img=cv2.circle(img,(cx,cy),radius,(94,218,121),-1)
            elif i in [18,20,22]:#右手
                img=cv2.circle(img,(cx,cy),radius,(16,144,247),-1)
            elif i in [27,29,31]:#左脚
                img=cv2.circle(img,(cx,cy),radius,(29,123,243),-1)
            elif i in [28,30,32]:#右脚
                img=cv2.circle(img,(cx,cy),radius,(193,182,255),-1)
            elif i in [9,10]:#嘴
                img=cv2.circle(img,(cx,cy),radius,(205,235,255),-1)
            elif i in [1,2,3,4,5,6,7,8]:#眼睛脸颊
                img=cv2.circle(img,(cx,cy),radius,(94,218,121),-1)
            else:#其他关键点
                img=cv2.circle(img,(cx,cy),radius,(0,255,0),-1)

        try:
            #23,25,27
            c1x=int(results.pose_landmarks.landmark[23].x*w)
            c1y=int(results.pose_landmarks.landmark[23].y*h)
            c2x=int(results.pose_landmarks.landmark[25].x*w)
            c2y=int(results.pose_landmarks.landmark[25].y*h)
            c3x=int(results.pose_landmarks.landmark[27].x*w)
            c3y=int(results.pose_landmarks.landmark[27].y*h)
            
            angle=Angle(c1x,c1y,c2x,c2y,c3x,c3y)
            img=cv2.putText(img,'Angle: '+str(int(angle)),(25*scaler,80*scaler),cv2.FONT_HERSHEY_SIMPLEX,1.25*scaler,(255,0,0),3)
            
            
        except:
            img=cv2.putText(img,'no legs',(255*scaler,100*scaler),cv2.FONT_HERSHEY_SIMPLEX,1.25*scaler,(255,0,0),3)
            pass
    
    else:
        failure_str='NO PERSON'
        img=cv2.putText(img,failure_str,(255*scaler,100*scaler),cv2.FONT_HERSHEY_SIMPLEX,1.25*scaler,(255,0,0),3)
        

    end_time=time.time()
    FPS=1/(end_time-start_time)
    img=cv2.putText(img,'FPS: '+str(int(FPS)),(25*scaler,50*scaler),cv2.FONT_HERSHEY_SIMPLEX,1.25*scaler,(255,0,0),3)
    
    pls=pd.Series(point_list)
    return img,angle,pls

def generate_video(input_path):
    filename=input_path.split('\\')[-1]
    outputname='out-'+filename
    
    print('Processing',filename)
    cap=cv2.VideoCapture(input_path)
    frame_count=0
    index=1
    f=pd.DataFrame()
    while (cap.isOpened()):
        success,frame=cap.read()
        frame_count+=1
        if not success:
            print('ERROR in processing video!')
            break
    cap.release()
    print('Total count frame:',frame_count)
    cap=cv2.VideoCapture(input_path)
    frame_size=(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    #fourcc=cv2.VideoWriter_fourcc('F','L','V','1')
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    fps=cap.get(cv2.CAP_PROP_FPS)
        
    out=cv2.VideoWriter(outputname,fourcc,fps,(int(frame_size[0]),int(frame_size[1])))
    
    anglelist=[]
    with tqdm(total=frame_count-1) as pbar:
        try:
            while(cap.isOpened()):
                success,frame=cap.read()
                if not success:
                    print('bar error 1')
                    break
                try:
                    frame,angle,point_list=process_frame(frame)
                    anglelist.append(angle)
                    f[f'frame{index}']=point_list
                except:
                    print('error in frame 1')
                    pass
                    
                if success==True:
                    out.write(frame)
                    pbar.update(1)
                    index+=1
                        
        except:
            print('bar error 2')
            pass
        
    cv2.destroyAllWindows()
    out.release()
    cap.release()
    f.to_csv(filename+'-pointlist'+'.csv')
    print('video saved',outputname)
    plt.plot(range(len(anglelist)),anglelist)
    plt.show()
    
generate_video(file_path)

#TODO(dizisama): 关键点集的合并报错但不影响使用,之后改一改看能不能一次性合并
#TODO(dizisama): 杂七杂八的到时候重构一下
#TODO(dizisama): 搞清楚返回的图像z轴和真实世界z轴的区别
