

import numpy as np 
import pandas as pd 
import os
import glob
import csv

POSE_PATH_DIR = "/home/negar/Documents/Datasets/ChicagoWild++/mediapipe_res_chicago"
treshold = 0.4

u = 0
failed = open("failedhands.txt", "a")

with open('./sign_hand_detection_wild++.csv', 'w') as f:
    writer = csv.writer(f)
    
    for path, subdirs, files in os.walk(POSE_PATH_DIR):
        for file in subdirs:
            for path2, subdirs2, files2 in os.walk(POSE_PATH_DIR+"/"+file):
                for file2 in subdirs2:
                    print(path2,file2)
                    # if file in ['misc_2', 'deafvideo_3', 'deafvideo_2', 'misc_1', 'youtube_1', 'deafvideo_1', 'gallaudet', 'deafvideo_6', 'youtube_2', 'awti', 'aslthat', 'youtube_6', 'deafvideo_5', 'aslized', 'youtube_3', 'deafvideo_4', 'youtube_4', 'youtube_5']:
                    #         continue
                    left_joints = []
                    right_joints = []
                    if len(sorted(glob.glob(path2+"/"+file2+"/*"))) ==1:
                        data = [path2,file2,"r"]
                        writer.writerow(data)
                        continue

                    for frame in sorted(glob.glob(path2+"/"+file2+"/*")):
                        
                        frame_np = np.load(frame,allow_pickle=True)
                        
                        if frame_np is None or len(frame_np) ==0:
                            continue

                        right_hand = frame_np[532:553][:]
                        left_hand = frame_np[511:532][:]
                        
                        # print("right: ",right_hand)
                        # print("left: ",left_hand)

                        right_hand[:,:2] = right_hand[:,:2] - (sum(right_hand)/len(right_hand))[:2]
                        left_hand[:,:2] = left_hand[:,:2] - (sum(left_hand)/len(left_hand))[:2]

                        left_joints.append(left_hand)
                        right_joints.append(right_hand)

                    left_joints = np.array(left_joints)
                    right_joints = np.array(right_joints)

                    left_conf = np.mean(np.mean(left_joints,axis=0),axis=0)[3]
                    right_conf = np.mean(np.mean(right_joints,axis=0),axis=0)[3]

                    left_var = sum(sum(abs(left_joints[left_joints.shape[0]//10+1:]-left_joints[left_joints.shape[0]//10:-1])))
                    right_var = sum(sum(abs(right_joints[right_joints.shape[0]//10+1:]-right_joints[right_joints.shape[0]//10:-1])))

                    if left_conf<treshold and right_conf <treshold :
                        print(left_conf)
                        print(right_conf)
                        print(path2,file2,"no hand detected")
                        failed.write(path2+"/"+file2+"\n")
                        failed.flush()

                    if left_conf<treshold :
                        # right hand 
                        data = [path2,file2,"r"]
                        writer.writerow(data)
                    elif right_conf <treshold :
                        # left hand 
                        data = [path2,file2,"l"]
                        writer.writerow(data)
                    else:
                        if left_var[0]+left_var[1] > right_var[0]+right_var[1]:
                            # left hand 
                            data = [path2,file2,"l"]
                            writer.writerow(data)
                        else:
                            # right hand 
                            data = [path2,file2,"r"]
                            writer.writerow(data)

