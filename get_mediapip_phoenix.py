import cv2
import glob, os, sys
import mediapipe as mp
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# For static images:

DATA_PATH = "/home/negar/secondssd/pooya/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/"
OUT_PATH = "/home/negar/secondssd/pooya/mediapipe_res_phoenix/"

BG_COLOR = (192, 192, 192) # gray

BODY_POINTS = mp_holistic.PoseLandmark._member_names_
BODY_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.POSE_CONNECTIONS]

HAND_POINTS = mp_holistic.HandLandmark._member_names_
HAND_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.HAND_CONNECTIONS]

FACE_POINTS_NUM = lambda additional_points=0: additional_points + 468
FACE_POINTS = lambda additional_points=0: [str(i) for i in range(FACE_POINTS_NUM(additional_points))]
FACE_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.FACEMESH_TESSELATION]

def visualize_data(img,idx,pts,path,fname):
    plt.clf()
    plt.imshow(img)
    plt.scatter(pts[:, 0], pts[:, 1], color="red", s=10)

    os.makedirs(path, exist_ok=True)
    
    plt.savefig(path+fname)
    
    


def component_points(component, width: int, height: int, num: int):
    if component is not None:
        lm = component.landmark
        return np.array([[p.x * width, p.y * height, p.z] for p in lm]), np.ones(num)

    return np.zeros((num, 3)), np.zeros(num)


def body_points(component, width: int, height: int, num: int):
    if component is not None:
        lm = component.landmark
        return np.array([[p.x * width, p.y * height, p.z] for p in lm]), np.array([p.visibility for p in lm])

    return np.zeros((num, 3)), np.zeros(num)


with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    refine_face_landmarks=True) as holistic:

  datas = []
  confs = []
  
  for p1dir in os.listdir(DATA_PATH):
    for p2dir in os.listdir(DATA_PATH+ p1dir):

        IMAGE_FILES = sorted(glob.glob(DATA_PATH+ p1dir + "/" + p2dir +"/*.png"))
                
        for idx, file in enumerate(IMAGE_FILES):
            print(file)
            
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            h, w, _ = image.shape

            # Convert the BGR image to RGB before processing.
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            body_data, body_confidence = body_points(results.pose_landmarks, w, h, 33)
            face_data, face_confidence = component_points(results.face_landmarks, w, h,
                                                    FACE_POINTS_NUM(10))
            lh_data, lh_confidence = component_points(results.left_hand_landmarks, w, h, 21)
            rh_data, rh_confidence = component_points(results.right_hand_landmarks, w, h, 21)
            body_world_data, body_world_confidence = body_points(results.pose_world_landmarks, w, h, 33)
            
            data = np.concatenate([body_data, face_data, lh_data, rh_data, body_world_data])
            conf = np.concatenate([body_confidence, face_confidence, lh_confidence, rh_confidence, body_world_confidence])
        
            datas.append(data)
            confs.append(conf)
            
            numpy_res = np.concatenate((data,conf[:,None]),axis= 1)
            #save the results as .npy file
            
            import os
            path = OUT_PATH  + p1dir  + "/" + p2dir + "/" + file.split(p2dir)[1][1:-3] + "npy"
            os.makedirs(OUT_PATH + p1dir  + "/" + p2dir + "/" , exist_ok=True)
            np.save(path, numpy_res) 
            
            # visualize_data(image.copy(),str(idx),rh_data, "./vistest/"+ p1dir + "/" + p2dir +"/" , str(idx) +".png" ) 

            annotated_image = image.copy()
            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".

            # condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            # bg_image = np.zeros(image.shape, dtype=np.uint8)
            # bg_image[:] = BG_COLOR
            # annotated_image = np.where(condition, annotated_image, bg_image)

            # Draw pose, left and right hands, and face landmarks on the image.
            mp_drawing.draw_landmarks(
                annotated_image,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            
            mp_drawing.draw_landmarks(
                annotated_image,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(
                annotated_image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            mp_drawing.draw_landmarks(
                annotated_image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            os.makedirs( OUT_PATH + p1dir  + "/" + p2dir + "/visualize/", exist_ok=True)
            cv2.imwrite( OUT_PATH + p1dir  + "/" + p2dir + "/visualize/" + str(idx) + '.png', annotated_image)

