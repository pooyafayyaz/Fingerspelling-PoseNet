import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import pandas as pd
import math
import random
from random import randrange
import cv2

lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# lips = lipsUpperOuter + lipsUpperInner  +lipsLowerInner + lipsLowerOuter
lips = lipsUpperOuter + lipsLowerOuter

class HandPoseDataset(Dataset):

    def rotate(self, origin: tuple, point: tuple, angle: float):
        """
        Rotates a point counterclockwise by a given angle around a given origin.
        :param origin: Landmark in the (X, Y) format of the origin from which to count angle of rotation
        :param point: Landmark in the (X, Y) format to be rotated
        :param angle: Angle under which the point shall be rotated
        :return: New landmarks (coordinates)
        """

        ox, oy = origin
        px, py = point[0],point[1]

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

        return qx, qy

    def normalize(self, pose , mirror=False):
        #mirror if signing hands are different
        if mirror:
            # Negate the x-coordinates of the points array to mirror the points along the y-axis 
            pose[:, 0] = -pose[:, 0] + np.max(pose, axis=0)[0]
            # angle_deg += 90
        
        # rotate with respect to the elbow and wrist 
        # Convert the angle to radians
        # angle_rad = np.deg2rad(np.abs(180-angle_deg))

        # Create the 2D rotation matrix
        # rotation_matrix = np.array([
        #     [np.cos(angle_rad), -np.sin(angle_rad)],
        #     [np.sin(angle_rad), np.cos(angle_rad)],
        # ])

        # Multiply the points by the rotation matrix to rotate them
        # first[:,:] = np.dot(first[:,:], rotation_matrix)

        #set coordinate frame as the wrist
        pose[:,:] -= pose[0]  
        pose[:,:] -= np.min(pose, axis=0)
        
        #scale them to a box of 1x1
        max_vals = np.max(pose, axis=0)
        pose[:,:] /= max(max_vals)

        # Subtract the mean from each element and divide by the maximum absolute value
        # The values are then [-0.5,0.5] spread over zero 
        pose[:,:] = pose[:,:] - np.mean(pose[:,:])
        pose[:,:] = pose[:,:] / np.max(np.abs(pose[:,:]))
        pose[:,:] = pose[:,:] * 0.5

        return pose

    def normalize_face(self, pose):
        # print(pose)

        #set coordinate frame as the lip
        pose[:,:] -= pose[0]  
        pose[:,:] -= np.min(pose, axis=0)
        
        #scale them to a box of 1x1
        max_vals = np.max(pose, axis=0)
        pose[:,:] /= max(max_vals)

        # Subtract the mean from each element and divide by the maximum absolute value
        # The values are then [-0.5,0.5] spread over zero 
        pose[:,:] = pose[:,:] - np.mean(pose[:,:])
        pose[:,:] = pose[:,:] / np.max(np.abs(pose[:,:]))
        pose[:,:] = pose[:,:] * 0.5

        return pose


    def normalize_body(self, pose ):

        #set coordinate frame as the neck
        pose[:,:] -= ((pose[0]+pose[1])/2)  
        pose[:,:] -= np.min(pose, axis=0)
        
        #scale them to a box of 1x1
        max_vals = np.max(pose, axis=0)
        pose[:,:] /= max(max_vals)

        # Subtract the mean from each element and divide by the maximum absolute value
        # The values are then [-0.5,0.5] spread over zero 
        pose[:,:] = pose[:,:] - np.mean(pose[:,:])
        pose[:,:] = pose[:,:] / np.max(np.abs(pose[:,:]))
        pose[:,:] = pose[:,:] * 0.5

        return pose



    def __random_pass(self,prob):
        return random.random() < prob
    
    def augment_shear(self, hand_landmarks: dict, type: str, squeeze_ratio: tuple,type_sheer):
        """
        AUGMENTATION TECHNIQUE.
            - Squeeze. All the frames are squeezed from both horizontal sides. Two different random proportions up to 15% of
            the original frame's width for both left and right side are cut.
            - Perspective transformation. The joint coordinates are projected onto a new plane with a spatially defined
            center of projection, which simulates recording the sign video with a slight tilt. Each time, the right or left
            side, as well as the proportion by which both the width and height will be reduced, are chosen randomly. This
            proportion is selected from a uniform distribution on the [0; 1) interval. Subsequently, the new plane is
            delineated by reducing the width at the desired side and the respective vertical edge (height) at both of its
            adjacent corners.
        :param sign: Dictionary with sequential skeletal data of the signing person
        :param type: Type of shear augmentation to perform (either 'squeeze' or 'perspective')
        :param squeeze_ratio: Tuple containing the relative range from what the proportion of the original width will be
                            randomly chosen. These proportions will either be cut from both sides or used to construct the
                            new projection
        :return: Dictionary with augmented (by squeezing or perspective transformation) sequential skeletal data of the
                signing person
        """

        if type == "squeeze":
            move_left = random.uniform(*squeeze_ratio)
            move_right = random.uniform(*squeeze_ratio)

            src = np.array(((0, 1), (1, 1), (0, 0), (1, 0)), dtype=np.float32)
            dest = np.array(((0 + move_left, 1), (1 - move_right, 1), (0 + move_left, 0), (1 - move_right, 0)),
                            dtype=np.float32)
            mtx = cv2.getPerspectiveTransform(src, dest)

        elif type == "perspective":
            move_ratio = random.uniform(*squeeze_ratio)
            src = np.array(((0, 1), (1, 1), (0, 0), (1, 0)), dtype=np.float32)

            if type_sheer:
                dest = np.array(((0 + move_ratio, 1 - move_ratio), (1, 1), (0 + move_ratio, 0 + move_ratio), (1, 0)),
                                dtype=np.float32)
            else:
                dest = np.array(((0, 1), (1 - move_ratio, 1 - move_ratio), (0, 0), (1 - move_ratio, 0 + move_ratio)),
                                dtype=np.float32)

            mtx = cv2.getPerspectiveTransform(src, dest)


        augmented_landmarks = cv2.perspectiveTransform(np.array([hand_landmarks], dtype=np.float32), mtx)

        augmented_zero_landmark = cv2.perspectiveTransform(np.array([[[0, 0]]], dtype=np.float32), mtx)[0][0]
        augmented_landmarks = np.stack([np.where(sub == augmented_zero_landmark, [0, 0], sub) for sub in augmented_landmarks])

        hand_landmarks = np.array(augmented_landmarks[0])
        return hand_landmarks
        


    # def augment_arm_joint_rotate(hand_landmarks: dict, probability: float, angle_range: tuple) -> dict:
    #     """
    #     AUGMENTATION TECHNIQUE. The joint coordinates of both arms are passed successively, and the impending landmark is
    #     slightly rotated with respect to the current one. The chance of each joint to be rotated is 3:10 and the angle of
    #     alternation is a uniform random angle up to +-4 degrees. This simulates slight, negligible variances in each
    #     execution of a sign, which do not change its semantic meaning.
    #     :param sign: Dictionary with sequential skeletal data of the signing person
    #     :param probability: Probability of each joint to be rotated (float from the range [0, 1])
    #     :param angle_range: Tuple containing the angle range (minimal and maximal angle in degrees) to randomly choose the
    #                         angle by which the landmarks will be rotated from
    #     :return: Dictionary with augmented (by arm joint rotation) sequential skeletal data of the signing person
    #     """


    #     # Iterate over both directions (both hands)
    #     for side in ["left", "right"]:
    #         # Iterate gradually over the landmarks on arm
    #         for landmark_index, landmark_origin in enumerate(ARM_IDENTIFIERS_ORDER):
    #             landmark_origin = landmark_origin.replace("$side$", side)

    #             # End the process on the current hand if the landmark is not present
    #             if landmark_origin not in body_landmarks:
    #                 break

    #             # Perform rotation by provided probability
    #             if __random_pass(probability):
    #                 angle = math.radians(random.uniform(*angle_range))

    #                 for to_be_rotated in ARM_IDENTIFIERS_ORDER[landmark_index + 1:]:
    #                     to_be_rotated = to_be_rotated.replace("$side$", side)

    #                     # Skip if the landmark is not present
    #                     if to_be_rotated not in body_landmarks:
    #                         continue

    #                     body_landmarks[to_be_rotated] = [__rotate(body_landmarks[landmark_origin][frame_index], frame,
    #                         angle) for frame_index, frame in enumerate(body_landmarks[to_be_rotated])]

    #     return __wrap_sign_into_row(body_landmarks, hand_landmarks)


    def augment_data(self, data,selected_aug,angle=None,type_sheer= None):
        # if selected_aug == 0:
        data =  np.array([self.rotate((0.5, 0.5), frame, angle) for frame in data])

        # if selected_aug == 1:
        #     data = self.augment_shear(data, "perspective", (0, 0.1),type_sheer)

        # if selected_aug == 2:
        #     data = self.augment_shear(data, "squeeze", (0, 0.15),type_sheer)

        # if selected_aug == 3:
        #     data = self.augment_arm_joint_rotate(data, 0.3, (-4, 4))
        return data 

    def readHandPose(self, file ):
        left_joints = []
        right_joints = []
        face_joints = []
        body_joints = []

        df = self.sign_hand_csv[self.sign_hand_csv['B'].str.contains(file.split('/')[-2],regex=False)]
        aug = False
        if self.augmentations and random.random() < self.augmentations_prob:
            aug = True
            angle = math.radians(random.uniform(-13, 13))
            selected_aug = randrange(3)
            type_sheer = self.__random_pass(0.5)

        for ii, frame in enumerate(sorted(glob.glob(file+"/*.npy"))):
            frame_np = np.load(frame,allow_pickle=True)

            if frame_np is None or len(frame_np) ==0:
                continue

            right_hand = frame_np[532:553][:,:2]
            left_hand = frame_np[511:532][:,:2]
            body = frame_np[11:17] 
            face = frame_np[lips] 

            if (df['C']== 'l').iloc[0]:
                if (left_hand[:,:2].sum()==0):
                    if ii != 0:
                        left_hand[:] = left_joints[-1] 
                    else:
                        left_hand[:] = np.zeros((21,2))
                else:                     
                    if aug:
                        left_hand = self.augment_data(left_hand,selected_aug,angle,type_sheer)                        

                    left_hand[:,:2] = self.normalize(left_hand[:,:2], mirror = True) 
            else:
                if (right_hand[:,:2].sum()==0):
                    if ii != 0:
                        right_hand[:] = right_joints[-1]
                    else:
                        right_hand[:] = np.zeros((21,2))
                else:
                    if aug:
                        right_hand = self.augment_data(right_hand,selected_aug,angle,type_sheer)   
                    right_hand[:,:2] = self.normalize(right_hand[:,:2], mirror = False) 
            
            if (face[:,:2].sum()==0):
                if ii != 0:
                    face[:] = face[-1]
                else:
                    # face[:] = np.zeros((43,4))
                    face[:] = np.zeros((21,4))
            else:
                face[:,:2] = self.normalize_face(face[:,:2])

            if (body[:,:2].sum()==0):
                if ii != 0:
                    body[:] = body[-1]
                else:
                    body[:] = np.zeros((6,4))
            else:
                body[:,:2] = self.normalize_body(body[:,:2])

            left_joints.append(left_hand)
            right_joints.append(right_hand)

            face_joints.append(face[:,:2])
            body_joints.append(body[:,:2])

        for fjoint_idx in range(len(face_joints)-2,-1,-1):
            if (face_joints[fjoint_idx].sum()==0):
                face_joints[fjoint_idx] = face_joints[fjoint_idx+1].copy()

        for bjoint_idx in range(len(body_joints)-2,-1,-1):
            if (body_joints[bjoint_idx].sum()==0):
                body_joints[bjoint_idx] = body_joints[bjoint_idx+1].copy()

        if (df['C']== 'l').iloc[0]:
            for ljoint_idx in range(len(left_joints)-2,-1,-1):
                if (left_joints[ljoint_idx].sum()==0):
                    left_joints[ljoint_idx] = left_joints[ljoint_idx+1].copy()
        else:
            for rjoint_idx in range(len(right_joints)-2,-1,-1):
                if (right_joints[rjoint_idx].sum()==0):
                    right_joints[rjoint_idx] = right_joints[rjoint_idx+1].copy()
        
        # print(np.array(left_joints).shape)
        # print(np.array(body_joints).shape)
        # print(np.concatenate((left_joints, body_joints, face_joints),1).shape)
        if self.additional_joints != None:
            left_joints = np.concatenate((left_joints, face_joints),1)
            right_joints = np.concatenate((right_joints, face_joints),1)
        else: 
            left_joints = np.array(left_joints)
            right_joints = np.array(right_joints)

        return right_joints[:,:,:2] if (df['C']== 'r').iloc[0] else left_joints[:,:,:2]

    def __init__(self, data_dir, label_csv , hand_detected_label, target_enc_df , subset, transform = None, augmentations = True,augmentations_prob=0.5, additional_joints= None ):
        self.data_dir = data_dir

        self.files = []
        self.labels = []
        self.transform = transform
        self.augmentations = augmentations
        self.augmentations_prob = augmentations_prob
        self.additional_joints = additional_joints
        
        self.sign_hand_csv = pd.read_csv(hand_detected_label,names=('A', 'B', 'C'))
        
        self.all_data = pd.read_csv(label_csv)
        self.all_data = self.all_data[self.all_data['filename'].notna()]
        self.all_data = self.all_data[self.all_data['label_proc'].notna()]
        
        for p1dir in os.listdir(data_dir):
            for p2dir in os.listdir(data_dir+ p1dir):
                # print("#####")
                # print(p1dir + "/" + p2dir)
                # print(self.all_data['filename'].str.contains(p1dir + "/" + p2dir, regex=False).any())
                
                df = self.all_data[self.all_data['filename'].str.contains(p1dir + "/" + p2dir, regex=False)]
                if len(df) == 0 :
                    continue
                if df["partition"].iloc[0] == subset:
                    self.files.append(data_dir+ p1dir + "/" + p2dir +"/")
                    self.labels.append(target_enc_df[ target_enc_df["names"].str.contains(p1dir + "/" + p2dir, regex=False) ]["enc"].iloc[0])

    def __len__(self):
        return len(self.files)
    
    def get_file_path(self, idx):
        return self.files[idx]
        
    def __getitem__(self, idx):
        
        file_path = os.path.join(self.data_dir, self.files[idx])
        hand_pose = self.readHandPose(file_path)
        hand_pose = torch.from_numpy(hand_pose).float()
        if self.transform:
            hand_pose = self.transform(hand_pose)

        label = self.labels[idx]
        return  hand_pose , torch.as_tensor(label)
