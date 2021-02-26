import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw

from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials


class AzureAPI:
    def __init__(
    self,
    images_dir,
    output_dir,
    key,
    endpoint
    ):
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.face_client = FaceClient(endpoint, CognitiveServicesCredentials(key))

    def infer_face_features(self):
        face_attributes = ['age', 'gender', 'headPose', 'smile', 'hair', 'facialHair',  'glasses']

        images = [f for f in os.listdir(self.images_dir) if f[0] not in '._']

        attrs = np.zeros([len(images),  8,  1])
        # Detect a face with attributes, returns a list[DetectedFace]
        imageCounter = 0
        for image in images:
            face_fd = open(os.path.join(self.images_dir, image), "rb")
            detected_faces = self.face_client.face.detect_with_stream(face_fd, return_face_attributes=face_attributes)
            if not detected_faces:
                print('No face detected from image: ',  format(os.path.basename(image)))
                faceAttr = np.expand_dims(np.array([-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]),  axis = 1)
                attrs[imageCounter] = faceAttr
                continue

            for face in detected_faces:
                gender = 0
                print(str(face.face_attributes.gender))
                if str(face.face_attributes.gender)== 'Gender.male':
                    gender = 1

                glasses = 1
                if str(face.face_attributes.glasses)== 'GlassesType.no_glasses':
                    glasses = 0

                yaw = face.face_attributes.head_pose.yaw
                pitch = face.face_attributes.head_pose.pitch
                bald = face.face_attributes.hair.bald
                beard =  face.face_attributes.facial_hair.beard
                age = face.face_attributes.age
                expression = face.face_attributes.smile
                faceAttr = np.expand_dims(np.array([gender,  glasses,  yaw,  pitch,  bald,  beard,  age,  expression]),  axis = 1)
                attrs[imageCounter] = faceAttr
                print('Facial attributes detected:')
                print('Gender: ', face.face_attributes.gender)
                print('Glasses: ', face.face_attributes.glasses)
                print('Head pose yaw: ', face.face_attributes.head_pose.yaw)
                print('Head pose pitch: ', face.face_attributes.head_pose.pitch)
                print('baldness: ', face.face_attributes.hair.bald)
                print('beard:',  face.face_attributes.facial_hair.beard)
                print('Age: ', face.face_attributes.age)
                print('Smile: ', face.face_attributes.smile)

        np.save(self.output_dir + '/attributes',  attrs)

