# -*- coding: utf-8 -*-
# @Time : 20-6-9 上午10:20
# @Author : zhuying
# @Company : Minivision
# @File : anti_spoof_predict.py
# @Software : PyCharm

import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F


from .model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from .data_io import transform as trans
from .utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE
}


class Detection:
    def __init__(self):
        caffemodel = "./models/resources/detection_model/Widerface-RetinaFace.caffemodel"
        deploy = "./models/resources/detection_model/deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                   out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        return bbox

    def get_multiple_bboxes(self, img, max_faces=5):
        """Get multiple face bounding boxes (up to max_faces)"""
        height, width = img.shape[0], img.shape[1]
        original_height, original_width = height, width
        
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img_resized = cv2.resize(img,
                                   (int(192 * math.sqrt(aspect_ratio)),
                                    int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img

        blob = cv2.dnn.blobFromImage(img_resized, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        
        # Filter detections by confidence
        if len(out.shape) == 1:
            out = out.reshape(1, -1)
        
        valid_detections = out[out[:, 2] > self.detector_confidence]
        
        if len(valid_detections) == 0:
            return []
        
        # Sort by confidence (highest first)
        valid_detections = valid_detections[np.argsort(valid_detections[:, 2])[::-1]]
        
        # Limit to max_faces
        valid_detections = valid_detections[:max_faces]
        
        bboxes = []
        for detection in valid_detections:
            confidence = detection[2]
            left = int(detection[3] * original_width)
            top = int(detection[4] * original_height)
            right = int(detection[5] * original_width)
            bottom = int(detection[6] * original_height)
            
            # Ensure bbox is within image bounds
            left = max(0, left)
            top = max(0, top)
            right = min(original_width, right)
            bottom = min(original_height, bottom)
            
            bbox = [left, top, right - left, bottom - top]
            bboxes.append((bbox, confidence))
        
        return bboxes


class AntiSpoofPredict(Detection):
    def __init__(self, device_id):
        super(AntiSpoofPredict, self).__init__()
        self.device = torch.device("cuda:{}".format(device_id)
                                   if torch.cuda.is_available() else "cpu")

    def _load_model(self, model_path):
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input,)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        return None

    def predict(self, img, model_path):
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        self._load_model(model_path)
        self.model.eval()
        with torch.no_grad():
            result = self.model.forward(img)
            result = F.softmax(result).cpu().numpy()
        return result











