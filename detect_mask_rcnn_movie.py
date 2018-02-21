import os
import sys
import random
import math
import numpy as np
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import coco
import utils
import model as modellib

import cv2

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "Mask_RCNN/", "mask_rcnn_coco.h5")
file_name = sys.argv[1]
output_path = file_name.rsplit('.', 1)[0]  + "_mask." +  file_name.rsplit('.', 1)[1]

"""
Mask_RCNNの設定
"""

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, i = 0):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [num_instances, height, width]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    """
    # Number of instances
    N = boxes.shape[0]
    #colors = random_colors(N)
    masked_image = image
    for i in range(N):
        if not np.any(boxes[i]):
            continue
        # Label and box
        #color = colors[i]
        color = (0, 0.4, 0.8)
        y1, x1, y2, x2 = boxes[i]
        if ((x2 - x1) > image.shape[1]/2) or ((y2 - y1) > image.shape[0]/2):
            continue
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        if label == "car" or label == "bus" or label == "truck":
            image = cv2.rectangle(masked_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            print("flame_number:{}, label:{}, center:({},{}), width:{}, height:{}, score:{:.3f}".format(nowFlame, label, (x1 + x2)/2, (y1 + y2)/2, x2 - x1, y2 - y1,score))
            caption = label + "{:.3f}".format(score)
            cv2.putText(masked_image, caption, ((x1 + x2) // 2, y1 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            # Mask
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)

    #cv2.imwrite("test_mask.png", image)
    return masked_image


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)


"""
ここからビデオの取り込み
"""

capture = cv2.VideoCapture(file_name)

flameFPS = capture.get(cv2.CAP_PROP_FPS)
flameNum = capture.get(cv2.CAP_PROP_FRAME_COUNT)
Width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
Height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
flameSpan = 1.0 / flameFPS

nowFlame = 0

if file_name.rsplit('.', 1)[1] == 'avi':
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    print('avi')

if file_name.rsplit('.', 1)[1] == 'mp4':
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    print('mp4')

VWriter = cv2.VideoWriter(output_path,fourcc, flameFPS, (Width, Height))
capture.set(cv2.CAP_PROP_POS_FRAMES, nowFlame)

while(capture.isOpened()):
    ret, frame = capture.read()
    if ret == False:
        nowFlame = nowFlame + 1
        continue

    image = frame
    results = model.detect([image], verbose=1)
    r = results[0]
    masked_image =display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'], i = nowFlame)
    #image = cv2.imread("test_mask.png")
    VWriter.write(masked_image)
    nowFlame += 1
    nowFlame += 1
    if nowFlame == flameNum :
        break
