import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import requests
import time
from io import BytesIO
from PIL import Image

import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.catalog import Metadata

import os

def get_predictor():
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    print(os.getcwd())
    weights_path = os.path.join(os.getcwd(),"model_final.pth")

    # cfg.MODEL.WEIGHTS = os.path.join(os.getcwd, "zebra_tactile.pth")
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

    ####add MetadataCatalog
    # MetadataCatalog.get("my_dataset").thing_classes = ['zebra crossing', 'tactile', 'pedestrian_traffic_light', 'vehicle_traffic_light']
    my_metadata = Metadata()
    my_metadata.set(thing_classes = ['zebra crossing', 'tactile', 'pedestrian_traffic_light', 'vehicle_traffic_light'])
    predictor = DefaultPredictor(cfg)
    return predictor,my_metadata

# classes = ['zebra crossing', 'tactile', 'pedestrian_traffic_light', 'vehicle_traffic_light']
# data_path = '/data/RGB_labelled/'
# for d in ["train", "test"]:
#     DatasetCatalog.register(
#         "category_" + d,
#         lambda d=d: get_data_dicts(data_path+d, classes)
#     )
#     MetadataCatalog.get("category_" + d).set(thing_classes=classes)
#
# microcontroller_metadata = MetadataCatalog.get("category_train")

# from detectron2.modeling import build_model
# model = build_model(cfg)
#
# weights_path = os.path.join(os.getcwd(),"zebra_tactile.pth")
# # cfg.MODEL.WEIGHTS = os.path.join(os.getcwd, "zebra_tactile.pth")
#
# from detectron2.checkpoint import DetectionCheckpointer
# DetectionCheckpointer(model).load(weights_path)

# predictor = DefaultPredictor(cfg)


def get_faster_predictor():
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    print(os.getcwd())
    weights_path = os.path.join(os.getcwd(),"model_final_faster_rcnn.pth")

    # cfg.MODEL.WEIGHTS = os.path.join(os.getcwd, "zebra_tactile.pth")
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

    ####add MetadataCatalog
    my_metadata = Metadata()
    my_metadata.set(thing_classes = ['zebra crossing', 'tactile', 'pedestrian_traffic_light', 'vehicle_traffic_light'])
    # MetadataCatalog.get("my_dataset").thing_classes = ['zebra crossing', 'tactile', 'pedestrian_traffic_light', 'vehicle_traffic_light']
    predictor = DefaultPredictor(cfg)
    return predictor , my_metadata


predictor, my_metadata = get_predictor()


os.getcwd()
img_path = os.path.join(os.getcwd(),"data","RGB","26Jan")
img_name = os.path.join(img_path ,"1_zebrametre(s).png")
print(img_name)

# img_name ="C:\\\Users\\fabia\\Desktop\\road_crossing\\road-crossing\\data\\RGB\\7Feb\\12_zebra_7.png"
img_name = r"C:\Users\fabia\Desktop\road_crossing\road-crossing\data\RGB\26Jan\1_zebrametre(s).png"


img = cv2.imread(img_name)
print(img)
# cv2.imshow(img,"img")
# microcontroller_metadata = MetadataCatalog.get("category_train")
def test_img(img):
    outputs = predictor(img)
    # v = Visualizer(img[:, :, ::-1],
    #               metadata=microcontroller_metadata,
    #               scale=0.8,
    #               instance_mode=ColorMode.IMAGE # removes the colors of unsegmented pixels
    # )
    v = Visualizer(img[:, :, ::-1],  metadata=my_metadata,scale=0.8)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize = (14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()

    class_list =  outputs['instances'].pred_classes.tolist()
    indices = list()

    for i in range(len(class_list)):
        if class_list[i] == 0:
           indices.append(i)
    print(indices, "indices")

    mask_img = outputs["instances"][0].pred_masks.cpu().numpy()
    mask_img.shape
test_img(img)
# outputs = predictor(img)
# v = Visualizer(img[:, :, ::-1],
#               metadata=my_metadata,
#               scale=0.8,
#               instance_mode=ColorMode.IMAGE # removes the colors of unsegmented pixels
# )
# # v = Visualizer(img[:, :, ::-1], metadata=my_metadata, scale=0.8)
# v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# plt.figure(figsize = (14, 10))
# plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
# plt.show()
def get_indices_detection(outputs, dect_class=0):
    class_list =  outputs['instances'].pred_classes.tolist()
    indices = list()

    for i in range(len(class_list)):
        if class_list[i] == dect_class:
           indices.append(i)
    print(indices, "indices")
    return indices

def get_masks(outputs,indices):
    mask_img = outputs["instances"][0].pred_masks.cpu().numpy()
    print(mask_img[0])
    plt.imshow(mask_img[0])
    return mask_img

# indices = get_indices_detection(outputs)
# get_masks(outputs,0)
