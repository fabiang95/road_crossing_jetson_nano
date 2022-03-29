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

import os

import pyrealsense2 as rs

from collections import deque
from termcolor import cprint
from pprint import pprint

import multiprocessing as mp

from gtts import gTTS
from playsound import playsound

# setup detectron2 for zebra and tactile =======================================
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
    MetadataCatalog.get("my_dataset").thing_classes = ['zebra crossing', 'tactile', 'pedestrian_traffic_light', 'vehicle_traffic_light']
    predictor = DefaultPredictor(cfg)
    return predictor

def get_default_predictor():
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda"
    predictor = DefaultPredictor(cfg)
    return predictor

def get_zebra_predictor():
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    print(os.getcwd())
    weights_path = os.path.join(os.getcwd(),"model_final_new.pth")

    # cfg.MODEL.WEIGHTS = os.path.join(os.getcwd, "zebra_tactile.pth")
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

    ####add MetadataCatalog
    MetadataCatalog.get("my_dataset").thing_classes = ['zebra crossing', 'tactile', 'pedestrian_traffic_light', 'vehicle_traffic_light']
    predictor = DefaultPredictor(cfg)
    return predictor

predictor = get_default_predictor()

# utility FUNCTIONS ============================================================

def get_depth_intrinsics():
    pipeline = rs.pipeline()
    cfg = pipeline.start() # Start pipeline and get the configuration it found
    profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for deptha stream
    intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
    pipeline.stop()
    return intr
intr = get_depth_intrinsics()

def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics):
    print(depth_image)
    print(type(depth_image))
    [height, width] = depth_image.shape

    nx = np.linspace(0, width-1, width)
    ny = np.linspace(0, height-1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - camera_intrinsics.ppx)/camera_intrinsics.fx
    y = (v.flatten() - camera_intrinsics.ppy)/camera_intrinsics.fy

    z = depth_image.flatten() / 1000;
    x = np.multiply(x,z)
    y = np.multiply(y,z)

    x = x[np.nonzero(z)]
    y = y[np.nonzero(z)]
    z = z[np.nonzero(z)]

    # # to append human and instance index to x y z coordinate
    # human = bool
    # instance = index

# 	"""
# 	Convert the depthmap to a 3D point cloud
# 	Parameters:
# 	-----------
# 	depth_frame 	 	 : rs.frame()
# 						   The depth_frame containing the depth map
# 	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
# 	Return:
# 	----------
# 	x : array
# 		The x values of the pointcloud in meters
# 	y : array
# 		The y values of the pointcloud in meters
# 	z : array
# 		The z values of the pointcloud in meters
# 	"""
    return x, y, z

def get_indices_detection(outputs, dect_class=0):
    #function to get indices of instance with that has same class as dect_class
    class_list =  outputs['instances'].pred_classes.tolist()
    indices = list()
    for i in range(len(class_list)):
        if class_list[i] == dect_class:
           indices.append(i)
    print(indices, "indices")
    return indices

def get_masks(outputs,indices):
    #to output mask as a numpy array
    mask_img = outputs["instances"][0].pred_masks.cpu().numpy()
    return mask_img[0]

def retain_true(rgbd,mask):
    #to avoid manipulating original masks
    #everything outside of mask will have value of zero, only mask area will have values (color)
    rgbd_copy=np.copy(rgbd)
    rgbd_copy[mask == False] = 0
    return rgbd_copy

def get_instance_depth_img(depth_img,mask):
    inst_depth_img = retain_true(depth_img,mask)
    return

def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics):
    print(depth_image)
    print(type(depth_image))
    [height, width] = depth_image.shape

    nx = np.linspace(0, width-1, width)
    ny = np.linspace(0, height-1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - camera_intrinsics.ppx)/camera_intrinsics.fx
    y = (v.flatten() - camera_intrinsics.ppy)/camera_intrinsics.fy

    z = depth_image.flatten() / 1000;
    x = np.multiply(x,z)
    y = np.multiply(y,z)

    x = x[np.nonzero(z)]
    y = y[np.nonzero(z)]
    z = z[np.nonzero(z)]

    # # to append human and instance index to x y z coordinate
    # human = bool
    # instance = index


# 	"""
# 	Convert the depthmap to a 3D point cloud
# 	Parameters:
# 	-----------
# 	depth_frame 	 	 : rs.frame()
# 						   The depth_frame containing the depth map
# 	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
# 	Return:
# 	----------
# 	x : array
# 		The x values of the pointcloud in meters
# 	y : array
# 		The y values of the pointcloud in meters
# 	z : array
# 		The z values of the pointcloud in meters
# 	"""
    return x, y, z

def text_to_speech(distance,theta):
    txt = f"distance {distance} meters, angle {theta} degrees"
    txt = str(txt)
    tts = gTTS(txt, tld="com",slow=True)
    tts.save("direction.mp3")
    print("saved: "+txt)
    time.sleep(0.2)
    playsound("direction.mp3")
    time.sleep(0.1)
    os.remove("direction.mp3")

def speech_xzy(x,y,z):
    txt = f"x coordinate {x} meters,y coordinate {y} meters,z coordinate {z} meters"
    txt = str(txt)
    tts = gTTS(txt, tld="com")
    tts.save("direction.mp3")
    print("saved: "+txt)
    time.sleep(0.2)
    playsound("direction.mp3")
    time.sleep(0.1)
    os.remove("direction.mp3")

# workers ======================================================================

def realsense_producers(shared, end_event):

    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(
        rs.option.visual_preset, 3
    )  # Set high accuracy for depth sensor
    depth_scale = depth_sensor.get_depth_scale()

    clipping_distance_in_meters = 1
    clipping_distance = clipping_distance_in_meters / depth_scale

    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while not end_event.is_set():
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                raise RuntimeError("Could not acquire depth or color frames.")

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # print(depth_image.shape,color_image.shape)
            # if depth_image.shape != color_image[0].shape:
            #     print("depth and color of different dimensions")
            shared["realsense"] = {'rgb':color_image,'depth':depth_image}

            # shared["realsense"]["depth"] = depth_image
            # shared["realsense"]["rgb"] = color_image
            # adds color and depth image to a shared dictionary

    except Exception as e:
        print(e)

    finally:
        pipeline.stop()

def color_consumer(shared, end_event):

    while not end_event.is_set():
        # depth = shared["realsense"].pop('depth',None)
        if shared["realsense"]:
            print("realsense dictionary not empty")
            color_img = shared["realsense"].pop('rgb',None)
            depth_img = shared["realsense"].pop('depth',None)
            outputs = predictor(color_img)
            zebra_indices = get_indices_detection(outputs,0)
            if zebra_indices:
                for i in zebra_indices:
                    masks = get_masks(outputs,i)
                    inst_depth = retain_true(depth_img,masks)
                    x,y,z = convert_depth_frame_to_pointcloud(depth_img, intr)
                    avg_x,avg_y,avg_z = round(x.mean(), 2),round(y.mean(), 2),round(z.mean(), 2)
                    speech_xzy(avg_x,avg_y,avg_z)
            else:
                print("no zebra crossings")

        else:
            print("empty dict")
            time.sleep(1)
        # if depth.isArray():
        #     print(depth)
        #
        # elif not depth:
        #     time.sleep(1)
        #     print("shared dict was empty")

    # while not end_event.is_set():
    #     color_img = cv2.imread(shared["realsense"]["rgb"])
    #     depth = shared["realsense"]["depth"]
    #     print(depth.shape)
    #     cv2.imshow('color',color_img)
    #     k = cv2.waitKey(1)
    #     if k == 27:
    #         end_event.set()
    #     time.sleep(1)
    # cv2.destroyAllWindows()

# MAIN =========================================================================
if __name__ == "__main__":
    end_event = mp.Event()

    manager = mp.Manager()
    shared = manager.dict()
    shared['realsense'] = manager.dict()

    producer_process = mp.Process(target=realsense_producers,
                                args=(shared, end_event))

    consumer_process = mp.Process(target=color_consumer,
                                args=(shared, end_event))

    try:
        producer_process.start()
        consumer_process.start()

        producer_process.join()
        consumer_process.join()
    except KeyboardInterrupt:
        print("CTRL+C")
    except Exception as e:
        print(e)
    finally:
        end_event.set()
        producer_process.join()
        consumer_process.join()

        manager.shutdown()

        print("Bye hope you crossed the road safely!")
