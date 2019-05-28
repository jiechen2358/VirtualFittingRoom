import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import io
import lmdb
import sqlite3
import pandas as pd
import json
import argparse
from PIL import Image
from IPython.display import display
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

parser = argparse.ArgumentParser(description='option for Fashion Segmentation')
parser.add_argument('-t', '--test', action='store_true', help='perform test')
args = parser.parse_args()

# Items used
# 1:bag|2:belt|3:boots|4:footwear|5:outer|6:dress|7:sunglasses|8:pants|9:top|10:shorts|11:skirt|12:headwear|13:scarf/tie
subset = ['bag', 'belt', 'outer', 'dress', 'pants', 'top', 'shorts', 'skirt', 'scarf/tie']


class TrainConfig(Config):
    """
    Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    NAME = "fashion"
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    # Number of classes (including background)
    NUM_CLASSES = 1 + 13 # background + 13 shapes
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class TestConfig(TrainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class PhotoData(object):
    def __init__(self, path):
        self.env = lmdb.open(
            path, map_size=2 ** 36, readonly=True, lock=False
        )

    def __iter__(self):
        with self.env.begin() as t:
            with t.cursor() as c:
                for key, value in c:
                    yield key, value

    def __getitem__(self, index):
        key = str(index).encode('ascii')
        with self.env.begin() as t:
            data = t.get(key)
        if not data:
            return None
        with io.BytesIO(data) as f:
            image = Image.open(f)
            image.load()
            return image

    def __len__(self):
        return self.env.stat()['entries']


class FashionDataset(utils.Dataset):

    def load_fashion(self, count=5, start=0, class_ids=None):
        if args.test:
            print('load data for testing.')
        else:
            print('load data for training.')
        
        if not class_ids:
            class_ids = sorted(coco.getCatIds())

        if class_ids:
            all_ids = []
            for id in class_ids:
                all_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            all_ids = list(set(all_ids))
        else:
            # All images
            all_ids = list(coco.imgs.keys())
        random.seed(2)
        random.shuffle(all_ids)

        # Add classes
        all_class_ids = sorted(coco.getCatIds())
        for i in all_class_ids:
            print('{}:{}'.format(i, coco.loadCats(i)[0]['name']))
            self.add_class("fashion", i, coco.loadCats(i)[0]['name'])

        image_ids = []
        print('number of images: ' + str(count))
        # Since we have 50K+ images and we only use several Thousand images
        # There will be no overlaps. If you plan to use 50k+, please redefine retrival logic.
        if args.test:
            image_ids = all_ids[-(count+start):]
        else:
            image_ids = all_ids[:count+start]

        # Add images
        for i in image_ids:
            self.add_image(
                "fashion", image_id=i,
                path=None,
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

    def load_image(self, image_id):
        imgId = self.image_info[image_id]['id']
        image = photo_data[imgId]
        try:
            out = np.array(image.getdata()).astype(np.int32).reshape((image.size[1], image.size[0], 3))
        except: 
            # This handles GrayScaleImage
            out = np.array(image.getdata()).astype(np.int32).reshape((image.size[1], image.size[0]))
            out = np.stack((out,)*3, axis=-1)
        return out

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        pass

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = annotation['category_id']
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(FashionDataset, self).load_mask(image_id)

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


def train(init_with='coco'): # imagenet, coco, or last
    '''
        Perform training.
    '''
    # Load and display random samples

    # image_ids = np.random.choice(dataset_train.image_ids, 1)
    # for image_id in image_ids:
    #     image = dataset_train.load_image(image_id)
    #     mask, class_ids = dataset_train.load_mask(image_id)
    #     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # Which weights to start with?
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    epoch_1 = 100
    epoch_2 = epoch_1 + 100

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epoch_1,
                layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also 
    # pass a regular expression to select which layers to
    # train by name pattern.
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=epoch_2,
                layers="all")
    print("Training completed.")

def test():
    '''
        Perform testing.
    '''
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=test_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()
    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    APs = []
    # Test on a random image
    image_ids = np.random.choice(dataset_val.image_ids, 10)
    # image_ids = dataset_val.image_ids
    for image_id in image_ids:
        print('=== Test on image: ' + str(image_id))
        print('=> Load Ground truth:')
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, test_config,
                                   image_id, use_mini_mask=False)
        log("original_image", original_image)
        log("image_meta", image_meta)
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)
        print('=> Result:')
        results = model.detect([original_image], verbose=1)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
        print('AP for image ' + str(image_id) + ': ', AP)
    print("mAP: ", np.mean(APs))
    print("Test Complete.")


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
# Load photo data from lmdb
photo_data = PhotoData(r'..' + os.path.sep + '..' + os.path.sep + '..' + os.path.sep + 'photos.lmdb')
print('Length of photo data from Paperdoll lmdb:', len(photo_data))

if __name__ == "__main__":
    json_file = r'..' + os.path.sep + '..' + os.path.sep + '..' + os.path.sep + 'modanet2018_instances_train.json'
    d = json.load(open(json_file))
    coco = COCO(json_file)
    subset_ids = sorted(coco.getCatIds(catNms=subset))
    config = TrainConfig()
    config.display()
    test_config = TestConfig()
    test_config.display()

    # Training dataset
    train_count = 20000
    dataset_train = FashionDataset()
    dataset_train.load_fashion(train_count, start=0, class_ids=subset_ids)
    dataset_train.prepare()

    # Validation dataset
    val_count = 50
    dataset_val = FashionDataset()
    dataset_val.load_fashion(val_count, start=0, class_ids=subset_ids)
    dataset_val.prepare()

    if args.test:
        print("Perform testing ...")
        test()
    else:
        print("Perform training ...")
        train()