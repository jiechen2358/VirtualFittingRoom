# VirtualFittingRoom
Note: This project is still in progress ...

## Introduction
Given a portrait picture, the goal of our project is to replace the texture and style of any piece of attire with new texture or style based on the user's choice. For instance, by feeding one portrait image with selections of target textures either predetermined or from another image, our application will change the textures or styles of certain fashion items respectively: pants texture will be transferred from canvas to jeans; coat texture will be transferred from cotton to leather.Thus we can adopt different combinations in different scenarios depending on the input image: day or night; indoor or outdoor with different body poses.

## Demo files to train and test (Fashion) ModaNet data using Mask-RCNN
See page at: https://github.com/jiechen2358/VirtualFittingRoom/blob/master/mask_rcnn/samples/fashion/, which including:
* Instructions on how to perform training and testing.
* A sample script to perform training and testing.
* A sample Jupyter notebook to demo how to train model.
* A sample Jupyter notebook to demo how to load pretrained model and perform test.
* Test 100 images.ipynb, test on 100 images and achieve > 73% mAP with IoU set to 0.5.

## Dataset
* The raw image  data comes from PaperDoll dataset - a collection of images of street-fashion models.
* Annotations are provideded by eBay's ModaNet - a dateset labels the subset of PaperDoll. The labels include bounding boxes, segmentations of 13 categories, including bag, belt, boots, top, shorts, scarf, tie etc. The labels are formmated in COCO style.

### Project and Data Setup:
* Download ModaNet annotations from https://github.com/eBay/modanet. Put modanet2018_instances_train.json and modanet2018_instances_val.json files under root(VirtualFittingRoom) directory.
* ModaNet annotation is based on Paperdoll dataset: Please follow the instruction in https://github.com/kyamagu/paperdoll/tree/master/data/chictopia and download raw data (40GB). Then extract the LMDB content and put file in the VirtualFittingRoom root directory:

      tar xf photos.lmdb.tar

* In maskrcnn folder download mask_rcnn_coco.h5 from: https://github.com/matterport/Mask_RCNN/releases/tag/v2.0. Put .h5 file in mask_rcnn directory.

## Prerequisites
Python 3, TensorFlow >= 1.3, Keras >= 2.0.8 and other packages listed in requirements.txt:

      numpy scipy Pillow cython matplotlib scikit-image tensorflow keras opencv-python h5py imgaug 

Install LMDB and pandas:

      pip install lmdb pandas

### pycocotools
Install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

For Linux, you can also simply try following commands:

      pip install cython pycocotools

## Reference
* For Instance Segmentation, our implementation based on https://github.com/matterport/Mask_RCNN.
* For Neural Style Transfer, our implementation based on CS231n Assignment3 Style Transfer Notebook. We modified it to some script so that we can use it more conveniently. See https://github.com/jiechen2358/VirtualFittingRoom/tree/master/styletransfer.
* For Neural Style Transfer, we also refer https://github.com/anishathalye/neural-style (Copyright (c) 2015-2018 Anish Athalye. Released under GPLv3). The reason is we found it includes some mechanism to maintain the color. One of our future work is to remain color information, either on content or on texture by introduce a new color loss.
