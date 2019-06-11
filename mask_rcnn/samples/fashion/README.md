# Fashion Items Segmentation
## train_fashion.ipynb
It is a demo jupyter notebook that shows how to load, train and test the virtual fitting room program.
Most importantly, it shows how to train Mask R-CNN on Modanet dataset. Try to run with a GPU, and you would be able to get OKish result in a few minutes. It uses pre-trained weights to initialize.

A newer version is fashion_v2.0.ipynb.

# Test Results
Test 100 Images Note book shows testing on 100 images with our latest trained weights.
It can achieve > 73% mAP with IoU set to 0.5.

## LoadandTest.ipynb
This note book demos how to load a pre-trained model and perform test.

## fashion.py
It is a sample script to perform training and testing on PaperDoll dataset with ModaNet annotations using Mask-RCNN.
1. Following the setup Instructions in the project main page.
2. To train:

       python fashion.py

3. To test:

       python fashion.py --test
       
