# Fashion Items Segmentation
## train_fashion.ipynb
It is a demo jupyter notebook that shows how to load, train and test the virtual fitting room program.
Most importantly, it shows how to train Mask R-CNN on Modanet dataset. Try to run with a GPU, and you would be able to get OKish result in a few minutes. It uses pre-trained weights to initialize.

## LoadandTest.ipynb
This note book demos how to load a pre-trained model and perform test.

A newer version is fashion_v2.0.ipynb
## fashion.py
It is a sample script to perform training and testing on PaperDoll dataset with ModaNet annotations using Mask-RCNN.
1. Following the setup Instructions in the project main page.
2. To train:

       python fashion.py

3. To test:

       python fashion.py --test
       
