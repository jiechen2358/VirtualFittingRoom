# VirtualFittingRoom

To setup:
* download ModaNet annotations from https://github.com/eBay/modanet. Put modanet2018_instances_train.json and modanet2018_instances_val.json files under root(VirtualFittingRoom) directory
* ModaNet annotation is based on Paperdoll dataset: Please follow the instruction in https://github.com/kyamagu/paperdoll/tree/master/data/chictopia and download raw data (40GB).
* In maskrcnn folder download mask_rcnn_coco.h5 from: https://github.com/matterport/Mask_RCNN/releases/tag/v2.0

# A simple Demo to train ModaNet data on Mask-RCNN
See notebook: https://github.com/jiechen2358/VirtualFittingRoom/blob/master/mask_rcnn/samples/fashion/train_fashion.ipynb
