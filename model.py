import torchvision
import torch
from torch import nn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes):
    
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights = "DEFAULT" ,pretrained=True, pretrained_backbone=True, progress=True, trainable_backbone_layers=4)

    #setting pretrained parameters = False
    for param in model.parameters():
        param.requires_grad = False
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model
