import torch
import os.path
from pprint import pprint

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

import classes


def loading_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fasterrcnn_resnet50_fpn_v2()
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels, len(classes.classes))

    pathModel = os.path.join(args.model_path, "best.pt")
    if (os.path.isfile(pathModel)):
        checkpoint = torch.load(pathModel, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    if (os.path.isdir(args.output) == False):
        os.mkdir(args.output)

    return model


if __name__ == '__main__':
    model = loading_model()
