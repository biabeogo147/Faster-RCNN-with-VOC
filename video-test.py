import cv2
import numpy
import shutil
import os.path
import numpy as np
import torch
import argparse
from pprint import pprint
from tqdm.autonotebook import tqdm
from torch.optim.optimizer import Optimizer
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.transforms import Compose, ToTensor, Resize, ColorJitter, Normalize
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

import voc_dataset
import classes


def get_args():
    parser = argparse.ArgumentParser(description='voc detection testing script')
    parser.add_argument('--data-path', '-d', type=str, default='.', help='path to dataset')
    parser.add_argument('--confident-threshold', '-t', type=float, default=0.2, help='path to dataset')
    parser.add_argument('--output', '-o', type=str, default='output', help='output image path')
    parser.add_argument('--model-path', '-m', type=str, default='model', help='model path')
    args = parser.parse_args()
    return args


def test(args):
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn_v2().to(device)

    pathModel = os.path.join(args.model_path, "best.pt")
    if (os.path.isfile(pathModel)):
        checkpoint = torch.load(pathModel, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    if (os.path.isdir(args.output) == False):
        os.mkdir(args.output)

    origin_video = cv2.VideoCapture(args.data_path)
    vid_writer = cv2.VideoWriter(os.path.join(args.output, os.path.basename(args.data_path)),
                                 cv2.VideoWriter_fourcc(*'MJPG'), int(origin_video.get(cv2.CAP_PROP_FPS)),
                                 (int(origin_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(origin_video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    model.eval()
    while (origin_video.isOpened()):
        ret, origin_image = origin_video.read()

        if (ret == False):
            break

        image = origin_image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        image = torch.from_numpy(image.transpose(2, 0, 1) / 255.0).unsqueeze(0).float()
        image = image.to(device)

        predictions = model(image)

        boxes = predictions[0]["boxes"]
        labels = predictions[0]["labels"]
        scores = predictions[0]["scores"]
        for box, label, score in zip(boxes, labels, scores):
            if score > args.confident_threshold:
                xmin, ymin, xmax, ymax = map(int, box)
                cv2.rectangle(origin_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(origin_image, classes.classes[label], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, cv2.LINE_AA)
        vid_writer.write(origin_image)
    vid_writer.release()
    origin_video.release()


if __name__ == "__main__":
    args = get_args()
    test(args)