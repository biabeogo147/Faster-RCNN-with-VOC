import os.path
import shutil

import numpy
import torch
import argparse
from pprint import pprint

from tqdm.autonotebook import tqdm
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import Compose, ToTensor, Resize, ColorJitter, Normalize, RandomAffine
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

import voc_dataset


def get_args():
    parser = argparse.ArgumentParser(description='voc detection training script')
    parser.add_argument('--log-dir', '-log', type=str, default='logs-tensorboard', help='log directory')
    parser.add_argument('--num-workers', '-nw', type=int, default=12, help='number of workers')
    parser.add_argument('--batch-size', '-b', type=int, default=4, help='input batch size')
    parser.add_argument('--data-path', '-d', type=str, default='data', help='path to dataset')
    parser.add_argument('--download', '-dl', type=bool, default=False, help='download dataset')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', '-l', type=float, default=0.001, help='learning rate')
    parser.add_argument('--model-path', '-m', type=str, default='model', help='model path')
    args = parser.parse_args()
    return args


def collate_fn(batch):
    # images = []
    # labels = []
    # for image, label in batch:
    #     images.append(image)
    #     labels.append(label)
    images, labels = zip(*batch)
    return images, list(labels)


def train(args):
    print(args)

    train_transform = Compose([
        RandomAffine(
            degrees=[-10, 10],
            translate=[0.1, 0.1],
            scale=[0.9, 1.1],
            shear=[-10, 10],
        ),
        ColorJitter(brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.1), #Data Augmentation
        ToTensor(), #Don't need normalize because the model has a normalization layer
    ])
    val_transform = Compose([
        ToTensor(),
    ])
    train_dataset = voc_dataset.VOCDataset(root=args.data_path, year="2012", image_set="train", download=args.download, transform=train_transform)
    val_dataset = voc_dataset.VOCDataset(root=args.data_path, year="2012", image_set="val", download=args.download, transform=val_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # pprint(train_dataset.classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                       trainable_backbone_layers=3)
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels, len(train_dataset.classes))
    # model.roi_heads.box_predictor.cls_score.out_features = len(train_dataset.classes)
    model.to(device)
    # pprint(model)

    start_epoch = 0
    best_mAP_score = -1
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pathModel = os.path.join(args.model_path, "last.pt")
    if (os.path.isfile(pathModel)):
        checkpoint = torch.load(pathModel, map_location=device)
        start_epoch = checkpoint["epoch"]
        best_mAP_score = checkpoint["best_mAP_score"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (os.path.isdir(args.log_dir) == False):
        os.mkdir(args.log_dir)
    # else:
    #     shutil.rmtree(args.log_dir)
    writer = SummaryWriter(args.log_dir)

    for e in range(start_epoch + 1, args.epochs):
        model.train()
        train_loss = []
        progress_bar = tqdm(train_loader, colour="green")
        for i, (images, targets) in enumerate(progress_bar):
            images = list(image.to(device) for image in images)
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

            lost_components = model(images, targets)
            loss = sum(loss for loss in lost_components.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            mean_loss = numpy.mean(train_loss)
            writer.add_scalar("Train/Loss", mean_loss, e * len(train_loader) + i)
            progress_bar.set_description("Epoch {} - Loss {:0.4f}".format(e, mean_loss))

        model.eval()
        metric = MeanAveragePrecision(iou_type="bbox")
        progress_bar = tqdm(val_loader, colour="green")
        with torch.no_grad():
            for images, targets in progress_bar:
                images = list(image.to(device) for image in images)
                targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

                predictions = model(images)
                metric.update(predictions, targets)
        mAP_score = metric.compute()
        writer.add_scalar("Val/map", mAP_score["map"], e)
        writer.add_scalar("Val/map 50", mAP_score["map_50"], e)
        writer.add_scalar("Val/map 70", mAP_score["map_75"], e)
        print("Epoch {} - Validation map: {:0.4f}".format(e, mAP_score["map"]))

        checkpoint = {
            "epoch": e,
            "best_mAP_score": best_mAP_score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if (mAP_score["map"] > best_mAP_score):
            best_mAP_score = mAP_score["map"]
            checkpoint["best_mAP_score"] = best_mAP_score
            torch.save(checkpoint, os.path.join(args.model_path, "best.pt"))
        torch.save(checkpoint, os.path.join(args.model_path, "last.pt"))


if __name__ == "__main__":
    args = get_args()
    train(args)