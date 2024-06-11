import classes

import torch
from pprint import pprint
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

class VOCDataset(VOCDetection):
    def __init__(self, root, year, image_set, download, transform):
        super(VOCDataset, self).__init__(root, year, image_set, download, transform)
        self.classes = classes.classes


    def __getitem__(self, index):
        image, data = super(VOCDataset, self).__getitem__(index)
        bboxes, labels = [], []
        for obj in data["annotation"]["object"]:
            bbox = obj["bndbox"]
            labels.append(self.classes.index(obj["name"]))
            bboxes.append([int(bbox["xmin"]), int(bbox["ymin"]), int(bbox["xmax"]), int(bbox["ymax"])])
        target = {
            "boxes": torch.FloatTensor(bboxes),
            "labels": torch.LongTensor(labels),
        }
        return image, target


if __name__ == "__main__":
    transform = Compose([
        ToTensor(),
    ])
    train_dataset = VOCDataset(root="data", year="2012", image_set="train", download=False, transform=transform)
    val_dataset = VOCDataset(root="data", year="2012", image_set="val", download=False, transform=transform)

    image, target = train_dataset[0]
    pprint(target)