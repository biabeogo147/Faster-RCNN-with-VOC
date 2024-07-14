import cv2
import torch
import os.path
import argparse
from pprint import pprint

import model_loading
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
    model = model_loading.loading_model(args)

    origin_image = cv2.imread(args.data_path)

    image = origin_image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (torch.from_numpy(image.transpose(2, 0, 1) / 255.0).unsqueeze(0).float()).to(device)

    model.eval()
    predictions = model(image)
    pprint(predictions)

    boxes = predictions[0]["boxes"]
    labels = predictions[0]["labels"]
    scores = predictions[0]["scores"]
    for box, label, score in zip(boxes, labels, scores):
        if score > args.confident_threshold:
            xmin, ymin, xmax, ymax = map(int, box)
            cv2.rectangle(origin_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(origin_image,
                        classes.classes[label],
                        (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    cv2.imwrite(os.path.join(args.output, os.path.basename(args.data_path)), origin_image)


if __name__ == "__main__":
    args = get_args()
    test(args)