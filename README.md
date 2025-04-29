# 🌟 Faster-RCNN-with-VOC 🌟

This project implements **Faster R-CNN** for object detection using the **Pascal VOC dataset** 🎯. It includes scripts for training the model and testing it on both images 📸 and videos 🎥.

---

## 📂 Project Structure

- 🛠️ `.gitignore`: Specifies files and directories to be ignored by Git.
- 📋 `classes.py`: Defines the VOC dataset classes.
- 🖼️ `image_test.py`: Script for testing the model on images.
- ⚙️ `model_loading.py`: Manages loading and configuration of the Faster R-CNN model.
- 📖 `README.md`: You're reading it! Project documentation.
- 🚀 `train.py`: Script for training the Faster R-CNN model on the VOC dataset.
- 📹 `video_test.py`: Script for testing the model on videos.
- 🗃️ `voc_dataset.py`: Handles loading and preprocessing of the VOC dataset.

---

## 🏋️‍♂️ Training

The `train.py` script trains the Faster R-CNN model on the VOC dataset. Customize the training process with these arguments:

- 📊 `--log-dir/-log`: Directory for TensorBoard logs (default: `logs-tensorboard`).
- 👷 `--num-workers/-nw`: Number of data loading workers (default: `12`).
- 📦 `--batch-size/-b`: Batch size for training (default: `4`).
- 📁 `--data-path/-d`: Path to the VOC dataset (default: `data`).
- ⬇️ `--download/-dl`: Automatically download the dataset if not present (default: `False`).
- ⏳ `--epochs/-e`: Number of training epochs (default: `10`).
- 📈 `--lr/-l`: Learning rate (default: `0.001`).
- 💾 `--model-path/-m`: Path to save/load the model (default: `model`).

**Example Command**:
```bash
python train.py --data-path ./voc_data --epochs 20 --lr 0.0005 --batch-size 8
```

---

## 🖼️ Testing on Images

The `image_test.py` script runs inference on images using the trained model. Arguments include:

- 📁 `--data-path/-d`: Path to the test images (default: `.`).
- ✅ `--confident-threshold/-t`: Confidence threshold for detections (default: `0.2`).
- 📤 `--output/-o`: Directory to save output images (default: `output`).
- 💾 `--model-path/-m`: Path to the trained model (default: `model`).

**Example Command**:
```bash
python image_test.py --data-path ./test_images --confident-threshold 0.3 --output ./results
```

---

## 🎥 Testing on Videos

The `video_test.py` script runs inference on videos. It shares the same arguments as the image testing script:

- 📁 `--data-path/-d`: Path to the test video (default: `.`).
- ✅ `--confident-threshold/-t`: Confidence threshold for detections (default: `0.2`).
- 📤 `--output/-o`: Directory to save output video (default: `output`).
- 💾 `--model-path/-m`: Path to the trained model (default: `model`).

**Example Command**:
```bash
python video_test.py --data-path ./test_video.mp4 --confident-threshold 0.25 --output ./video_results
```

---

## 💡 Notes

- 📢 Ensure the VOC dataset is properly formatted and placed in the specified `--data-path` directory.
- 💾 The model is saved/loaded from the `--model-path` directory, so ensure it exists during training and testing.
- ⚖️ Adjust the `--confident-threshold` based on your use case to balance precision and recall during inference.

---

✨ **Happy Training and Testing!** ✨