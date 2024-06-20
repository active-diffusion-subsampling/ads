import os
import fnmatch
from utils.keras_utils import load_img_as_tensor
import numpy as np
from tqdm import tqdm


def find_png_files(directory):
    png_files = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, "*.png"):
            png_files.append(os.path.join(root, filename))
    return png_files


# Directory pointing to MNIST train set
MNIST_TRAIN_PATH = "/path/to/MNIST_png/train/"
OUTPUT_PATH = "benchmark/data_assets/mnist_train_set_variance.npy"
png_files = find_png_files(MNIST_TRAIN_PATH)

n = 0
mean_sum = np.zeros((1, 32, 32, 1))
mean_sq_sum = np.zeros((1, 32, 32, 1))

for train_img_path in tqdm(png_files):
    train_img = [
        load_img_as_tensor(
            str(train_img_path),
            image_shape=[32, 32],
            grayscale=True,
        )
    ]
    n += 1
    mean_sum += train_img

mean = mean_sum / n

for train_img_path in tqdm(png_files):
    train_img = [
        load_img_as_tensor(
            str(train_img_path),
            image_shape=[32, 32],
            grayscale=True,
        )
    ]

    mean_sq_sum += (train_img - mean) ** 2

variance = mean_sq_sum / n
np.save(OUTPUT_PATH, variance)
