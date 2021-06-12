import os
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from src.factory import get_model


pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = '6d7f7b7ced093a8b3ef6399163da6ece'


def get_args():
    parser = argparse.ArgumentParser(description="This script evaluate age estimation model "
                                                 "using the APPA-REAL validation data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    weight_file = args.weight_file

    if not weight_file:
        weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=os.path.dirname(os.path.abspath(__file__)))

    # load model and weights
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
    model = get_model(cfg)
    model.load_weights(weight_file)

    dataset_root = Path(__file__).parent.joinpath("megaage_asian")
    gt_valid_path = dataset_root.joinpath("file_names.txt")
    image_paths = []
    with open(str(gt_valid_path)) as f:
      reader = f.read().strip().split('\n')
      for temp_path in reader:
        pre, post = temp_path.split('/')
        post = '_'.join(post.split('_')[1:])
        image_paths.append(pre + '/' + post)
      real_ages = [int(temp_path.split('/')[-1].split('_')[1]) for temp_path in reader]
    batch_size = 8

    faces = np.empty((batch_size, img_size, img_size, 3))
    ages = []
    for i, image_path in tqdm(enumerate(image_paths)):
        try:
          cv2.resize(cv2.imread(str(image_path), 1), (img_size, img_size))
          faces[i % batch_size] = cv2.resize(cv2.imread(str(image_path), 1), (img_size, img_size))
          if (i + 1) % batch_size == 0 or i == len(image_paths) - 1:
              results = model.predict(faces)
              ages_out = np.arange(0, 101).reshape(101, 1)
              predicted_ages = results[1].dot(ages_out).flatten()
              ages += list(predicted_ages)
              # len(ages) can be larger than len(image_names) due to the last batch, but it's ok.
        except Exception as e:
          continue

    real_abs_error = 0.0

    for i, real_age in enumerate(real_ages):
        real_abs_error += abs(ages[i] -real_age)
    print("MAE Real: {}".format(real_abs_error / len(real_ages)))

if __name__ == '__main__':
    main()
