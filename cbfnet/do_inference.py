import json
import numpy as np
from nilearn.image import load_img, new_img_like
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import argparse

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help='Path to model')
parser.add_argument("-w", "--weights", help='Path to weights')
parser.add_argument("-i", "--input", help='Input nifti image with PLDs appended')

args = parser.parse_args()

with open(args.model) as json_file:
    json_config = json_file.read()
model = model_from_json(json_config)
model.load_weights(args.weights)

in_file = args.input

# This is a scaling calculated from the initial training dataset
img = load_img(in_file).get_fdata() / 96.

output = model.predict(img[np.newaxis,...])

out_nii = new_img_like(in_file, output[0])

# This is unscaled CBF and ATT!
out_nii.to_filename(args.input + "CBF-ATT")
