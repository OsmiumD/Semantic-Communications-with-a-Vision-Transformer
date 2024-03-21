# %%
import socket
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import numpy
import numpy as np
import tensorflow as tf
import math
import struct
import cv2

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.model import SemViT_Encoder_Only, SemViT_Decoder_Only
from utils.networking import receive_and_save_binary, send_binary
from utils.image import imBatchtoImage
from utils.usrp_utils import to_constellation_array
from config.train_config import BATCH_SIZE
from config.usrp_config import NORMALIZE_CONSTANT, TEMP_DIRECTORY
from usrp.pilot import PILOT_SIZE

ARCH = 'CCVVCC'
NUM_SYMBOLS = 512
CKPT_NAME = './ckpt/CCVVCC_512_10dB_599'
TARGET_JPEG_RATE = 2048

encoder_network = SemViT_Encoder_Only(
    ARCH,
    [256, 256, 256, 256, 256, 256],
    [1, 1, 3, 3, 1, 1],
    has_gdn=False,
    num_symbols=NUM_SYMBOLS,
)

decoder_network = SemViT_Decoder_Only(
    ARCH,
    [256, 256, 256, 256, 256, 256],
    [1, 1, 3, 3, 1, 1],
    has_gdn=False,
    num_symbols=NUM_SYMBOLS,
)

encoder_network.built = True
decoder_network.built = True
encoder_network.load_weights(CKPT_NAME).expect_partial()
decoder_network.load_weights(CKPT_NAME).expect_partial()

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

if __name__ == '__main__':
    image = cv2.imread('image.png')
    image: numpy.ndarray
    print(image)
    tf_image = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0
    h, w, c = tf_image.shape
    tf_image = tf.reshape(tf_image, (1, h, w, c))
    tf_image = tf.image.extract_patches(
        tf_image,
        sizes=[1, 32, 32, 1],
        strides=[1, 32, 32, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    tf_image = tf.reshape(tf_image, (-1, 32, 32, c))
    data = encoder_network(tf_image)
    i = data[:, :, 0].numpy().flatten()
    q = data[:, :, 1].numpy().flatten()
    i = np.round(np.clip(i / NORMALIZE_CONSTANT * 32767, -32767, 32767))
    q = np.round(np.clip(q / NORMALIZE_CONSTANT * 32767, -32767, 32767))
    constellations = to_constellation_array(i, q, i_pilot=True, q_pilot=True)
    np.savez_compressed(f'{TEMP_DIRECTORY}/constellations.npz', constellations=constellations)
