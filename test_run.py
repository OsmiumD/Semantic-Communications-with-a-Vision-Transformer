# %%
import socket
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import numpy
import numpy as np
import tensorflow as tf
from io import StringIO
from PIL import Image
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
from skimage.metrics import peak_signal_noise_ratio
from utils.usrp_utils import compensate_signal, receive_constellation_tcp

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
    # print(image)
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
    print(tf.shape(data))

    i = data[:, :, 0].numpy().flatten()
    q = data[:, :, 1].numpy().flatten()
    i = np.round(np.clip(i / NORMALIZE_CONSTANT * 32767, -32767, 32767))
    q = np.round(np.clip(q / NORMALIZE_CONSTANT * 32767, -32767, 32767))
    constellations = np.concatenate((i, q))
    np.savez_compressed(f'{TEMP_DIRECTORY}/data.npz', data=data)
    np.savez_compressed(f'{TEMP_DIRECTORY}/constellations.npz', constellations=constellations)
    rcv_i = constellations[:512]
    rcv_q = constellations[512:]
    rcv_iq = np.c_[rcv_i, rcv_q] * NORMALIZE_CONSTANT / 32767
    rcv_iq = tf.cast(tf.convert_to_tensor(rcv_iq), tf.float32)
    rcv_iq = tf.reshape(rcv_iq, (1, -1, 2))
    proposed_result = decoder_network(rcv_iq)
    tf.keras.utils.save_img(f'{TEMP_DIRECTORY}/decoded_image.png', imBatchtoImage(proposed_result))
    cv2.imwrite(f'{TEMP_DIRECTORY}/decoded_image.png', imBatchtoImage(proposed_result).numpy() * 255)

    psnr = peak_signal_noise_ratio(tf_image.numpy(), proposed_result.numpy())
    print(f'semantic psnr: {psnr}')
    # print(tf_image.numpy() - proposed_result.numpy())

    img_png = Image.open('./image.png')
    img_png.save('./temp/img_jpeg.jpeg', quality=90)

    img_jpeg = cv2.imread('./temp/img_jpeg.jpeg')
    psnr_jpeg = peak_signal_noise_ratio(tf_image.numpy().reshape(32, 32, 3), img_jpeg / 255)
    print(f'jpeg psnr: {psnr_jpeg}')
