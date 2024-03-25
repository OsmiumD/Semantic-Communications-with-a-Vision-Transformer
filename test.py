import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    image = cv2.imread('./sc.jpg')
    print(image.shape)
    resize = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
    print(resize.shape)
    cv2.imwrite('image.png', resize)
