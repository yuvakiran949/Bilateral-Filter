import cv2 as cv
import numpy as np
import time
import sys

def gkernel(l=3, sig=2):

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)


def convulution(k_size, image, sigma_s, sigma_b):


    rows, columns, dim = image.shape
    output_img = np.zeros(image.shape, dtype=np.uint8)
    start_t = time.time()
    # extending the image
    extend = int(k_size/2)
    # print(extend)
    # print(k_size)
    extended_img = np.zeros(shape=(rows + (2 * extend), columns + (2 * extend), dim), dtype=np.uint8)
    extended_img[extend:-extend, extend:-extend, :] = image
    rows, columns, dim = extended_img.shape
    #gaussian_kernel
    gaussian_kernel = gkernel(k_size, sigma_s)
    diff_data = {}

    # convolving
    for row in range(extend, rows-extend):
        for column in range(extend, columns-extend):
            part_img = extended_img[row-extend:row+extend+1, column-extend:column+extend+1]
            part_img = np.int64(part_img)
            diff = part_img - extended_img[row, column] 
            n_sigma_b = np.exp(-0.5 * np.square(diff)/ np.square(sigma_b))

            for n in range(3):
                x = gaussian_kernel * n_sigma_b[:, :, n]
                x = x/np.sum(x)
                new_pixel = np.uint8(np.sum(x*part_img[:, :, n]))
                output_img[row-extend, column-extend, n] = new_pixel
    end_time = time.time()
    print(f"time taken: {int(end_time-start_t)}")
    return output_img


path = sys.argv[1]
img = cv.imread(path)
height, width = img.shape[:2]

img_denoised = convulution(21, img, 21, 9)

cv.imwrite("denoised.jpg", img_denoised)
