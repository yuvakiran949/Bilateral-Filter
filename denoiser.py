import cv2 as cv
import numpy as np
import time

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
            
            # weight = 0
            # new_pixel = 0
            # for i in range(k_size):
            #     for j in range(k_size):
            #         intensity_diff = (np.int64(part_img[extend, extend]) - np.int64(part_img[i, j]))
            #         n_sigma_s = gaussian_kernel[i, j]
            #         # print(f"sigma_s: {n_sigma_s}")
            #         n_sigma_b = np.exp(-0.5 * intensity_diff/ np.square(sigma_b))
            #         x = n_sigma_b * n_sigma_s
            #         weight+=x
            #         new_pixel = new_pixel + x*(part_img[i, j])
                    
            # # print(f"pixel_value: {new_pixel/weight}")
            # output_img[row-extend, column-extend] = np.uint8(new_pixel/weight)
    
    end_time = time.time()
    print(f"time taken: {int(end_time-start_t)}")
    return output_img

img = cv.imread("Q1/noisy1.jpg")
height, width = img.shape[:2]

img_denoised = convulution(21, img, 21, 9)

# img_convuluted = convulution(21, img, 20, 9)
# img_convuluted = convulution(5, img_convuluted, 20, 9)


# cv.imshow("jj", img_convuluted)

# img_compressed_part = img[0:171, 171:, :]
# height_p, width_p = img_compressed_part.shape[:2]
# img_compressed_part = cv.resize(img_compressed_part, (int(width_p/2), int(height_p/2)), cv.INTER_AREA)

# img_compressed = convulution(21, img_compressed_part, 20, 9)
# img_compressed = convulution(5, img_compressed, 20, 9)
# img_compressed = convulution(3, img_compressed, 20, 9)


# img_decompressed = cv.resize(img_compressed, (width_p, height_p), cv.INTER_LANCZOS4)
# cv.imshow("5-15", img_decompressed)

# img_part_1 = img[0:171, 0:171, :]
# img_convulution_1 = convulution(21, img_part_1, 20, 9)
# cv.imshow("part-1", img_convulution_1)

# img_compressed = convulution(5, img_compressed_part, 20, 9)
# img_compressed = convulution(19, img_compressed_part, 20, 9)

# img_decompressed = cv.resize(img_compressed, (width_p, height_p))
# cv.imshow("5-19", img_decompressed)

# img_compressed = convulution(5, img_compressed_part, 20, 9)
# img_compressed = convulution(21, img_compressed_part, 20, 9)

# img_decompressed = cv.resize(img_compressed, (width_p, height_p))
# cv.imshow("5-21", img_decompressed)



# extended_img = cv.resize(extended_img, (width, height), cv.INTER_CUBIC)

# cv.imshow("kol", img_decompressed)
# cv.imwrite("Q1/compress_denoised1.jpg", extended_img)
cv.waitKey(0)