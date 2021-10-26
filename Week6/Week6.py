import matplotlib.pyplot as plt
import numpy as np
import cv2

img_0333 = cv2.imread("./Fig0333_a__test_pattern_blurring_orig_.tif")
img_0333 = cv2.cvtColor(img_0333, cv2.COLOR_RGB2BGR)

img_0334 = cv2.imread("./Fig0334_a__hubble-original_.tif")
img_0334 = cv2.cvtColor(img_0334, cv2.COLOR_RGB2BGR)

img_0335 = cv2.imread("./Fig0335_a__ckt_board_saltpep_prob_pt05_.tif")
img_0335 = cv2.cvtColor(img_0335, cv2.COLOR_RGB2BGR)


plt.imshow(img_0333)
plt.show()
plt.imshow(img_0334)
plt.show()
plt.imshow(img_0335)
plt.show()

def smoothing_spacial_filter(image, m, n):
    kernel = np.ones((m, n), np.float32)/(n*m)
    smoothed_image = cv2.filter2D(image, -1, kernel)
    plt.imshow(smoothed_image)
    plt.show()
    return smoothed_image

def gray_level(image, Tmin, Tmax, Vtrue= None, Vfalse = None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    m,n = image.shape
    L = image.max()
    neg_image = L - image
    tresh_image = np.zeros((m,n), dtype = int)

    for i in range(m):
        for j in range(n):
            if Tmin < image[i,j] < Tmax:
                if Vtrue == None:
                    tresh_image[i,j] = image[i,j]
                else:
                    tresh_image[i, j] = Vtrue
            else:
                if Vfalse == None:
                    tresh_image[i, j] = image[i, j]
                else:
                    tresh_image[i, j] = Vfalse
    plt.imshow(tresh_image, cmap="gray")
    plt.show()
    return tresh_image

def OSF_median_filter(image, lvl):
    filtered_image = cv2.medianBlur(image, lvl)
    plt.imshow(filtered_image)
    plt.show()
    return filtered_image

smoothing_spacial_filter(img_0333, 10, 10)
smoothing_spacial_filter(img_0333, 20, 20)
smoothing_spacial_filter(img_0333, 5, 100)
smoothing_spacial_filter(img_0333, 60, 10)

smoothing_spacial_filter(img_0334, 20, 20)
gray_level(smoothing_spacial_filter(img_0334, 20, 20), 100, 180, 255, None)

OSF_median_filter(img_0335, 3)
