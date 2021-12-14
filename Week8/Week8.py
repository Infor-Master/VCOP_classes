import matplotlib.pyplot as plt
import numpy as np
import cv2

def color_diagram():
    img_color = np.zeros([12, 18, 3])
    for iR in range(0, 256, 51):
        for iG in range(0, 256, 51):
            for iB in range(0, 256, 51):
                #inverter para obter valores altos primeiro
                R = 255-iR
                G = 255-iG
                B = 255-iB
                x = (iB / 51)
                y = (iG / 51)
                gx = (iR / 51) % 3
                gy = ((iR / 51) - ((iR / 51) % 3)) / 3

                posY = (gy*6) + y
                posX = (gx*6) + x
                img_color[int(posY), int(posX)] = [R, G, B]

    img_gray = np.zeros([1, 16, 3])
    for iK in range(0, 256, 17):
        posX = (iK / 16)
        img_gray[0, int(posX)] = [iK, iK, iK]

    fig, axs = plt.subplots(2)
    axs[0].imshow(img_color.astype(int))
    axs[1].imshow(img_gray.astype(int))
    plt.show()
    plt.close()
    return img_color.astype(int)

def color_model_HSI(img):
    y, x, c = img.shape

    H = np.zeros([y, x, 1])
    S = np.zeros([y, x, 1])
    I = np.zeros([y, x, 1])

    for xi in range(x):
        for yi in range(y):
            rgb = img[yi, xi]
            R = rgb[0]
            G = rgb[1]
            B = rgb[2]

            sum = R + B + G
            sr = np.sqrt(((R - G) * (R - G)) + ((R - B) * (G - B)))

            div = 0
            if sr != 0:
                div = ((R-G)+(R-B))/sr

            Ht = np.degrees(np.arccos(div/2))
            if G >= B:
                H[yi, xi] = Ht
            else:
                H[yi, xi] = (360 - Ht)

            S[yi, xi] = 0
            if sum != 0:
                S[yi, xi] = 1 - 3*(min(rgb)/(R+B+G))


            I[yi, xi] = (R+B+G)/3

    fig, axs = plt.subplots(2,2)
    axs[0, 0].imshow(img)
    axs[0, 1].imshow(H, cmap="gray", vmin=0, vmax=360)
    axs[1, 0].imshow(S.astype(float), cmap="gray", vmin=0, vmax=1)
    axs[1, 1].imshow(I.astype(int), cmap="gray", vmin=0, vmax=255)
    plt.show()
    plt.close()
    return H, S, I

def color_model_CMYK(img):
    y, x, c = img.shape

    C = np.zeros([y, x, 1])
    M = np.zeros([y, x, 1])
    Y = np.zeros([y, x, 1])
    K = np.zeros([y, x, 1])

    for xi in range(x):
        for yi in range(y):
            rgb = img[yi, xi]
            R = rgb[0]/255
            G = rgb[1]/255
            B = rgb[2]/255


            Kc = 1 - max([R, G, B])
            C[yi, xi] = ((1 - R - Kc) / (1 - Kc))
            M[yi, xi] = ((1 - G - Kc) / (1 - Kc))
            Y[yi, xi] = ((1 - B - Kc) / (1 - Kc))
            K[yi, xi] = Kc


    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(C.astype(float), cmap="gray", vmin=0, vmax=1)
    axs[0, 1].imshow(M.astype(float), cmap="gray", vmin=0, vmax=1)
    axs[1, 0].imshow(Y.astype(float), cmap="gray", vmin=0, vmax=1)
    axs[1, 1].imshow(K.astype(float), cmap="gray", vmin=0, vmax=1)
    plt.show()
    plt.close()
    return C, M, Y, K

def img_color_model(img, color):
    y, x, c = img.shape
    res = img

    for xi in range(x):
        for yi in range(y):
            rgb = img[yi, xi]
            R = (rgb[0] * color[0]) / 255
            G = (rgb[1] * color[1]) / 255
            B = (rgb[2] * color[2]) / 255
            res[yi, xi][0] = R
            res[yi, xi][1] = G
            res[yi, xi][2] = B
    plt.imshow(res)
    plt.show()
    plt.close()

    return res


# 14
img = color_diagram()
#
# 29

img = cv2.imread("rgb_diagram.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

H, S, I = color_model_HSI(img)

# 49

img = cv2.imread("color.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

C, M, Y, K = color_model_CMYK(img)


