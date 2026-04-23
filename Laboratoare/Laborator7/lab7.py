
import numpy as np
from skimage.exposure import histogram
from sklearn.svm import SVC
from scipy.spatial.distance import cdist

def feature_local(imagine, d = 3):
    pad = d // 2
    pad_img = np.pad(imagine, pad, mode = 'constant', constant_values=0)
    h, w = imagine.shape
    vectori = []

    for i in range(h):
        for j in range(w):
            val_centrala = imagine[i, j]
            vecini = pad_img[i : i + d, j : j + d]

            mat_binara = (vecini >= val_centrala).astype(int)
            vec_binar = tuple(mat_binara.flatten())
            vectori.append(vec_binar)

    vec_set = list(set(vectori))
    index_vec = {vec: i for i, vec in enumerate(vec_set)}

    histograma = np.zeros(len(vec_set))
    for vec in vectori:
        histograma[index_vec[vec]] +=1

    return histograma, vec_set

def regiuni_gradient(imagine, k = 5):
    img_float = imagine.astype(np.float64)

    gx = np.zeros_like(img_float)
    gx[:, :-1] = img_float[:, 1:] - img_float[:, :-1]

    gy = np.zeros_like(img_float)
    gy[:-1, :] = img_float[1:, :] - img_float[:-1, :]

    g = (gx ** 2) + (gy ** 2)

    h, w = imagine.shape
    regiuni = []

    for i in range(0, h - 3 + 1, 3):
        for j in range(0, w - 3 + 1, 3):
            regiune_mag = g[i : i + 3, j : j + 3]
            medie = np.mean(regiune_mag)

            reg_og = imagine[i : i + 3, j : j + 3]
            regiuni.append((medie, reg_og))

    regiuni.sort(key = lambda x : x[0], reverse = True)

    k_reg = [element[1] for element in regiuni[:k]]

    return k_reg


def supr_nm(imagine):
    img_float = imagine.astype(np.float64)

    gx = np.zeros_like(img_float)
    gx[:, :-1] = img_float[:, 1:] - img_float[:, :-1]

    gy = np.zeros_like(img_float)
    gy[:-1, :] = img_float[1:, :] - img_float[:-1, :]

    g = np.sqrt((gx ** 2) + (gy ** 2))

    theta = np.arctan2(gy, gx) * 180.0 / np.pi

    theta[theta < 0] +=180

    res = np.zeros_like(g)

    h,w = imagine.shape

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            unghi = theta[i, j]
            mag = g[i, j]

            vec1 = 255
            vec2 = 255

            if (0 <= unghi < 22.5) or (157.5 <= unghi <= 180):
                vec1 = g[i, j + 1]
                vec2 = g[i, j - 1]
            elif (22.5 <= unghi < 67.5):
                vec1 = g[i + 1, j - 1]
                vec2 = g[i - 1, j + 1]
            elif (67.5 <= unghi < 112.5):
                vec1 = g[i + 1, j]
                vec2 = g[i - 1, j]
            elif (112.5 <= unghi < 157.5):
                vec1 = g[i - 1, j - 1]
                vec2 = g[i + 1, j + 1]

            if (mag > vec1) and (mag > vec2):
                res[i, j] = mag
            else:
                res[i, j] = 0

    return res

def vec_binar(imagine):
    h, w = imagine.shape
    vec_final = []

    for i in range(0, h - 3 + 1, 3):