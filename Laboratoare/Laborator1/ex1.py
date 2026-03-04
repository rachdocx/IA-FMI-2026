import numpy as np
import matplotlib.pyplot as plt
from skimage import io
imgs = np.zeros((9,400,600))
for i in range (9):
    imgs[i] = np.load(f"images/car_{i}.npy")

#print(imgs)

suma = np.sum(imgs)
print(suma)

suma_coloane = np.sum(imgs, axis = (1, 2))
print(suma_coloane)

print(np.argmax(suma_coloane, axis = 0))
print(np.argmax(suma_coloane))

m_img = np.mean(imgs, axis=0)
#
# io.imshow(m_img.astype(np.uint8))
# io.show()

print(np.std(imgs))

imagini_normalizate  = (imgs - m_img) / np.std(imgs)
for i  in range (9):
    io.imshow(imagini_normalizate[i].astype(np.uint8))
    io.show()

imagini_cropate = imgs[:, 200:300, 280:400]

for i  in range (9):
    io.imshow(imagini_cropate[i].astype(np.uint8))
    io.show()