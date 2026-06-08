import numpy as np
import pandas as pd
import cv2
import os
class knn:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels


    def clasificare(self, test_image, k, metrica):
        if metrica == 'l2':  #calculul distantei euclidiene
            distanta = np.sqrt(np.sum((self.train_images - test_image) ** 2, axis=1))
        elif metrica == 'l1': #calculul distantei manhattan
            distanta = np.sum(np.abs(self.train_images - test_image), axis=1)

        nearest_indices = np.argsort(distanta)  #sortate crescator
        nearest_indices = nearest_indices[:k] # primii k vecini indici

        nearest_labels = self.train_labels[nearest_indices]
        voturi = np.bincount(nearest_labels)
        return np.argmax(voturi)    # clasa cu cele mai multe voturi

def img(csv_path, imgs_path, label=True):
    df = pd.read_csv(csv_path)
    imagini = []
    labels = []

    for index, row in df.iterrows():
        img_path = os.path.join(imgs_path, row['id'])

        imagine = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #incarcam imaginea pe un singur canal
        img_flatten = imagine.flatten() # 2d -> 1d vector    # adica doar greyscale
        imagini.append(img_flatten)

        if label:
            labels.append(row['label'])
    if label:
        return np.array(imagini), np.array(labels)
    else:
        return np.array(imagini), df

train_csv_path = ('/Users/gabrielrachieru/PycharmProjects/'
                  'IA-FMI-2026/SignalObjectDetection-Kaggle/signal-object-detection/train.csv')
train_imgs_path = ('/Users/gabrielrachieru/PycharmProjects'
                   '/IA-FMI-2026/SignalObjectDetection-Kaggle/signal-object-detection/train')

test_csv_path = ('/Users/gabrielrachieru/PycharmProjects'
                 '/IA-FMI-2026/SignalObjectDetection-Kaggle/signal-object-detection/test.csv')
test_imgs_path = ('/Users/gabrielrachieru/PycharmProjects'
                  '/IA-FMI-2026/SignalObjectDetection-Kaggle/signal-object-detection/test')

train_images, train_labels = img(train_csv_path, train_imgs_path, label=True)
test_images, test_df = img(test_csv_path, test_imgs_path, label=False)

model = knn(train_images, train_labels)

predictii = []
for i in range(len(test_images)):
    predictie = model.clasificare(test_images[i], k=5, metrica='l2')
    predictii.append(predictie)

test_df['label'] = predictii
test_df.to_csv('knn_sub.csv', index=False)