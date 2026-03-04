import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

train_images = np.loadtxt('train_images.txt')
train_labels = np.loadtxt('train_labels.txt').astype(int)
test_images = np.loadtxt('test_images.txt')
test_labels = np.loadtxt('test_labels.txt').astype(int)

#2
def values_to_bin(poze, bin):
    return np.digitize(poze, bin) - 1


#3
bins = np.linspace(0, 255, 5)

trained_bin = values_to_bin(train_images, bins)
test_bin = values_to_bin(test_images, bins)

naive_bayes_model = MultinomialNB()

naive_bayes_model.fit(trained_bin, train_labels)

acuratete = naive_bayes_model.score(test_bin, test_labels)

print(acuratete)

#4

for i in [3, 5, 7, 9, 11]:
    bins = np.linspace(0, 255, i)
    trained_bin = values_to_bin(train_images, bins)
    test_bin = values_to_bin(test_images, bins)

    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(trained_bin, train_labels)
    acuratete = naive_bayes_model.score(test_bin, test_labels)
    print(f"numar de bin-uri: {i}, Acuratete: {acuratete}")

# numar de bin-uri: 3, Acuratete: 0.826
# numar de bin-uri: 5, Acuratete: 0.836
# numar de bin-uri: 7, Acuratete: 0.842 => cel mai bun nr de bin uri
# numar de bin-uri: 9, Acuratete: 0.842
# numar de bin-uri: 11, Acuratete: 0.842

#5

bins = np.linspace(0, 255, 7)
trained_bin = values_to_bin(train_images, bins)
test_bin = values_to_bin(test_images, bins)

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(trained_bin, train_labels)

predictii = naive_bayes_model.predict(test_bin)

greseli = np.where(predictii != test_labels)[0]

for i in range(10):
    j = greseli[i]
    imagine_gresita = np.reshape(test_images[j], (28, 28))
    plt.title(f"trebuia sa fie {predictii[j]}")
    plt.imshow(imagine_gresita.astype(np.uint8), cmap='gray')
    plt.show()

#6
def confusion_matrix(y_true, y_pred):
    num_classes = int(np.max(y_true)) + 1
    matrix = np.zeros((num_classes, num_classes))
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label][pred_label] += 1
    return matrix

matrice = confusion_matrix(test_labels, predictii)
print(matrice)