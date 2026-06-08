import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

traincsvpath ="/Users/gabrielrachieru/PycharmProjects/IA-FMI-2026/SignalObjectDetection-Kaggle/signal-object-detection/train.csv"
trainimgpath = "/Users/gabrielrachieru/PycharmProjects/IA-FMI-2026/SignalObjectDetection-Kaggle/signal-object-detection/train"
testcsvpath = "/Users/gabrielrachieru/PycharmProjects/IA-FMI-2026/SignalObjectDetection-Kaggle/signal-object-detection/test.csv"
testimgpath ="/Users/gabrielrachieru/PycharmProjects/IA-FMI-2026/SignalObjectDetection-Kaggle/signal-object-detection/test"
hog = cv2.HOGDescriptor((64,128),(16,16),(8,8),(8,8),9)
                        #img dimensions (64,128)
                        #blocks size(16,16)
                        #blocks shift (8,8)
                        #nr of bins: 9
traindata = pd.read_csv(traincsvpath)
testdata_df = pd.read_csv(testcsvpath)
traindata_dict = traindata.to_dict('records')
testdata_dict = testdata_df.to_dict('records')
trainfeatures =[]
trainlabels =[]

for row in traindata_dict:
    imgpath = trainimgpath+"/"+str(row['id'])
    img= cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))
    features =hog.compute(img).flatten()
    trainfeatures.append(features)
    trainlabels.append(row['label'])

trainfeatures = np.array(trainfeatures)
trainlabels = np.array(trainlabels)
testfeatures =[]

for row in testdata_dict:
    imgpath = testimgpath+"/"+str(row['id'])
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))
    features = hog.compute(img).flatten()
    testfeatures.append(features)

testfeatures = np.array(testfeatures)
xtrain, xval, ytrain, yval = train_test_split(trainfeatures, trainlabels, test_size=0.2, random_state=26, stratify=trainlabels)
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xval = scaler.transform(xval)
testfeatures = scaler.transform(testfeatures)

model = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=26)
model.fit(xtrain, ytrain)
valpreds = model.predict(xval)
ac=accuracy_score(yval, valpreds)
print(f"acuratete pe validare: {ac*100:.2f}%")
predictii= model.predict(testfeatures)
testdata_df['label']=predictii
testdata_df[['id','label']].to_csv('svm_hog_sub.csv',index=False)



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
cm = confusion_matrix(yval, valpreds, labels=[1, 2, 3, 4, 5])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title(f"SVM cu HOG: {ac * 100:.2f}%")
plt.savefig("svm_hog_confusion_matrix.png")
plt.show()