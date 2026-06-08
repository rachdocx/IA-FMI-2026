import random
import copy
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as o

def normdistribpoze():
    picsperclass= np.array([3500, 3000,3000, 3000,3000], dtype=np.float32)
    distr = 1.0 /picsperclass
    distr /= distr.sum() * 5
    return torch.tensor(distr, dtype=torch.float32).to("mps")


class myDataset(Dataset):
    def __init__(self, datadic, imgdir, train = True):
        self.pics= datadic
        self.imgdir = imgdir
        self.train = train

    def __len__(self):
        return len(self.pics)

    def __getitem__(self, item):
        curimg = self.pics[item]
        path = self.imgdir + "/" + curimg['id']
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img =img.astype(np.float32)/255.0

        if self.train:
            if random.random()>0.5:
                img = copy.deepcopy(np.fliplr(img))
            # if random.random()>0.5:
            #     img = copy.deepcopy(np.flip(img, 0))

        img = np.expand_dims(img, axis=0)

        if 'label' in curimg:
            return torch.tensor(img, dtype=torch.float32), torch.tensor(curimg['label']-1, dtype=torch.long)
        else:
            return torch.tensor(img, dtype=torch.float32), curimg['id']

class myCNN(nn.Module):
    def __init__ (self):
        super().__init__()
        self.s0 = nn.Conv2d(1, 16, kernel_size=(7,3), padding=(3,1))
        self.b0 = nn.BatchNorm2d(16)
        self.p0 = nn.MaxPool2d(2)

        self.s1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(32)
        self.p1 = nn.MaxPool2d(2)

        self.s2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.b2 = nn.BatchNorm2d(64)
        self.p2 = nn.MaxPool2d(2)

        self.s3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(128)
        self.p3 = nn.MaxPool2d(2)

        self.s4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.b4 = nn.BatchNorm2d(256)

        self.relu = nn.ReLU()
        self.gap =nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.4)
        self.fc = nn.Linear(256,5)

    def forward(self, x):

        x = self.s0(x)
        x = self.b0(x)
        x = self.relu(x)
        x = self.p0(x)

        x = self.s1(x)
        x = self.b1(x)
        x = self.relu(x)
        x = self.p1(x)

        x = self.s2(x)
        x = self.b2(x)
        x = self.relu(x)
        x = self.p2(x)

        x = self.s3(x)
        x = self.b3(x)
        x = self.relu(x)
        x = self.p3(x)

        x = self.s4(x)
        x = self.b4(x)
        x = self.relu(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x


traincsvpath ="/Users/gabrielrachieru/PycharmProjects/IA-FMI-2026/SignalObjectDetection-Kaggle/signal-object-detection/train.csv"
trainimgpath = "/Users/gabrielrachieru/PycharmProjects/IA-FMI-2026/SignalObjectDetection-Kaggle/signal-object-detection/train"
testcsvpath = "/Users/gabrielrachieru/PycharmProjects/IA-FMI-2026/SignalObjectDetection-Kaggle/signal-object-detection/test.csv"
testimgpath ="/Users/gabrielrachieru/PycharmProjects/IA-FMI-2026/SignalObjectDetection-Kaggle/signal-object-detection/test"

data = pd.read_csv(traincsvpath)
data= data.sample(frac=1, random_state=26).reset_index(drop=True)
datadic = data.to_dict('records')

testdata = pd.read_csv(testcsvpath)
testdic = testdata.to_dict('records')
testDataset = myDataset(testdic, testimgpath, False)
testLoader = DataLoader(testDataset, batch_size=128, shuffle=False)

ntest=len(testdata)
foldspredstest = np.zeros((ntest, 5))
nrfolds=5
ndata=len(data)
foldsize = ndata //nrfolds
device ="mps"
epch =80

valpred = []
vallabels = []

classesweights = normdistribpoze()

for i in range(nrfolds):
    istart = i * foldsize
    if i < nrfolds - 1:
        ifinal = (i+1) * foldsize
    else:
        ifinal = ndata

    print(f"-------fold nr {i}---------------")
    validationdata = datadic[istart:ifinal]
    traindata = datadic[:istart] + datadic[ifinal:]

    trainDataset = myDataset(traindata, trainimgpath, True)
    validationDataset = myDataset(validationdata, trainimgpath, False)

    trainLoader = DataLoader(trainDataset, batch_size=128, shuffle=True)
    validationLoader = DataLoader(validationDataset, batch_size=128, shuffle=False)

    model = myCNN().to(device)

    criterion = nn.CrossEntropyLoss(weight=classesweights, label_smoothing=0.05)
    optimizer = o.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epch, eta_min=1e-5)

    bestacc = 0.0
    bestloss = float('inf')

    nrepchswithnochange = 0

    for e in range(epch):
        model.train()
        loss = 0.0
        for img, label in trainLoader:
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            preds = model(img)
            curloss = criterion(preds, label)
            curloss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss+= curloss.item()

        model.eval()
        validationloss = 0.0
        rightguess = 0
        total = 0

        with torch.no_grad():
            for img, label in validationLoader:
                img = img.to(device)
                label = label.to(device)

                preds = model(img)
                curloss = criterion(preds, label)
                validationloss+=curloss.item()

                dummy, guess = torch.max(preds, 1)
                total +=label.size(0)

                for j in range(len(guess)):
                    if guess[j] == label[j]:
                        rightguess += 1

        avgvalidationloss = validationloss / len(validationLoader)
        acc = (rightguess / total) *100

        if bestloss - avgvalidationloss > 0.01:
            bestloss = avgvalidationloss
            nrepchswithnochange = 0
            torch.save(model.state_dict(), f"model_fold_{i}.pt")
        else:
            if nrepchswithnochange == 15:
                break
            else:
                nrepchswithnochange +=1

        if acc-bestacc > 0.01:
            bestacc =acc
            torch.save(model.state_dict(), f"model_fold_{i}.pt")

        scheduler.step()
        print(f"epoca{e+1} are acuratetea {acc:.2f}%")

    model.load_state_dict(torch.load(f"model_fold_{i}.pt", weights_only=True))
    model.eval()

    with torch.no_grad():
        for img, label in validationLoader:
            img = img.to(device)
            preds = model(img)
            dummy, guess =torch.max(preds, 1)
            valpred.extend(guess.cpu().tolist())
            vallabels.extend(label.cpu().tolist())

    predfold = []
    with torch.no_grad():
        for img, id in testLoader:
            img = img.to(device)
            p1 = torch.softmax(model(img), dim=1)
            img_flip = torch.flip(img, dims=[3])
            p2 = torch.softmax(model(img_flip), dim=1)
            preds = (p1 + p2) / 2.0
            predfold.extend(preds.cpu().tolist())

    foldspredstest += np.array(predfold)

finalprediction = np.argmax(foldspredstest, axis=1)+ 1
testdata['label'] = finalprediction
testdata[['id','label']].to_csv('submisie.csv', index=False)





all_val_labels = np.array(vallabels) + 1
all_val_preds = np.array(valpred) + 1
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
cm = confusion_matrix(all_val_labels, all_val_preds, labels=[1, 2, 3, 4, 5])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title(f"CNN: {acc * 100:.2f}%")
plt.savefig("nn_confusion_matrix1.png")
plt.show()
