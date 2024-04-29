# -*- coding: utf-8 -*-
"""

Get some chips

"""


import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import sys
import os
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
from PIL import Image
import numpy as np


#%%
# Set random seeds
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # if use gpu
random.seed(seed)
np.random.seed(seed)


plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
#set size of charactor
plt.rc('font', size=35)#set charactor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainTransformer = transforms.Compose([transforms.Resize((224,224)),#All unified sizes are 224 in width and height
                                       transforms.ToTensor(),#Convert image values to tensors
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])#The mean and variance used for normalization of each layer

#Preprocessing of test set data
testTransformer = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

imagePath = 'food11'
trainData = ImageFolder(os.path.join(imagePath, 'train'), transform=trainTransformer)#Load training set and preprocess
testData = ImageFolder(os.path.join(imagePath, 'test'), transform=testTransformer)#Load test set and preprocess

batchSize=64# set batchsize

trainDataLoader  = DataLoader(trainData, batchSize, shuffle=True, num_workers=0, pin_memory=True)#Encapsulation data
testDataLoader  = DataLoader(testData, batchSize, num_workers=0, pin_memory=True)


trainNum = len(trainDataLoader.dataset)
testNum = len(testDataLoader.dataset)
classNames = trainData.classes
print(classNames)
#%%
# read all figures
plt.figure(figsize=(40,16), dpi=100)
I = 1
for class_ in classNames:#Iterative data
    plt.subplot(3, 5, I)
    randomImgFloder = os.path.join(imagePath, 'train', class_)
    randomImg = random.choice(os.listdir(randomImgFloder))
    # open figure
    image = Image.open(os.path.join(randomImgFloder, randomImg)) 
    plt.imshow(image)
    plt.title(class_)
    plt.axis('off')  #off the coordinate axis
    I += 1
plt.show()
#%%
model = torchvision.models.shufflenet_v2_x2_0(pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=len(classNames), bias=True)


        #%%
# Model input device
model.to(device)
epoch = 10#Set training frequency
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)#Set optimizer
criterion = nn.CrossEntropyLoss()#Set loss function


trainSteps = len(trainDataLoader)#计算每个epoch计算次数
testSteps = len(testDataLoader)
#Place the loss and accuracy values for each epoch
trainLoss, testLoss = [], []
trainAcc, testAcc = [], []

#traning
for epochs in range(epoch):
    beiginTrainLoss = 0#Initialize loss value
    beginvalLoss = 0
    trainAccEpoch = 0#Initialize accuracy value
    testAccEpoch = 0
    
    model.train()#open training model
    trainBar = tqdm(trainDataLoader, file=sys.stdout, colour='red')#Encapsulated as a progress bar for easy observation of training progress
    for step, data in enumerate(trainBar):#Iterative training
        features, labels = data#Extract features and labels
        features, labels = features.to(device), labels.to(device)#Data transmission device
        optimizer.zero_grad()#Gradient zeroing
        outputs = model(features)#Calculation model output
        loss = criterion(outputs, labels)#Calculate loss value
        trainAccEpoch += (torch.argmax(outputs, dim=1) == labels).sum().item()#Calculate the overall accurate number of epochs
        loss.backward()#Backpropagation
        optimizer.step()#Optimizer calculation
        beiginTrainLoss += loss.item()##Calculate the loss value for this epoch
        trainBar.desc = f'TrainEpoch[{epoch} / {epochs+1}], Loss{loss.data:.3f}'#Passed in to the progress bar, printed for each epoch
    trainLoss.append(beiginTrainLoss / trainSteps)#Add loss value list for easy visualization in the future
    
    model.eval()
    with torch.no_grad():
        testBar = tqdm(testDataLoader, file=sys.stdout, colour='red')
        for data in testBar:
            features, labels = data
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)

            testAccEpoch += (torch.argmax(outputs, dim=1) == labels).sum().item()#Calculate the overall accurate number of test epochs
            
            testLoss_ = criterion(outputs, labels)#Calculate the loss value of the test set
            beginvalLoss += testLoss_.item()#Calculate the loss value for this epoch
    testLoss.append(beginvalLoss / testSteps)#Add test set epcoh loss value
            
    testAcc_ = testAccEpoch / testNum#The accuracy of a single epoch is determined by dividing the correct number by the total number
    trainAcc_ = trainAccEpoch / trainNum
    #Save accuracy for easy visualization
    trainAcc.append(trainAcc_)
    testAcc.append(testAcc_)
    
    print(f'TrainEpoch [{epoch}/{epochs+1}] Training loss value: {(beiginTrainLoss / trainSteps):.3f} ,Training accuracy: {trainAcc_:.3f}, Verification accuracy {testAcc_:.3f}')
#torch.save(model, 'shufflenet.pth')
#%%
plt.figure(figsize=(10,8),dpi=100)#Set Canvas
plt.plot(range(epoch),trainLoss,label='Train Loss', color='black')#Training Set Loss
plt.plot(range(epoch),testLoss,label='Test Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10,8),dpi=100)
plt.plot(range(epoch),trainAcc,label='Train Accuracy', color='royalblue')
plt.plot(range(epoch),testAcc,label='Test Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#%%
#Set the model to evaluation mode
model.eval()



#Enable prediction mode
#Store predicted results and real labels
preLabels = []
trueLabels = []
probs = []

model.to('cpu')
#Category
with torch.no_grad():
    for images, labels in tqdm(testDataLoader):
        #Sending data to a device (GPU or CPU)
        images = images.to('cpu')
        #Forward propagation calculation prediction value
        outputs = model(images)
        outputs_ = nn.functional.softmax(outputs, dim=1)
        for P in outputs_:
            probs.append(P.cpu().numpy())
       #Get predicted labels
        predictions = torch.argmax(outputs, dim=1)
        
        #Store predicted results and real labels
        preLabels.extend(predictions.cpu().numpy())
        trueLabels.extend(labels.numpy())
#Replace label values
preLabels = [classNames[I] for I in preLabels]
trueLabels = [classNames[I] for I in trueLabels]
accuracyScore = ACC(trueLabels, preLabels)

print('Accuracy : {}'.format(accuracyScore))


CM = confusion_matrix(trueLabels,
                      preLabels)


plt.figure(figsize=(10,8),dpi=100)
sns.heatmap(CM,
            annot=True,
            xticklabels=classNames,
            yticklabels=classNames,
            fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.title('Test Confusion Matrix')
plt.show()

print(ACC(trueLabels, preLabels))   

