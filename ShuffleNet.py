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
# 设置随机种子
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # 如果使用GPU
random.seed(seed)
np.random.seed(seed)

#%%
#显示中文
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
#设置字体大小
plt.rc('font', size=35)#设置字体


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainTransformer = transforms.Compose([transforms.Resize((224,224)),#所有统一尺寸为宽高224
                                       transforms.ToTensor(),#将图片数值转为张量
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])#R,G,B每层的归一化用到的均值和方差

#测试集数据预处理
testTransformer = transforms.Compose([transforms.Resize((224,224)),##所有同意尺寸为宽高224
                                      transforms.ToTensor(),#将图片转为张量
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])#进行归一化

imagePath = 'food11'
trainData = ImageFolder(os.path.join(imagePath, 'train'), transform=trainTransformer)#加载训练集并预处理
testData = ImageFolder(os.path.join(imagePath, 'test'), transform=testTransformer)#加载测试集并预处理

batchSize=64#设置batchsize

trainDataLoader  = DataLoader(trainData, batchSize, shuffle=True, num_workers=0, pin_memory=True)#封装数据
testDataLoader  = DataLoader(testData, batchSize, num_workers=0, pin_memory=True)#封装数据


trainNum = len(trainDataLoader.dataset)
testNum = len(testDataLoader.dataset)
classNames = trainData.classes
print(classNames)
#%%
# 显示所读取的图片
plt.figure(figsize=(40,16), dpi=100)
I = 1
for class_ in classNames:#迭代数据
    plt.subplot(3, 5, I)
    randomImgFloder = os.path.join(imagePath, 'train', class_)
    randomImg = random.choice(os.listdir(randomImgFloder))
    # 打开图片
    image = Image.open(os.path.join(randomImgFloder, randomImg)) 
    plt.imshow(image)
    plt.title(class_)
    plt.axis('off')  # 关闭坐标轴
    I += 1
plt.show()
#%%
model = torchvision.models.shufflenet_v2_x2_0(pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=len(classNames), bias=True)


        #%%
# 模型送入设备
model.to(device)
epoch = 10#设置训练次数
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)#设置优化器
criterion = nn.CrossEntropyLoss()#设置损失函数


trainSteps = len(trainDataLoader)#计算每个epoch计算次数
testSteps = len(testDataLoader)
#放置每个epcoh的损失值与准确率值
trainLoss, testLoss = [], []
trainAcc, testAcc = [], []

#开始训练
for epochs in range(epoch):#迭代次数
    beiginTrainLoss = 0#初始化损失值
    beginvalLoss = 0
    trainAccEpoch = 0#初始化准确率值
    testAccEpoch = 0
    
    model.train()#开启训练模型
    trainBar = tqdm(trainDataLoader, file=sys.stdout, colour='red')#封装为进度条, 方便观察训练进度
    for step, data in enumerate(trainBar):#迭代训练
        features, labels = data#取出特征和标签
        features, labels = features.to(device), labels.to(device)#数据送入设备
        optimizer.zero_grad()#梯度清零
        outputs = model(features)#计算模型输出
        loss = criterion(outputs, labels)#计算损失值
        trainAccEpoch += (torch.argmax(outputs, dim=1) == labels).sum().item()#计算这个epoch的总体准确个数
        loss.backward()#反向传播
        optimizer.step()#优化器计算
        beiginTrainLoss += loss.item()##计算这个epoch的损失值
        trainBar.desc = f'TrainEpoch[{epoch} / {epochs+1}], Loss{loss.data:.3f}'#传入到进度条, 每个epoch打印
    trainLoss.append(beiginTrainLoss / trainSteps)#加入损失值列表, 方便后续可视化
    
    model.eval()
    with torch.no_grad():
        testBar = tqdm(testDataLoader, file=sys.stdout, colour='red')#测试集#封装为进度条, 方便观察训练进度
        for data in testBar:#迭代测试集
            features, labels = data#取出特征和标签
            features, labels = features.to(device), labels.to(device)#数据送入设备
            outputs = model(features)##计算模型输出

            testAccEpoch += (torch.argmax(outputs, dim=1) == labels).sum().item()#计算测试epoch的总体准确个数
            
            testLoss_ = criterion(outputs, labels)#计算测试集损失值
            beginvalLoss += testLoss_.item()#计算这个epoch的损失值
    testLoss.append(beginvalLoss / testSteps)#添加测试集epcoh损失值
            
    testAcc_ = testAccEpoch / testNum#正确个数除以总体个数即为单个epoch的准确率
    trainAcc_ = trainAccEpoch / trainNum
    #将准确率保存起来, 方便可视化
    trainAcc.append(trainAcc_)
    testAcc.append(testAcc_)
    
    print(f'TrainEpoch [{epoch}/{epochs+1}] Training loss value: {(beiginTrainLoss / trainSteps):.3f} ,Training accuracy: {trainAcc_:.3f}, Verification accuracy {testAcc_:.3f}')
#torch.save(model, 'shufflenet.pth')
#%%
plt.figure(figsize=(10,8),dpi=100)#设置画布
plt.plot(range(epoch),trainLoss,label='Train Loss', color='black')#训练集Loss
plt.plot(range(epoch),testLoss,label='Test Loss', color='red')
plt.xlabel('Epoch')#设置x轴名字
plt.ylabel('Loss')#设置y轴名字
plt.legend()
plt.show()

plt.figure(figsize=(10,8),dpi=100)#设置画布
plt.plot(range(epoch),trainAcc,label='Train Accuracy', color='royalblue')#测试集Loss
plt.plot(range(epoch),testAcc,label='Test Accuracy', color='green')
plt.xlabel('Epoch')#设置x轴名字
plt.ylabel('Accuracy')#设置y轴名字
plt.legend()
plt.show()
#%%
# 设置模型为评估模式
model.eval()



#开启预测模式
# 存储预测结果和真实标签
preLabels = []
trueLabels = []
probs = []

model.to('cpu')
#类别
with torch.no_grad():
    for images, labels in tqdm(testDataLoader):
        # 将数据送入设备（GPU 或 CPU）
        images = images.to('cpu')
        # 前向传播计算预测值
        outputs = model(images)
        outputs_ = nn.functional.softmax(outputs, dim=1)
        for P in outputs_:
            probs.append(P.cpu().numpy())
        # 获取预测标签
        predictions = torch.argmax(outputs, dim=1)
        
        # 存储预测结果和真实标签
        preLabels.extend(predictions.cpu().numpy())
        trueLabels.extend(labels.numpy())
#替换标签值
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

