# -*- coding: utf-8 -*-
"""

Get some chips

"""

import torch
print(torch.__version__)
import torchvision
import numpy as np
from PIL import Image as PILImage
import tkinter as tk
import ttkbootstrap as ttk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg#绘图显示
from tkinter import filedialog#选择文件
from tkinter import messagebox
import torch.nn as nn
from torchvision.transforms import transforms
import torch.nn.functional as F
from tkinter import filedialog#选择文件
import os
classNames = ['apple_pie', 'cheesecake', 'chicken_curry', 'french_fries', 'fried_rice', 'hamburger', 'hot_dog', 'ice_cream', 'omelette', 'pizza', 'sushi']

def chooseModel():
    global model
    modelPath = filedialog.askopenfilename()
    modelName = os.path.basename(modelPath)
    model = torchvision.models.shufflenet_v2_x2_0(pretrained=False)
    model.fc = nn.Linear(in_features=2048, out_features=len(classNames), bias=True)
    model = torch.load(modelPath, map_location='cpu')
    model.eval()




    win.title(modelName)
# 设置模型为评估模式
Transformer = transforms.Compose([transforms.Resize((224, 224)),##所有同意尺寸为宽高224
                                     transforms.ToTensor(),#将图片转为张量
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])#进行归一化
device = torch.device('cpu')

# 定义识别函数
def identify():
    global answer, outputs
    filePath = filedialog.askopenfilename()
    img = PILImage.open(filePath).convert('RGB')
    fig, ax = plt.subplots(figsize=(2.5,2.5))
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    canvas = FigureCanvasTkAgg(fig, master=win)#设置放置画布
    canvas.draw()
    canvas.get_tk_widget().place(x=100, y=10)
    
    img_ = Transformer(img)
    img_ = torch.unsqueeze(img_, 0)
    img_ = img_.to(device)
    outputs = model(img_)
    outputs = F.softmax(outputs, dim=1)
    predicted = torch.max(outputs, dim=1)[1].cpu().item()
    answer = classNames[predicted]
    outputs = outputs[0, :]

win = ttk.Window(title='Image Recognition')#实例化窗口
PCWidth = win.winfo_screenwidth()#获取屏幕宽
PCHeight = win.winfo_screenheight()#获取屏幕高
win.geometry('500x430+{}+{}'.format(int(PCWidth/4),int(PCHeight/4)))#设置窗口尺寸

selectButton = tk.Button(win, text='Select Image', command=identify)
selectButton.place(x=170, y=330+30, width=150, height=40)

selectModelButton = tk.Button(win, text='Select Model', command=chooseModel)
selectModelButton.place(x=10, y=330+30, width=150, height=40)

answerLabel = tk.Label(text=' ')
answerLabel.place(x=170+30, y=260+60)

def deliever():
    try:
        answerLabel.config(text='{}'.format(answer))
        
    except:
        messagebox.showerror('Tip', message='Please select a image')

answerButton = tk.Button(win, text='Recognition', command=deliever)
answerButton.place(x=330, y=330+30, width=150, height=40)


win.mainloop()


