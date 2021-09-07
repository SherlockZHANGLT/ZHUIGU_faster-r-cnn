import torch
import torchvision
import os
import warnings
import Image
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import sys
import time
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,num_classes=10)
# For training
json_dir = 'detection_ver190129.json'
file = open(json_dir, 'r', encoding='utf-8')
s = json.load(file)
file.close()
dataset = Image.get_MRCNN_json(s)
all_patients = len(dataset.image_info)
all_ids = np.arange(all_patients)
images=[]
targets = []
for i in range(all_patients):
    image = dataset.image_info[i]['image']
    image=Image.to_torch(image)
    image = image.to(device)
    images.append(image)
    
    boxes=[]   
    organ_class = []
    #num_masks = len(dataset.image_info[i]["masks"])
    num_masks=9
    for m in range(num_masks):
        mask = (dataset.image_info[i]["mask"]==m+1)
        #print(np.array(mask).shape)
        if mask.max() < 0.5:
            continue
        box=Image.extract_bboxes(mask)
        boxes.append(box)
        current_class = m+1
        organ_class.append(current_class)
    
    boxes=torch.Tensor(boxes)
    organ_class=torch.Tensor(organ_class)
    d = {}
    d['boxes'] = boxes.to(device)
    #d['boxes'] = boxes
    organ_class=organ_class.to(torch.int64)
    d['labels'] = organ_class.to(device)
    #d['labels'] = organ_class
    targets.append(d)
    
model = model.cuda()

import utils
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]

# SGD
optimizer = torch.optim.SGD(params, lr=0.0003,momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

for epoch in range(30):
    # train for one epoch, printing every 10 iterations
    # engine.py的train_one_epoch函数将images和targets都.to(device)了
    model.train()
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, all_patients - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    for m in range(45):
        image = images[m*10:(m+1)*10-2]
        target = targets[m*10:(m+1)*10-2]
        loss_dict = model(image, target)
        losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        lr_scheduler.step()
        if(m ==44):
            print("finish_"+str(epoch))
            print("loss:"+str(losses))
    model.eval()
    error=0
    s=0
    for m in range(450):
        if m % 10 != 0:
            x=dataset.image_info[m]['image']
            mmask=dataset.image_info[m]['mask']
            x=Image.to_torch(x)
            test_images=[]
            test_images.append(x.to(device))
            predictions = model(test_images)
            for pre in predictions:
                box=pre['boxes'].cuda().cpu().detach().numpy()
                labels = pre['labels'].cuda().cpu().detach().numpy()
                scores=pre['scores'].cuda().cpu().detach().numpy()
                num=[-1,-1,-1,-1,-1,-1,-1,-1,-1]
                num=np.array(num)
                for i in range(len(box)):
                    if(scores[i]<50):
                        continue
                    elif(num[labels[i]-1]==-1):
                        num[labels[i]-1]=i
                    elif (scores[i]>scores[num[labels[i]-1]]):
                        num[labels[i]-1]=i
                for i in range(9):
                    if(num[i] != -1):
                        bbox=mmask[int(box[num[i]][0]):int(box[num[i]][2]),int(box[num[i]][1]):int(box[num[i]][3])]
                        bbox=bbox.flatten()
                        bbox=bbox.tolist()
                        if(max(bbox,key=bbox.count) != i+1):
                            error=error+1
                        s=s+1
    if(s):
        print("finish_"+str(epoch)+"_ :"+str(1-error/s))
    else:
        print("finish_"+str(epoch)+"_not found!")
print('finish training')

model.eval()
error=0
s=0
for m in range(450):
    if m % 10 !=0:
        x=dataset.image_info[m]['image']
        mmask=dataset.image_info[m]['mask']
        x=Image.to_torch(x)
        images=[]
        images.append(x.to(device))
        predictions = model(images)
        for pre in predictions:
            box=pre['boxes'].cuda().cpu().detach().numpy()
            labels = pre['labels'].cuda().cpu().detach().numpy()
            scores=pre['scores'].cuda().cpu().detach().numpy()
            if m % 18 == 0 :
                print(m)
                print(labels)
                print(scores)
            num=[-1,-1,-1,-1,-1,-1,-1,-1,-1]
            num=np.array(num)
            for i in range(len(box)):
                if(num[labels[i]-1]==-1):
                    num[labels[i]-1]=i
                elif (scores[i]>scores[num[labels[i]-1]]):
                    num[labels[i]-1]=i
            pic = dataset.image_info[m]['image']
            pic=cv2.cvtColor(pic, cv2.COLOR_GRAY2BGR)
            for i in range(9):
                    if(num[i] != -1):
                        bbox=mmask[int(box[num[i]][0]):int(box[num[i]][2]),int(box[num[i]][1]):int(box[num[i]][3])]
                        bbox=bbox.flatten()
                        bbox=bbox.tolist()
                        if(max(bbox,key=bbox.count) != i+1):
                            error=error+1
                        s=s+1
                        ptLeftTop = (int(box[num[i]][0]),int(box[num[i]][1]))  #（左上角x, 左上角y）
                        ptRightBottom = (int(box[num[i]][2]), int(box[num[i]][3])) #（右下角x, 右下角y）
                        point_color = (int(labels[i]*205/10), 0 , 255-int(labels[i]*205/10))  # RGB框的颜色，自定
                        thickness = 1
                        lineType = 4
                        cv2.rectangle(pic, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
            cv2.imwrite("result/"+str(m)+".jpg", pic)  # 将画过矩形框的图片保存
if(s):
    print("test:"+str(1-error/s))
else:
    print("test:0!")
'''
import torch.optim as optim
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    # zero the parameter gradients
    optimizer.zero_grad()
 
    # forward + backward + optimize
    output = model(images, targets)
    loss = criterion(output, targets['labels'])
    loss.backward()
    optimizer.step()      
    # print statistics
    running_loss += loss.item()
    if i % 20 == 19:    # print every 2000 mini-batches
        print('[%d] loss: %.3f' %
                (epoch + 1, running_loss / 200))
'''
'''
# For inference
#print(targets[0]['boxes'].cuda().cpu().numpy())
#print(targets[0]['labels'].cuda().cpu().numpy())
model.eval()
x=dataset.image_info[0]['image']
x=Image.to_torch(x)
images=[]
images.append(x.to(device))
predictions = model(images)
pic = dataset.image_info[0]['image']
pic=cv2.cvtColor(pic, cv2.COLOR_GRAY2BGR)
for pre in predictions:
    box=pre['boxes'].cuda().cpu().detach().numpy()
    labels = pre['labels'].cuda().cpu().detach().numpy()
    scores=pre['scores'].cuda().cpu().detach().numpy()
    print(scores)
    for i in range(len(box)):
        if(scores[i]>0.5):
            print(box[i])
            print(labels[i])
            print(scores[i])
            ptLeftTop = (box[i][0], box[i][1])  #（左上角x, 左上角y）
            ptRightBottom = (box[i][2], box[i][3]) #（右下角x, 右下角y）
            point_color = (int(labels[i]*205/10), 0 , 255-int(labels[i]*205/10))  # RGB框的颜色，自定
            thickness = 1
            lineType = 4
            cv2.rectangle(pic, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
cv2.imwrite("result.jpg", pic)  # 将画过矩形框的图片保存
 '''   
'''    
# optionally, if you want to export the model to ONNX:
torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)      
boxes =torch.rand(4, 11, 4)
labels = torch.randint(1, 91, (4, 11))
for i in range(all_patients):
    for j in range(11):
        if boxes[i][j][0]>boxes[i][j][2]:
            tmp=boxes[i][j][0].clone().detach()
            boxes[i][j][0]=boxes[i][j][2].clone().detach()
            boxes[i][j][2]=tmp.clone().detach()
        if boxes[i][j][1]>boxes[i][j][3]:
            tmp=boxes[i][j][1].clone().detach()
            boxes[i][j][1]=boxes[i][j][3].clone().detach()
            boxes[i][j][3]=tmp.clone().detach()
    d = {}
    d['boxes'] = boxes[i]
    d['labels'] = labels[i]
    targets.append(d)
'''
