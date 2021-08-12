'''
import os
import warnings
import json
import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

json_dir = '../MaskRcnn/data/detection_ver200109_small.json'
file = open(json_dir, 'r', encoding='utf-8')
s = json.load(file)
file.close()
dataset = image.get_MRCNN_json(s)
all_patients = len(dataset.image_info)
all_ids = np.arange(all_patients)
for i in range(all_patients):
    # 输入的dataset是spondylolisthesis_dataset类型变量，现在是其中第i张图，把各个成员变量都拆出来。
    patient_id = dataset.image_info[i]['patient']
    image = dataset.image_info[i]['image']
    image=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #image = np.transpose(image, [2,0,1])
    mask = dataset.image_info[i]['mask']
    masks = dataset.image_info[i]['masks']
    width = dataset.image_info[i]['height']
    height = dataset.image_info[i]['width']
    category_for_dataset = dataset.image_info[i]['category']
    if i==1:
        cv2.imwrite('test1.png', image)
        '''
'''
import torch
import torchvision
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# For training
images=torch.rand(4, 3, 600, 1200)
boxes =torch.rand(4, 11, 4)
labels = torch.randint(1, 91, (4, 11))
images = list(image for image in images)
print(images)
'''

'''
import torch
import torchvision
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# For training
json_dir = '../MaskRcnn/data/detection_ver200109_small.json'
file = open(json_dir, 'r', encoding='utf-8')
s = json.load(file)
file.close()
dataset = Image.get_MRCNN_json(s)
all_patients = len(dataset.image_info)
all_ids = np.arange(all_patients)
images=[]
for i in range(all_patients):
    image = dataset.image_info[i]['image']
    image=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = np.transpose(image, [2,0,1])
    images.append(image)
    
    boxes=[]
    num_masks = len(dataset.image_info[i]["masks"])
    for m in range(0, num_masks):
        mask = dataset.image_info[i]["masks"][m]
        #print(np.array(mask).shape)
        if mask.max() < 0.5:
            continue
        box=Image.extract_bboxes(mask)
        boxes.append(box)
        
    organ_class = []
    for j in range(len(dataset.image_info[i]["category"])):
        current_class = j
        organ_class.append(current_class)
    print(np.array(boxes).shape)
    #print(np.array(organ_class).shape)

import torch
import torchvision
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
boxes =torch.rand(4, 11, 4)
labels = torch.randint(1, 91, (4, 11))

for i in range(4):
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
    print(boxes[i].shape)
    print(labels[i].shape)
'''
'''
import torch
import torchvision
import os
import warnings
import Image
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# For training
json_dir = '../MaskRcnn/data/detection_ver200109_small.json'
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
    image=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = np.transpose(image, [2,0,1])
    image=image/255
    image=torch.from_numpy(image)
    images.append(image.to(torch.float32))
    
    boxes=[]
    num_masks = len(dataset.image_info[i]["masks"])
    for m in range(0, num_masks):
        mask = dataset.image_info[i]["masks"][m]
        #print(np.array(mask).shape)
        if mask.max() < 0.5:
            continue
        box=Image.extract_bboxes(mask)
        boxes.append(box)
        
    organ_class = []
    for j in range(len(dataset.image_info[i]["category"])):
        current_class = j
        organ_class.append(current_class)
    
    boxes=torch.Tensor(boxes)
    organ_class=torch.Tensor(organ_class)
    d = {}
    d['boxes'] = boxes
    d['labels'] = organ_class.to(torch.int64)
    targets.append(d)

print(targets[0]['boxes'].cuda().cpu().numpy())
'''
'''
import Image
import json
json_dir = '../MaskRcnn/data/detection_ver200109_small.json'
file = open(json_dir, 'r', encoding='utf-8')
s = json.load(file)
file.close()
dataset = Image.get_MRCNN_json(s)
x=dataset.image_info[0]['image']
print(x.shape)
x=Image.to_torch(x)
print(x.shape)
'''

import torch
import torchvision
import os
import warnings
import Image
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.set_printoptions(threshold=np.inf)

#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,num_classes=7)
# For training
json_dir = '../MaskRcnn/data/detection_ver190129.json'
file = open(json_dir, 'r', encoding='utf-8')
s = json.load(file)
file.close()
dataset = Image.get_MRCNN_json(s)
all_patients = len(dataset.image_info)
all_ids = np.arange(all_patients)
images=[]
targets = []
print(dataset.image_info[0]["mask"])
#print("______________________________________________________________________________________________________________________")
#print("______________________________________________________________________________________________________________________")
#print("______________________________________________________________________________________________________________________")
#print("______________________________________________________________________________________________________________________")
#print(dataset.image_info[449]["mask"])
'''
for i in range(all_patients):
    #image = dataset.image_info[i]['image']
    #image=Image.to_torch(image)
    #image = image.to(device)
    #images.append(image)
    
    boxes=[]   
    organ_class = []
    num_masks = len(dataset.image_info[i]["masks"])
    print(num_masks)
    for m in range(0, num_masks):
        mask = dataset.image_info[i]["masks"][m]
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
    d['boxes'] = boxes
    d['labels'] = organ_class.to(torch.int64)
    targets.append(d)
    
image = dataset.image_info[449]['image']
image=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# 矩形左上角和右下角的坐标，绘制一个红色的矩形
box=targets[449]['boxes'].cuda().cpu().detach().numpy()
print(box)
for i in range(len(box)):
    ptLeftTop = (box[i][0], box[i][1])  #（左上角x, 左上角y）
    ptRightBottom = (box[i][2], box[i][3]) #（右下角x, 右下角y）
    point_color = (int(i*255/9), 0 , 255-int(i*255/9))  # RGB框的颜色，自定
    thickness = 1
    lineType = 4
    cv2.rectangle(image, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

cv2.imwrite("new_result.jpg", image)  # 将画过矩形框的图片保存
'''
'''
import torch
import torchvision
import os
import warnings
import Image
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,num_classes=7)
# For training
json_dir = '../MaskRcnn/data/detection_ver190129.json'
file = open(json_dir, 'r', encoding='utf-8')
s = json.load(file)
file.close()
dataset = Image.get_MRCNN_json(s)
print(len(dataset.image_info))
'''