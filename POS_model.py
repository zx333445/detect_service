#!/usr/bin/env python
# coding=utf-8

import time
import io
import torch
from PIL import Image, ImageDraw, ImageFont
from netcmd.backbone_utils import swin_fpn_backbone
from netcmd.cmd import CascadeMiningDet
from torchvision import transforms

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def creat_model():

    # get devices
    
    print("using {} device.".format(device))

    # create model
    CLASSES = {"__background__", "CTC", "CTC样"}
    backbone = swin_fpn_backbone()
    model = CascadeMiningDet(backbone, num_classes=len(CLASSES))
    model.to(device)

    pth_path = os.path.join(os.path.dirname(__file__), "parameters/swins_cmd.pth")
    statdic = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(statdic)
    model.eval()

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def predict(image_bytes):
    # creat model
    model = creat_model()

    # load image
    original_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # from pil image to tensor, do not normalize image
    transform=transforms.Compose([transforms.ToTensor()])
    img = transform(original_img)
    
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)  # type: ignore
    outputs = model(img.to(device))
    

    if len(outputs[-1]["boxes"]) == 0:
        print('没有检测到CTC/CTC样细胞')
    else:
        new_output_index = torch.where(outputs[-1]["scores"] > 0.6)
        new_boxes = outputs[-1]["boxes"][new_output_index]
        new_scores = outputs[-1]["scores"][new_output_index]
        new_labels = outputs[-1]["labels"][new_output_index]

        coords = [] 
        for i in range(len(new_boxes)):
            new_box = new_boxes[i].tolist()
            coords.append([new_box[0], new_box[1],
                            new_box[2], new_box[3]])
        coords_score = new_scores.tolist()
        coords_labels = new_labels.tolist()
        if len(coords) == 0:
            print('没有检测到CTC/CTC样细胞')
            
        draw = ImageDraw.Draw(original_img)

        # 根据图片大小调整绘制定位框的线条宽度和书写概率文字的大小
        tl = round(0.002*(original_img.size[0]+original_img.size[1]) + 1)
        font = ImageFont.truetype(font="/usr/share/fonts/dejavu/DejaVuSans.ttf",size=5*tl)
        for box,score,label in zip(coords,coords_score,coords_labels):
            if label == 1:
                draw.rectangle(box, outline=(255,0,0), width=tl)
                draw.text((box[0] + tl,box[1] + tl), f'{score:.2f}',(255,0,0),font)
            else:
                # draw.rectangle(box, outline=(255,97,0), width=tl)
                draw.ellipse(box,outline=(255,0,0),width=tl)
                draw.text((box[0] + tl,box[1] + tl), f'{score:.2f}',(255,0,0),font)

    return original_img


if __name__=="__main__":
    image_path = '/home/stat-zx/CTC_data/JPEGImages/07_Liao_img_036.jpg'
    predict(image_path)
