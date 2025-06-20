# -*- coding: utf-8 -*-
"""Part 3: Preprocessing

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1t8ZEcaU1EW8aLCJjcsF6ib5AxjN3Ol8h
"""

from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image
import cv2
import torch
import numpy as np

# Tokenizer and transforms
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text_transform = lambda text: tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_image(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        return img_transform(image).unsqueeze(0)
    except:
        print(f"⚠️ Could not load image from {img_path}, returning zeros.")
        return torch.zeros(1, 3, 224, 224)

def load_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(16):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))
            frame = torch.tensor(frame).permute(2, 0, 1) / 255.0
            frames.append(frame)
        cap.release()
        if not frames:
            return torch.zeros(1, 3, 16, 112, 112)
        video_tensor = torch.stack(frames).unsqueeze(0).permute(0, 2, 1, 3, 4)
        return video_tensor
    except:
        print(f"⚠️ Could not load video from {video_path}, returning zeros.")
        return torch.zeros(1, 3, 16, 112, 112)