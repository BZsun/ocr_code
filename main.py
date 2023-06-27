import pandas as pd
import numpy as np
from PIL import Image
import glob
import random
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
# from dataset import OcrDataset
from network import OcrModel
from ipdb import set_trace

class OcrDataset(Dataset):
    def __init__(self, data_dir, labels_df):
        self.data_dir = data_dir
        self.labels_df = labels_df
        self.img_names = labels_df['img'].values.tolist()
        self.labels = labels_df['labels'].values
        self.transform = transforms.Compose([
                                            transforms.ToTensor()
                                        ])
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.data_dir, str(img_name) + '.jpg')
        img = Image.open(img_path).convert('L')
        img = img.resize((32,32), Image.BICUBIC)
        img = img.point(lambda x: 0 if x < 128 else 255)
        img = np.array(img, dtype=np.float32) / 255.0
        label = self.labels[idx]
        img = self.transform(img)
        label = torch.LongTensor(label)

        return img, label

def parse_labelfile(labelfile):
    labs = []
    data = pd.read_csv(labelfile, names=['img', 'chars'])
    letters = [chr(i) for i in range(65,91)]
    letter2id = {val: ind for ind, val in enumerate(letters)}
    for char in data['chars']:
        mid = []
        for letter in char:
            mid.append(letter2id[letter])
        labs.append(mid)
    data['labels'] = labs

    return data, letter2id

def train_model(model, train_loader):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        train_loss = 0
        for batch in tqdm(train_loader):
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(inputs)
            
            loss = loss_fn(outputs.view(-1, num_classes), labels.view(-1))
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print("Epoch {} Avg Loss: {:.4f}".format(epoch, train_loss / len(train_loader)))
    torch.save(model.state_dict(), "ocr_model.pth")

def exec_infer(model, test_path):
    res = []
    model.load_state_dict(torch.load("ocr_model.pth"))
    model.eval()
    test_list = glob.glob(os.path.join(test_path, '*.jpg'))
    transform = transforms.Compose([
                                    transforms.ToTensor()
                                ])
    for file in test_list:
        img_name = os.path.basename(file).split('.')[0]
        img = Image.open(file).convert('L')
        img = img.resize((32,32), Image.BICUBIC)
        img = img.point(lambda x: 0 if x < 128 else 255)
        img = np.array(img, dtype=np.float32) / 255.0
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)
            _, pred = torch.max(output.data, dim=2)
            pred = ''.join([id2letter[i] for i in pred.flatten().tolist()])
            res.append([img_name, pred])

    return res

if __name__ == '__main__':
    file_path = '/mnt/e/github/ocr/data/train'
    test_path = '/mnt/e/github/ocr/data/test'
    TRAIN = False
    batch_size = 8
    num_workers = 4
    epochs = 10
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labelfile = os.path.join(file_path, 'labels.csv')
    labels, letter2id = parse_labelfile(labelfile)
    id2letter = {v: k for k, v in letter2id.items()}
    num_classes = len(letter2id)
    img_list = glob.glob(os.path.join(file_path, '*.jpg'))
    train_set = OcrDataset(file_path, labels)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    model = OcrModel(num_classes)
    model.to(device)
    if TRAIN:
        train_model(model, train_loader)
    else:
        res = exec_infer(model, test_path)
        print(res)
