import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import argparse
from image_preprocessing import PreprocessedImageFolder
from model import ResNetSwinTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============== Training ==============
def training(model, optimizer, train_loader, now_epoch, num_epochs):
    model.train()
    losses = []
    pbar = tqdm(train_loader, total=len(train_loader), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]" + " " * (80 - 10 - 10 - 10 - 10 - 3))
    criterion = nn.CrossEntropyLoss()
    
    for step, data in enumerate(pbar, 1):
        image, label = data
        image, label = image.to(device), label.to(device)
        
        optimizer.zero_grad()
        pred = model(image)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        pbar.set_description(f'Epoch {now_epoch} [TRAIN] loss = {losses[-1]:.2f}')

    print(f'Epoch {now_epoch} / {num_epochs} [TRAIN] mean loss = {np.mean(losses):.2f}')

def evaluate(model, val_loader):
    model.eval()
    preds, targets = [], []
    print("Evaluating...")
    with torch.no_grad():
        for data in val_loader:
            image, label = data
            image = image.to(device)
            pred = model(image)
            pred = pred.cpu().numpy().argmax(axis=1)
            preds.append(pred)
            targets.append(label.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    acc = np.mean(np.float32(preds == targets))
    return acc

def run(model, optimizer, train_loader, val_loader, num_epochs, modelroot):
    best_acc = 0
    for epoch in range(num_epochs):
        training(model, optimizer, train_loader, epoch, num_epochs)
        acc = evaluate(model, val_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(modelroot, 'best.pth'))
        print(f'Epoch {epoch} / {num_epochs} [VAL] best_acc = {best_acc:.2f}, acc = {acc:.2f}')

# ============== Test ==================
def test(model, test_loader, result_path):
    model.eval()
    preds = []
    names = []
    print("Testing...")
    with torch.no_grad():
        for data in test_loader:
            image, image_names = data
            image = image.to(device)
            pred = model(image)
            pred = pred.cpu().numpy().argmax(axis=1)
            preds.append(pred)
            names.extend(image_names)
    preds = np.concatenate(preds)
    with open(result_path, 'w') as f:
        for name, pred in zip(names, preds):
            f.write(name + ' ' + str(pred) + '\n')

# ============== Main ==============
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./TrainSet')
    parser.add_argument('--modelroot', type=str, default='./model_save')
    parser.add_argument('--testonly', action='store_true', default=False)
    parser.add_argument('--loadfrom', type=str, default='./model_save/best.pth')
    parser.add_argument('--result_path', type=str, default='./result.txt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    args = parser.parse_args()

    # 创建模型保存目录
    os.makedirs(args.modelroot, exist_ok=True)

    # 初始化模型
    model = ResNetSwinTransformer(num_classes=6, pretrain=True)
    model = model.to(device)

    transform_train = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if not args.testonly:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05)
        train_loader = DataLoader(
            PreprocessedImageFolder(
                root=os.path.join(args.dataroot, 'images/train'),
                annotation_path=os.path.join(args.dataroot, 'labels/train.txt'),
                transform=transform_train
            ),
            batch_size=args.batch_size,
            num_workers=2,
            shuffle=True,
            pin_memory=True
        )
        val_loader = DataLoader(
            PreprocessedImageFolder(
                root=os.path.join(args.dataroot, 'images/train'),
                annotation_path=os.path.join(args.dataroot, 'labels/val.txt'),
                transform=transform_val
            ),
            batch_size=args.batch_size,
            num_workers=2,
            shuffle=False,
            pin_memory=True
        )
        run(model, optimizer, train_loader, val_loader, args.num_epochs, args.modelroot)
    else:
        test_loader = DataLoader(
            PreprocessedImageFolder(
                root=args.dataroot,
                transform=transform_val
            ),
            batch_size=args.batch_size,
            num_workers=2,
            shuffle=False,
            pin_memory=True
        )
        model.load_state_dict(torch.load(args.loadfrom))
        test(model, test_loader, args.result_path) 
