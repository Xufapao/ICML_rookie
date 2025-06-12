#TODO 主函数入口
import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.models as models

train_img_pt = "../TrainSet/images/train"

# 配置超参数
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 6  # 根据你的分类数修改
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义训练集和验证集的转换
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),    # 随机垂直翻转
    transforms.RandomRotation(15),      # 随机旋转
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 加载数据集
class SoundDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (string): 数据集根目录
            mode (string): 'train' 或 'val'，用于选择训练集或验证集
            transform (callable, optional): 可选的图像转换
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        # 读取对应的txt文件
        if mode == 'train':
            txt_file = os.path.join(root_dir, 'labels', 'train.txt')
        else:
            txt_file = os.path.join(root_dir, 'labels', 'val.txt')
            
        self.samples = []
        self.labels = []
        with open(txt_file, 'r') as f:
            for line in f:
                img_name, label = line.strip().split()  # 分割图片名和标签
                self.samples.append(img_name)
                self.labels.append(int(label))  # 将标签转换为整数

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.samples[idx]
        label = self.labels[idx]
        img_path = os.path.join(self.root_dir, 'images', 'train', img_name)
        
        # 读取图像
        image = Image.open(img_path)
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 定义模型
def get_model():
    # 加载预训练的ResNet18模型
    model = models.resnet18(pretrained=True)
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 获取特征维度
    num_features = model.fc.in_features
    
    # 创建多层全连接网络
    fc_layers = []
    layer_sizes = [num_features] + [512] * 98 + [NUM_CLASSES]  # 100层，最后一层输出类别数
    
    for i in range(len(layer_sizes)-1):
        fc_layers.extend([
            nn.Linear(layer_sizes[i], layer_sizes[i+1]),
            nn.ReLU(),
            nn.BatchNorm1d(layer_sizes[i+1]),
            nn.Dropout(0.5)
        ])
    
    # 移除最后一层的ReLU、BatchNorm和Dropout
    fc_layers = fc_layers[:-3]
    
    # 替换原始的fc层
    model.fc = nn.Sequential(*fc_layers)
    
    # 解冻fc层的参数
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return epoch_loss, accuracy

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    return epoch_loss, accuracy

if __name__ == "__main__":
    # 创建训练集和验证集
    train_dataset = SoundDataset(root_dir='../TrainSet', mode='train', transform=train_transform)
    val_dataset = SoundDataset(root_dir='../TrainSet', mode='val', transform=val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 初始化模型
    model = get_model()
    model = model.to(DEVICE)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    # 训练循环
    best_val_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # 打印训练信息
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved best model with validation accuracy: {val_acc:.2f}%')
        print('--------------------')