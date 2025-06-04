import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

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
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整图像大小
                transforms.ToTensor(),  # 转换为tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化
                                 std=[0.229, 0.224, 0.225])
            ])
        
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
        image = Image.open(img_path).convert('RGB')
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 使用示例：
if __name__ == '__main__':
    # 创建训练集
    train_dataset = SoundDataset(root_dir='../TrainSet', mode='train')
    # 创建验证集
    val_dataset = SoundDataset(root_dir='../TrainSet', mode='val')
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 打印数据集大小
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 测试数据加载
    for images, labels in train_loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels: {labels}")
        break