import cv2
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class UltrasoundPreprocessor:
    def __init__(self):
        self.denoise_kernel = 5

    def denoise(self, image):
        """
        使用非局部均值去噪算法去除超声图像中的噪声
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3:
            # 转换为灰度图
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        # 应用非局部均值去噪
        denoised = cv2.fastNlMeansDenoising(
            image,
            None,
            h=10,  # 过滤强度
            templateWindowSize=7,  # 模板窗口大小
            searchWindowSize=21  # 搜索窗口大小
        )
        return denoised

    def enhance_roi(self, image):
        """
        增强感兴趣区域（ROI）
        """
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # 应用CLAHE（对比度受限的自适应直方图均衡）
        enhanced = clahe.apply(image)
        
        # 使用Sobel算子进行边缘检测
        grad_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        gradient = np.uint8(gradient * 255 / np.max(gradient))
        
        # 将边缘信息与增强后的图像融合
        enhanced = cv2.addWeighted(enhanced, 0.7, gradient, 0.3, 0)
        
        return enhanced

    def process(self, image):
        """
        完整的预处理流程
        """
        if isinstance(image, torch.Tensor):
            # 如果输入是张量，转换为numpy数组
            image = image.cpu().numpy()
            image = np.transpose(image, (1, 2, 0))
            image = (image * 255).astype(np.uint8)
        
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        # 1. 去噪
        denoised = self.denoise(image)
        
        # 2. ROI增强
        enhanced = self.enhance_roi(denoised)
        
        # 3. 转换回三通道图像
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        # 4. 转换为PIL图像
        enhanced_pil = Image.fromarray(enhanced_rgb)
        
        return enhanced_pil

# 修改Dataset类以包含预处理步骤
class PreprocessedImageFolder(Dataset):
    def __init__(self, root, annotation_path=None, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.preprocessor = UltrasoundPreprocessor()
        
        if annotation_path is not None:
            with open(annotation_path, 'r') as f:
                data_dir = [line.strip().split(' ') for line in f]
            data_dir = [(x[0], int(x[1])) for x in data_dir]
        else:
            data_dir = sorted(os.listdir(root))
            data_dir = [(x, None) for x in data_dir]
        self.data_dir = data_dir
        self.total_len = len(self.data_dir)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        image_path, label = os.path.join(self.root, self.data_dir[idx][0]), self.data_dir[idx][1]
        image = Image.open(image_path).convert('RGB')
        
        # 应用预处理
        image = self.preprocessor.process(image)
        
        if self.transform:
            image = self.transform(image)
            
        image_name = self.data_dir[idx][0]
        label = image_name if label is None else label
        return image, label

if __name__ == '__main__':
    # 测试代码
    import matplotlib.pyplot as plt
    
    # 创建预处理器
    preprocessor = UltrasoundPreprocessor()
    
    # 加载示例图像
    image_path = './TrainSet/images/train/0.jpg'  # 替换为实际的图像路径
    image = Image.open(image_path).convert('RGB')
    
    # 处理图像
    processed = preprocessor.process(image)
    
    # 显示对比结果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(122)
    plt.title('Processed Image')
    plt.imshow(processed)
    plt.axis('off')
    
    plt.savefig('preprocessing_comparison.png')
    plt.close() 