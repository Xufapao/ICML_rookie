import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def load_labels(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    labels = [int(line.strip().split(' ')[1]) for line in lines]
    return labels

def plot_distribution(counter, title):
    plt.figure(figsize=(10, 6))
    classes = sorted(counter.keys())
    counts = [counter[c] for c in classes]
    
    plt.bar(classes, counts)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    
    # Add count numbers on top of each bar
    for i, count in enumerate(counts):
        plt.text(classes[i], count, str(count), ha='center', va='bottom')
    
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def main():
    # 加载训练集标签
    train_labels = load_labels('./TrainSet/labels/train.txt')
    train_counter = Counter(train_labels)
    
    # 打印训练集统计信息
    print("\nTraining Set Statistics:")
    print("-" * 30)
    total_train = len(train_labels)
    for class_idx in sorted(train_counter.keys()):
        count = train_counter[class_idx]
        percentage = (count / total_train) * 100
        print(f"Class {class_idx}: {count} images ({percentage:.2f}%)")
    
    # 绘制分布图
    plot_distribution(train_counter, "Training Set Distribution")

if __name__ == '__main__':
    main() 