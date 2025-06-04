# 注意事项

1. 重点关注敏感性 (Recall for malignant)：临床场景下“恶性”漏检后果严重，可以在 loss 中给恶性更高权重，或者 threshold 调整时优先保证 high recall。

2. 预处理非常关键：尤其超声图像散斑噪声重，建议先做基础去噪与对比度增强，再送入 CNN。

3. 可视化训练曲线：用 TensorBoard 或 matplotlib（注意用 python_user_visible）把 loss/accuracy 曲线可视化，便于发现过拟合、欠拟合。

4. 在中小型数据集上，ResNet18/34 表现非常稳健，使得网络更容易训练，尤其适合在医学图像中提取复杂特征。

5. 