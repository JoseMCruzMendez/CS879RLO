import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from captum.attr import LayerGradCam, LayerAttribution
import matplotlib.pyplot as plt
import numpy as np

# =============================
# 1. 加载 MNIST 测试集
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # 转成3通道
    transforms.ToTensor()
])

test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# =============================
# 2. 加载模型
# =============================
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # MNIST 有 10 类
model.eval()

# =============================
# 3. 找一个正确分类 & 一个错误分类的样本
# =============================
correct_sample, correct_label, correct_pred = None, None, None
misclassified_sample, misclassified_label, misclassified_pred = None, None, None

for img, label in test_loader:
    output = model(img)
    pred = output.argmax(1).item()
    if pred == label.item() and correct_sample is None:
        correct_sample, correct_label, correct_pred = img, label.item(), pred
    if pred != label.item() and misclassified_sample is None:
        misclassified_sample, misclassified_label, misclassified_pred = img, label.item(), pred
    if correct_sample is not None and misclassified_sample is not None:
        break

print(f"✅ Correct example: True={correct_label}, Pred={correct_pred}")
print(f"❌ Misclassified example: True={misclassified_label}, Pred={misclassified_pred}")

# =============================
# 4. 定义 Grad-CAM 函数
# =============================
def gradcam_results(model, img, pred, layers):
    results = []
    for layer in layers:
        layer_gc = LayerGradCam(model, layer)
        attributions = layer_gc.attribute(img, target=pred)
        upsampled_attr = LayerAttribution.interpolate(attributions, img.shape[2:])
        attr = upsampled_attr.squeeze().detach().numpy()
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        results.append(attr)
    logits = model(img).squeeze().detach().numpy()
    return results, logits

# =============================
# 5. 准备可视化层
# =============================
layers = [
    model.layer1[0].conv1,
    model.layer2[0].conv1,
    model.layer3[0].conv1,
    model.layer4[1].conv2,
]

# =============================
# 6. 获取 Grad-CAM 结果
# =============================
correct_attr, correct_logits = gradcam_results(model, correct_sample, correct_pred, layers)
mis_attr, mis_logits = gradcam_results(model, misclassified_sample, misclassified_pred, layers)

# =============================
# 7. 绘图：保存到一张图
# =============================
fig, axes = plt.subplots(2, len(layers) + 2, figsize=(25, 10))

# 正确分类样本
axes[0, 0].imshow(correct_sample.squeeze().permute(1, 2, 0).numpy(), cmap="gray")
axes[0, 0].set_title(f"✅ Correct\nTrue={correct_label}, Pred={correct_pred}")
axes[0, 0].axis("off")

for i, attr in enumerate(correct_attr):
    axes[0, i + 1].imshow(correct_sample.squeeze().permute(1, 2, 0).numpy(), cmap="gray")
    axes[0, i + 1].imshow(attr, cmap="jet", alpha=0.5)
    axes[0, i + 1].set_title(f"Layer {i+1}")
    axes[0, i + 1].axis("off")

axes[0, -1].bar(range(10), correct_logits)
axes[0, -1].set_title("Logits")
axes[0, -1].set_xticks(range(10))

# 错误分类样本
axes[1, 0].imshow(misclassified_sample.squeeze().permute(1, 2, 0).numpy(), cmap="gray")
axes[1, 0].set_title(f"❌ Misclassified\nTrue={misclassified_label}, Pred={misclassified_pred}")
axes[1, 0].axis("off")

for i, attr in enumerate(mis_attr):
    axes[1, i + 1].imshow(misclassified_sample.squeeze().permute(1, 2, 0).numpy(), cmap="gray")
    axes[1, i + 1].imshow(attr, cmap="jet", alpha=0.5)
    axes[1, i + 1].set_title(f"Layer {i+1}")
    axes[1, i + 1].axis("off")

axes[1, -1].bar(range(10), mis_logits)
axes[1, -1].set_title("Logits")
axes[1, -1].set_xticks(range(10))

plt.tight_layout()
plt.savefig("mnist_gradcam_comparison.png")
plt.show()
