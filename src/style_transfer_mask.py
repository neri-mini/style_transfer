"""赛博画师 —— 附加题 3：局部风格迁移
"""

import csv
import os
import time

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

SRC = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(SRC)
PICTURES = os.path.join(PROJECT, "pictures_mask")
OUTPUT = os.path.join(PROJECT, "output_mask")

IMG_SIZE = 768
NUM_ITERS = 100
LBFGS_LR = 1.0
LBFGS_MAX_ITER = 20
LBFGS_HISTORY = 10

SET_NAME = "set1"

# 三个损失的权重
CONTENT_WEIGHT = 1.0        # alpha
STYLE_WEIGHT = 1e4          # beta
TV_WEIGHT = 1e-4            # gamma

CONTENT_LAYERS = [22]                       # relu4_2
STYLE_LAYERS = [1, 6, 11, 20, 29]           # relu1_1 ~ relu5_1

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif")


# part1:归一化及其逆变换
def to_norm(tensor):
    """对张量归一化"""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def to_tensor(tensor_normed):
    """逆归一化"""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor_normed.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor_normed.device).view(1, 3, 1, 1)
    return ((tensor_normed * std) + mean).clamp(0, 1)


def to_PIL_image(tensor):
    """tensor变PIL图片"""
    x = tensor.detach().clamp(0, 1).squeeze(0).cpu()
    arr = torch.round(x * 255.0).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(arr.permute(1, 2, 0).numpy())


# part2:读图与 mask
def load_image(path, device, size=IMG_SIZE):
    """读一张图,返回归一化张量 (1, 3, size, size)"""
    I = Image.open(path).convert("RGB")
    tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])
    return to_norm(tf(I).unsqueeze(0).to(device))


def load_mask(path, device, size=IMG_SIZE):
    """读 mask 图，返回 (1, 1, size, size) 的张量，取值 0~1"""
    m = Image.open(path).convert("L")
    tf = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.NEAREST),
                    T.CenterCrop(size), T.ToTensor()])
    return tf(m).unsqueeze(0).to(device)


def first_image(folder):
    if not os.path.isdir(folder):
        return None
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))
    return os.path.join(folder, files[0]) if files else None


# part3:VGG19特征提取
def load_vgg(device):
    """加载预训练VGG19的卷积部分,冻结所有参数"""
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    return vgg.to(device).eval().requires_grad_(False)


def get_features(vgg, x, track=False):
    """让x穿过vgg，返回{层索引:特征图}"""
    feats = {}
    with torch.set_grad_enabled(track):
        for i, layer in enumerate(vgg):
            x = layer(x)
            if i in CONTENT_LAYERS or i in STYLE_LAYERS:
                feats[i] = x
            if i >= max(CONTENT_LAYERS + STYLE_LAYERS):
                break
    return feats


# part4:带 mask 的数学公式
def gram_matrix(tensor):
    """Gram 矩阵（不带 mask，用于风格图本身）"""
    b, c, h, w = tensor.shape
    tensor = tensor.view(b, c, h * w)
    return torch.bmm(tensor, tensor.transpose(1, 2)) / (b * c * h * w)


def gram_matrix_masked(tensor, mask):
    """只统计 mask 内的 Gram 矩阵"""
    b, c, h, w = tensor.shape
    # sqrt ensures soft masks weight each outer product once, not twice.
    t = (tensor * mask.sqrt()).view(b, c, h * w)
    area = mask.sum() * c + 1e-8
    return torch.bmm(t, t.transpose(1, 2)) / area


def loss_content_masked(feats, content_target, mask, layer):
    """只在 mask 内计算的平方误差（除以 mask 内元素数，保持量级不变）"""
    diff = feats[layer] - content_target[layer]
    c = feats[layer].shape[1]
    return (diff ** 2 * mask).sum() / (mask.sum() * c + 1e-8)


def loss_tv(x):
    """全变分：相邻像素差的绝对值（横向 + 纵向）"""
    return (torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
            + torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])))


def loss_calculate(feats, content_target, style_grams, gen_normed, masks):
    """四个部分：mask 内内容损失 + mask 内风格损失 + TV
    masks: {层索引: 该层尺寸的 mask}
    """
    c = sum(loss_content_masked(feats, content_target, masks[l], l)
            for l in CONTENT_LAYERS)

    s = sum(F.mse_loss(gram_matrix_masked(feats[l], masks[l]), style_grams[l])
            for l in STYLE_LAYERS)

    t = loss_tv(gen_normed)
    total = CONTENT_WEIGHT * c + STYLE_WEIGHT * s + TV_WEIGHT * t
    return c, s, t, total


# part5:保存数据
def save_loss_csv(record, path):
    with open(path, "w", encoding="utf_8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "content_loss", "style_loss", "tv_loss",
                    "weighted_content", "weighted_style", "weighted_tv", "total"])
        for step, c, s, t, total in record:
            w.writerow([step, c, s, t, c * CONTENT_WEIGHT, s * STYLE_WEIGHT,
                        t * TV_WEIGHT, total])
# part6:核心流程
def transfer(vgg, content, style, mask, patience=5, min_delta=1e-4):
    """以内容图为起点，只优化 mask 内的像素
    返回 (生成图[已合成背景], 最优总损失, 实际步数, 损失记录)
    """
    content_target = get_features(vgg, content)
    style_feats = get_features(vgg, style)
    style_grams = {l: gram_matrix(style_feats[l]) for l in STYLE_LAYERS}

    # 内容图的像素值，最后合成背景要用
    content_pixel = to_tensor(content).clone()

    gen = content_pixel.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([gen], lr=LBFGS_LR, max_iter=LBFGS_MAX_ITER,
                                  history_size=LBFGS_HISTORY,
                                  line_search_fn="strong_wolfe")

    # 每层的 mask 都缩放到该层的空间尺寸
    with torch.no_grad():
        probe = get_features(vgg, to_norm(gen))
    masks = {l: F.adaptive_avg_pool2d(mask, probe[l].shape[2:]) for l in probe}
    def closure():
        with torch.no_grad():
            gen.clamp_(0, 1)
        optimizer.zero_grad()
        x = to_norm(gen * mask + content_pixel * (1 - mask))
        feats = get_features(vgg, x, track=True)
        c, s, t, total = loss_calculate(feats, content_target, style_grams, x, masks)
        total.backward()
        return total
    with torch.no_grad():
        x0 = to_norm(gen)
        c0, s0, t0, total0 = loss_calculate(
            get_features(vgg, x0), content_target, style_grams, x0, masks)
    record = [(0, c0.item(), s0.item(), t0.item(), total0.item())]
    best_gen, best_total = gen.detach().clone(), total0.item()
    prev_total, stale, done = best_total, 0, 0
    for step in range(1, NUM_ITERS + 1):
        optimizer.step(closure)
        with torch.no_grad():
            gen.clamp_(0, 1)
        with torch.no_grad():
            x = to_norm(gen * mask + content_pixel * (1 - mask))
            values = loss_calculate(
                get_features(vgg, x), content_target, style_grams, x, masks)
        c, s, t, total = (value.item() for value in values)
        record.append((step, c, s, t, total))
        done = step

        if total < best_total:
            best_total = total
            best_gen = gen.detach().clone()
        improve = (prev_total - total) / (abs(prev_total) + 1e-8)
        stale = stale + 1 if improve < min_delta else 0
        prev_total = total
        if stale >= patience:
            break
    # 硬合成
    composited = best_gen * mask + content_pixel * (1 - mask)
    return composited, best_total, done, record

# part7:主流程
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备 {device}，尺寸 {IMG_SIZE}")
    set_dir = os.path.join(PICTURES, SET_NAME)
    content_path = first_image(os.path.join(set_dir, "content_image"))
    style_path = first_image(os.path.join(set_dir, "style_image"))
    mask_path = os.path.join(set_dir, "mask", "mask.png")
    content = load_image(content_path, device)
    style = load_image(style_path, device)
    mask = load_mask(mask_path, device)
    vgg = load_vgg(device)
    t0 = time.time()
    gen, best, done, record = transfer(vgg, content, style, mask)
    out_dir = os.path.join(OUTPUT, SET_NAME)
    os.makedirs(out_dir, exist_ok=True)
    to_PIL_image(gen).save(os.path.join(out_dir, "result.png"))
    save_loss_csv(record, os.path.join(out_dir, "loss.csv"))

    _, c_l, s_l, t_l, total_l = min(record, key=lambda row: row[-1])
    print(f"  完成 {done} 步，内容 {c_l:.4f}  风格 {s_l:.3e}"
            f"tv {t_l:.5f}  总损失 {total_l:.4f}")
    print(f"  用时 {time.time() - t0:.0f} 秒")
    print(f"  保存至output_mask/{SET_NAME}/result.png")
    print("   全部完成")


if __name__ == "__main__":
    main()
