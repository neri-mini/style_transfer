"""赛博画师 —— 附加题 2：多风格融合

【思路】
单风格时风格损失只有一项，融合两种风格就是把风格损失拆成两项、
各自配一个权重：

    L_style = beta1 * L_style_1 + beta2 * L_style_2

为了让"权重"有意义（两项加起来的总力度可比），把总力度固定为
BETA_TOTAL，再用一个占比 w 在两种风格之间分配：

    beta1 = BETA_TOTAL * w          # 风格1（二次元）的权重
    beta2 = BETA_TOTAL * (1 - w)    # 风格2（夕阳）的权
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
PICTURES = os.path.join(PROJECT, "pictures_multi")
OUTPUT = os.path.join(PROJECT, "output_multi")

IMG_SIZE = 768
NUM_ITERS = 100
LBFGS_LR = 1.0
LBFGS_MAX_ITER = 20
LBFGS_HISTORY = 10

# 两种风格各占多少
WEIGHT_RATIOS = [0.25, 0.50, 0.75]

CONTENT_WEIGHT = 1.0        # alpha
BETA_TOTAL = 1e4            # beta
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
    """tensor变PIL图片（先 clamp，防止越界值被截断成彩色噪点）"""
    return T.ToPILImage()(tensor.detach().clamp(0, 1).squeeze(0).cpu())


# part2:读图
def load_image(path, device, size=IMG_SIZE):
    """读一张图,返回归一化张量 (1, 3, size, size)"""
    I = Image.open(path).convert("RGB")
    tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])
    return to_norm(tf(I).unsqueeze(0).to(device))


def list_images(folder):
    """列出目录下所有图片的完整路径（按文件名排序）"""
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(IMAGE_EXTS)]


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


# part4:数学公式
def gram_matrix(tensor):
    """Gram 矩阵"""
    b, c, h, w = tensor.shape
    tensor = tensor.view(b, c, h * w)
    return torch.bmm(tensor, tensor.transpose(1, 2)) / (b * c * h * w)


def extract_style_grams(vgg, style_tensor):
    """把一张风格图在各风格层的 Gram 矩阵预先算好（这些值全程不变）"""
    feats = get_features(vgg, style_tensor)
    return {l: gram_matrix(feats[l]) for l in STYLE_LAYERS}


def loss_calculate(feats, content_target, style_grams_list, gen_normed, weights):
    """计算内容损失 + 【两项】风格损失 + TV 损失，以及加权总损失"""
    alpha, beta1, beta2, gamma = weights
    c = sum(F.mse_loss(feats[i], content_target[i]) for i in CONTENT_LAYERS)
    gen_grams = {l: gram_matrix(feats[l]) for l in STYLE_LAYERS}
    s1 = sum(F.mse_loss(gen_grams[l], style_grams_list[0][l]) for l in STYLE_LAYERS)
    s2 = sum(F.mse_loss(gen_grams[l], style_grams_list[1][l]) for l in STYLE_LAYERS)

    t = (torch.mean(torch.abs(gen_normed[:, :, :-1, :] - gen_normed[:, :, 1:, :]))
         + torch.mean(torch.abs(gen_normed[:, :, :, :-1] - gen_normed[:, :, :, 1:])))

    total = alpha * c + beta1 * s1 + beta2 * s2 + gamma * t
    return c, s1, s2, t, total


# part5:核心流程
def transfer(vgg, content, style_grams_list, weights,
             patience=5, min_delta=1e-4):
    """以内容图为起点，用 LBFGS 优化 NUM_ITERS 步像素。

    返回 (生成图, 最优总损失, 实际步数, 损失记录)
    损失记录每行: (步数, 内容, 风格1, 风格2, tv, 总损失)
    """
    content_target = get_features(vgg, content)

    gen = to_tensor(content).clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([gen], lr=LBFGS_LR, max_iter=LBFGS_MAX_ITER,
                                  history_size=LBFGS_HISTORY,
                                  line_search_fn="strong_wolfe")

    def closure():
        with torch.no_grad():
            gen.clamp_(0, 1)             
        optimizer.zero_grad()
        x = to_norm(gen)                 
        feats = get_features(vgg, x, track=True)
        c, s1, s2, t, total = loss_calculate(
            feats, content_target, style_grams_list, x, weights)
        loss = total    
        loss.backward()
        return loss

    with torch.no_grad():
        x0 = to_norm(gen)
        c0, s10, s20, t0, total0 = loss_calculate(
            get_features(vgg, x0), content_target, style_grams_list, x0, weights)
    record = [(0, c0.item(), s10.item(), s20.item(), t0.item(), total0.item())]

    best_gen, best_total = gen.detach().clone(), total0.item()
    prev_total, stale, done = best_total, 0, 0

    for step in range(1, NUM_ITERS + 1):
        optimizer.step(closure)
        with torch.no_grad():
            gen.clamp_(0, 1)
        with torch.no_grad():
            x = to_norm(gen)
            values = loss_calculate(
                get_features(vgg, x), content_target, style_grams_list, x, weights)
        c, s1, s2, t, total = (value.item() for value in values)
        record.append((step, c, s1, s2, t, total))
        done = step

        if total < best_total:
            best_total = total
            best_gen = gen.detach().clone()

        improve = (prev_total - total) / (abs(prev_total) + 1e-8)
        stale = stale + 1 if improve < min_delta else 0
        prev_total = total
        if stale >= patience:
            break

    return best_gen, best_total, done, record


def save_loss_csv(record, path, weights):
    """把每步的损失写成 CSV（含加权值，方便看两种风格各占多少）"""
    alpha, beta1, beta2, gamma = weights
    with open(path, "w", encoding="utf_8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "content_loss", "style1_loss", "style2_loss", "tv_loss",
                    "weighted_content", "weighted_style1", "weighted_style2",
                    "weighted_tv", "total"])
        for step, c, s1, s2, t, total in record:
            w.writerow([step, c, s1, s2, t,
                        alpha * c, beta1 * s1, beta2 * s2, gamma * t, total])


# part6:主流程
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备 {device}，尺寸 {IMG_SIZE}")
    content_paths = list_images(os.path.join(PICTURES, "content_image"))
    style_paths = list_images(os.path.join(PICTURES, "style_image"))
    print(f"内容图: {os.path.basename(content_paths[0])}")
    for k, p in enumerate(style_paths, 1):
        print(f"风格{k}: {os.path.basename(p)}")
    print()

    vgg = load_vgg(device)
    content = load_image(content_paths[0], device)
    styles = [load_image(p, device) for p in style_paths]
    style_grams_list = [extract_style_grams(vgg, s) for s in styles]

    summary = []
    for w_ratio in WEIGHT_RATIOS:
        beta1 = BETA_TOTAL * w_ratio
        beta2 = BETA_TOTAL * (1.0 - w_ratio)
        weights = (CONTENT_WEIGHT, beta1, beta2, TV_WEIGHT)

        out_dir = os.path.join(OUTPUT, f"w_{w_ratio:.2f}")
        os.makedirs(out_dir, exist_ok=True)

        t0 = time.time()
        gen, best, done, record = transfer(vgg, content, style_grams_list, weights)
        to_PIL_image(gen).save(os.path.join(out_dir, "result.png"))
        save_loss_csv(record, os.path.join(out_dir, "loss.csv"), weights)

        _, c_l, s1_l, s2_l, _, total = min(record, key=lambda row: row[-1])
        wc, w1, w2 = CONTENT_WEIGHT * c_l, beta1 * s1_l, beta2 * s2_l
        style_sum = w1 + w2
        summary.append([f"{w_ratio:.2f}", f"{beta1:.4g}", f"{beta2:.4g}",
                        f"{c_l:.6f}", f"{s1_l:.10f}", f"{s2_l:.10f}",
                        f"{wc:.6f}", f"{w1:.6f}", f"{w2:.6f}",
                        f"{w1/style_sum:.4f}", f"{total:.6f}", done])
        print(f"  w={w_ratio:.2f}  beta1={beta1:.4g} beta2={beta2:.4g}  "
              f"内容 {c_l:7.4f}  风格1 {s1_l:.3e}  风格2 {s2_l:.3e}  "
              f"总损失 {total:7.4f}  {done}步  {time.time()-t0:.0f}s")
        print(f"-> output_multi/w_{w_ratio:.2f}/result.png")

    os.makedirs(OUTPUT, exist_ok=True)
    path = os.path.join(OUTPUT, "summary.csv")
    with open(path, "w", encoding="utf_8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["w_ratio", "beta1", "beta2", "content_loss",
                    "style1_loss", "style2_loss", "weighted_content",
                    "weighted_style1", "weighted_style2",
                    "style1_share_of_style", "total", "steps"])
        w.writerows(summary)
    print("全部完成")


if __name__ == "__main__":
    main()
