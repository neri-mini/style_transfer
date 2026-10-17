"""赛博画师 —— 附加题 1：风格强度控制

【思路】
风格强度的"旋钮"其实就是风格损失权重 beta。
直接暴露 beta 不好理解（它的取值是 1e2~1e5 这种量级），所以定义一个
0~1 的直观参数 STYLE_STRENGTH（风格强度），按【对数间隔】映射到 beta：

    STYLE_WEIGHT = 10 ** (2 + 2 * STYLE_STRENGTH)

    strength = 0.00  ->  beta = 1e2   （几乎无风格）
    strength = 0.25  ->  beta = 3.16e2
    strength = 0.50  ->  beta = 1e3
    strength = 0.75  ->  beta = 3.16e3
    strength = 1.00  ->  beta = 1e4   （本项目的默认值）

用对数而不是线性，是因为损失权重的"有效差异"是倍数关系：
beta 从 1e2 到 1e3 的变化，和从 1e3 到 1e4 是一样的量级跳跃。

【输出】
    output_ctrl/<set>/strength_<s>/result.png   各强度下的生成图
    output_ctrl/<set>/strength_<s>/loss.csv     该强度的逐步损失
    output_ctrl/summary.csv                     所有强度横向对比（题目要求的"解释差异"靠它）
"""
# 库的引入
import csv
import os
import time
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# 路径
SRC = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(SRC)
PICTURES = os.path.join(PROJECT, "pictures")        # 与基础版共用输入
OUTPUT = os.path.join(PROJECT, "output_ctrl")       # 输出单独一个目录

# 超参数
IMG_SIZE = 768
NUM_ITERS = 100
LBFGS_LR = 1.0
LBFGS_MAX_ITER = 20
LBFGS_HISTORY = 10

# 风格强度档位
STRENGTH_LEVELS = [0.25, 0.50, 0.75, 1.00]

# 三个损失的权重
CONTENT_WEIGHT = 1.0        # alpha，固定不动（作为基准）
TV_WEIGHT = 1e-4            # gamma，固定不动（只研究风格强度的影响）
STYLE_WEIGHT = 1e4          # 运行时会被 strength_to_weight() 覆盖

# 层索引
CONTENT_LAYERS = [22]                       # relu4_2
STYLE_LAYERS = [1, 6, 11, 20, 29]           # relu1_1 ~ relu5_1

# 归一化参数
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
    return T.ToPILImage()(tensor.detach().clamp(0, 1).squeeze(0).cpu())

# part2:读图和扫描
def load_image(path, device, size=IMG_SIZE):
    """读一张图,返回归一化张量 (1, 3, size, size)"""
    I = Image.open(path).convert("RGB")
    tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])
    return to_norm(tf(I).unsqueeze(0).to(device))


def discover_sets(pictures_path=PICTURES):
    """返回 [(set名, 内容图路径, 风格图路径)]"""
    found = []
    if not os.path.isdir(pictures_path):
        return found
    for name in sorted(os.listdir(pictures_path)):
        set_dir = os.path.join(pictures_path, name)
        paths = []
        for dir_name in ["content_image", "style_image"]:
            folder = os.path.join(set_dir, dir_name)
            pictures = sorted(p for p in os.listdir(folder)
                              if p.lower().endswith(IMAGE_EXTS))
            paths.append(os.path.join(folder, pictures[0]))
        found.append((name, paths[0], paths[1]))
    return found

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


# part4:数学公式！！！(骇死我了)
def gram_matrix(tensor):
    """Gram 矩阵"""
    b, c, h, w = tensor.shape
    tensor = tensor.view(b, c, h * w)
    return torch.bmm(tensor, tensor.transpose(1, 2)) / (b * c * h * w)


def loss_calculate(feats, content_target, style_target, gen_normed):
    """计算三个损失和加权总损失"""
    c = sum(F.mse_loss(feats[i], content_target[i]) for i in CONTENT_LAYERS)
    s = sum(F.mse_loss(gram_matrix(feats[i]), gram_matrix(style_target[i]))
            for i in STYLE_LAYERS)
    t = (torch.mean(torch.abs(gen_normed[:, :, :-1, :] - gen_normed[:, :, 1:, :]))
         + torch.mean(torch.abs(gen_normed[:, :, :, :-1] - gen_normed[:, :, :, 1:])))
    return c, s, t, CONTENT_WEIGHT * c + STYLE_WEIGHT * s + TV_WEIGHT * t

# part5:存储与计算
def save_loss_csv(record, path):
    with open(path, "w", encoding="utf_8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "content_loss", "style_loss", "tv_loss",
                    "weighted_content", "weighted_style", "weighted_tv", "total"])
        for step, c, s, t, total in record:
            w.writerow([step, c, s, t, c * CONTENT_WEIGHT, s * STYLE_WEIGHT,
                        t * TV_WEIGHT, total])

def strength_to_weight(strength):
    """把 0~1 的风格强度映射成风格损失权重 beta（对数间隔 1e2 ~ 1e4）"""
    return 10.0 ** (2.0 + 2.0 * strength)                        

# part6:核心流程
def transfer(vgg, content, style, patience=5, min_delta=1e-4):
    """以内容图为起点，用 LBFGS 优化 NUM_ITERS 步像素
    返回 生成图, 最优总损失, 实际步数, 损失记录
    """
    gen = to_tensor(content).clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([gen], lr=LBFGS_LR, max_iter=LBFGS_MAX_ITER,
                                  history_size=LBFGS_HISTORY,
                                  line_search_fn="strong_wolfe")
    
    content_target = get_features(vgg, content)
    style_target = get_features(vgg, style)

    def closure():
        with torch.no_grad():
            gen.clamp_(0, 1)
        optimizer.zero_grad()
        x = to_norm(gen)
        feats = get_features(vgg, x, track=True)
        c, s, t, total = loss_calculate(feats, content_target, style_target, x)
        loss = total
        loss.backward()
        return loss
    with torch.no_grad():
        x0 = to_norm(gen)
        c0, s0, t0, total0 = loss_calculate(
            get_features(vgg, x0), content_target, style_target, x0)
    record = [(0, c0.item(), s0.item(), t0.item(), total0.item())]

    best_gen, best_total = gen.detach().clone(), total0.item()
    prev_total, stale, done = best_total, 0, 0

    for step in range(1, NUM_ITERS + 1):
        optimizer.step(closure)
        with torch.no_grad():
            gen.clamp_(0, 1)
        with torch.no_grad():
            x = to_norm(gen)
            values = loss_calculate(
                get_features(vgg, x), content_target, style_target, x)
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

    return best_gen, best_total, done, record

# part6:主流程
def main():
    global STYLE_WEIGHT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备 {device}，尺寸 {IMG_SIZE}")
    sets = discover_sets()    
    vgg = load_vgg(device)
    summary = []
    os.makedirs(OUTPUT, exist_ok=True)
    for set_name,content_path,style_path in sets:
        print(f"对{set_name}进行生图")
        content = load_image(content_path, device)
        style = load_image(style_path, device)
        output_set_path = os.path.join(OUTPUT, set_name)
        os.makedirs(output_set_path, exist_ok=True)
        for s in STRENGTH_LEVELS:
            STYLE_WEIGHT = strength_to_weight(s)
            output_set_strength = os.path.join(output_set_path, f"strength_{s:.2f}")
            os.makedirs(output_set_strength,exist_ok=True)
            t0 = time.time()
            gen, best, done, record = transfer(vgg, content, style)
            to_PIL_image(gen).save(os.path.join(output_set_strength, "result.png"))
            save_loss_csv(record, os.path.join(output_set_strength, "loss.csv"))

            _, c_l, s_l, t_l, total = min(record, key=lambda row: row[-1])
            wc = CONTENT_WEIGHT * c_l
            ws = STYLE_WEIGHT * s_l
            share = ws / (wc + ws + TV_WEIGHT * t_l)
            summary.append([set_name, f"{s:.2f}", f"{STYLE_WEIGHT:.4g}",
                            f"{c_l:.6f}", f"{s_l:.10f}", f"{wc:.6f}",
                            f"{ws:.6f}", f"{share:.4f}", f"{total:.6f}", done])
            print(f"    strength {s:.2f} (beta={STYLE_WEIGHT:.4g})  "
                  f"内容 {c_l:7.4f}  风格 {s_l:.3e}  风格占比 {share*100:5.1f}%  "
                  f"总损失 {total:7.4f}  {done}步  {time.time()-t0:.0f}s")
    path = os.path.join(OUTPUT, "summary.csv")
    with open(path, "w", encoding="utf_8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "strength", "beta", "content_loss", "style_loss",
                    "weighted_content", "weighted_style", "style_share",
                    "total", "steps"])
        w.writerows(summary)
    print(f"汇总表 -> output_ctrl/summary.csv")
    print("全部完成")


if __name__ == "__main__":
    main()
