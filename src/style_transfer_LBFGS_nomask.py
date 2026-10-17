"""赛博画师风格迁移python文件
用预训练 VGG19 当特征提取器
优化器用 LBFGS（二阶 + 线搜索）
没用mask
原图放pictures文件夹下，风格图和内容图放同一set的style_image和content_image下
本文件实现对内的图扫描并生成风格迁移图，放在对应output/set下
目录约定（自动扫描，加新 set 只建文件夹放图，不用改代码）：
    pictures/set1/content_image/内容图
    pictures/set1/style_image/风格图
    output/set1/result.png              最终生成图
    output/set1/loss.csv                每步损失
    output/set1/compare/iter_*.png      过程图（对数间隔，起点的 iter_0000 是原图）
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
PICTURES = os.path.join(PROJECT, "pictures")
OUTPUT = os.path.join(PROJECT, "output")

# 超参数
IMG_SIZE = 768          # 生成图边长
NUM_ITERS = 100         # LBFGS 外层步数（每步内部还会做多次子迭代）
NUM_SNAPSHOTS = 8       # 过程图张数（对数间隔，0 = 不存）
LBFGS_LR = 1.0          # 线搜索的初始尺度
LBFGS_MAX_ITER = 20     # 每次 step 内部最多几次子迭代
LBFGS_HISTORY = 10      # 用最近多少步梯度近似曲率

# 三个损失的权重
CONTENT_WEIGHT = 1.0    # alpha（大了风格弱白训练，小了原图结构丢失）
STYLE_WEIGHT = 1e4      # beta（大了风格吃掉原图，严重扭曲，小了风格弱了）
TV_WEIGHT = 1e-4        # gamma（控制平滑程度，大了变糊，小了噪点多）

# 层索引
CONTENT_LAYERS = [22]                      # relu4_2：取内容
STYLE_LAYERS = [1, 6, 11, 20, 29]          # relu1_1~relu5_1：取风格

# 归一化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif")

# part1:归一化及其逆变换
def to_norm(tensor):
    """对张量归一化"""
    mean = torch.tensor(IMAGENET_MEAN,device=tensor.device).view(1,3,1,1)
    std = torch.tensor(IMAGENET_STD,device=tensor.device).view(1,3,1,1)
    return (tensor - mean) / std

def to_tensor(tensor_normed):
    """逆归一化"""
    mean = torch.tensor(IMAGENET_MEAN,device=tensor_normed.device).view(1,3,1,1)
    std = torch.tensor(IMAGENET_STD,device=tensor_normed.device).view(1,3,1,1)
    return ((tensor_normed * std) + mean).clamp(0,1)

def to_PIL_image(tensor):
    """tensor变PIL图片"""
    return T.ToPILImage()(tensor.detach().clamp(0,1).squeeze(0).cpu())
    
# part2:读图和扫描
def load_image(path,device,size=IMG_SIZE):
    """读一张图,返回归一化张量 (1, 3, size, size)"""
    I = Image.open(path).convert("RGB")
    tf = T.Compose([T.Resize(size),T.CenterCrop(size),T.ToTensor()])
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
              folder = os.path.join(set_dir,dir_name)
              pictures = sorted(p for p in os.listdir(folder) if p.lower().endswith(IMAGE_EXTS))
              paths.append(os.path.join(folder,pictures[0]))
          found.append((name,paths[0],paths[1]))
    return found

# part3:VGG19特征提取
def load_vgg(device):
    """加载预训练VGG19的卷积部分,先冻结所有参数"""
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    return vgg.to(device).eval().requires_grad_(False)


def get_features(vgg,x,track=False):
    """让x穿过vgg特征提取部分,返回{层索引:特征图张量}"""
    feats = {}
    with torch.set_grad_enabled(track):
        for i,layer in enumerate(vgg):
            x = layer(x)
            if i in CONTENT_LAYERS or i in STYLE_LAYERS:
               feats[i] = x
            if i >= max(CONTENT_LAYERS + STYLE_LAYERS):
                break  
    return feats                  

# part4:数学公式！！！(骇死我了)
def gram_matrix(tensor):
    """Gram 矩阵"""
    b,c,h,w = tensor.shape
    tensor = tensor.view(b,c,h * w)
    return torch.bmm(tensor,tensor.transpose(1,2)) / (b * c * h * w)

def loss_calculate(feats,content_target, style_target, gen_normed):
    """计算三个损失和加权总损失"""
    c = sum(F.mse_loss(feats[i], content_target[i]) for i in CONTENT_LAYERS)
    s = sum(F.mse_loss(gram_matrix(feats[i]) , gram_matrix(style_target[i])) for i in STYLE_LAYERS)
    t = (torch.mean(torch.abs((gen_normed[:,:,:-1,:] - gen_normed[:,:,1:,:]))) +  
         torch.mean(torch.abs((gen_normed[:,:,:,:-1] - gen_normed[:,:,:,1:]))))
    return c, s, t,  CONTENT_WEIGHT * c + STYLE_WEIGHT * s + TV_WEIGHT * t              

# part5:存储风格变迁图和csv
def get_snapshot_steps(iters=NUM_ITERS,count=NUM_SNAPSHOTS):
    """按对数间隔挑出存过程图的步数（前期密、后期疏）"""
    if count <= 0:
      return set()
    steps = {max(1,round(iters ** (i / count))) for i in range(1,count + 1)}
    return steps | {iters}

def save_loss_csv(record,path):
    with open(path,"w",encoding="utf_8",newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter","content_loss","style_loss","tv_loss",
                    "weighted_content", "weighted_style",
                    "weighted_tv", "total"])
        for step,c,s,t,total in record:
            w.writerow([step,c,s,t,c * CONTENT_WEIGHT,s * STYLE_WEIGHT,t * TV_WEIGHT,total])
    print(f"已把对应set的csv存入{path}")

# part6:核心流程
def transfer(vgg,content,style,snapshot_dir=None,snapshot_steps=None,
             patience=5,min_delta=1e-4):
    """干什么的：以内容图为起点，用 LBFGS 优化 NUM_ITERS 步像素
    输入 vgg,内容图张量，风格图张量，风格变迁文件夹路径，风格变迁要存储的步数
    返回 生成图张量, 最优总损失, 最优步数, 损失记录, 实际步数
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
        feats = get_features(vgg,x,track=True)
        c, s, t, total = loss_calculate(feats,content_target,style_target,x)
        loss = total
        loss.backward()
        return loss
    with torch.no_grad():
        x0 = to_norm(gen)
        c0,s0,t0,total0 = loss_calculate(get_features(vgg,x0),content_target,style_target,x0)
    record = [(0,c0.item(),s0.item(),t0.item(),total0.item())]
    best_gen, best_total, best_step = gen.detach().clone(), total0.item(), 0
    prev_total, stale, done = best_total, 0, 0
    
    for step in range(1,NUM_ITERS +1):
        optimizer.step(closure)
        with torch.no_grad():
            gen.clamp_(0, 1)          
        # A line search may evaluate rejected trial points. Measure the accepted image.
        with torch.no_grad():
            x = to_norm(gen)
            values = loss_calculate(
                get_features(vgg, x), content_target, style_target, x)
        c, s, t, total = (value.item() for value in values)
        record.append((step, c, s, t, total))
        done = step

        if total < best_total:
            best_total, best_step = total, step
            best_gen = gen.detach().clone()
        improve = (prev_total - total) / (abs(prev_total)+1e-8)
        stale = stale + 1 if improve < min_delta else 0
        prev_total = total

        print(f"第{step:6d}/{NUM_ITERS}步 内容损失 {c:8.4f} 风格损失 {s:.6f} tv损失 {t:.6f} 总损失{total:.9f}")

        if snapshot_dir and step in snapshot_steps:
            to_PIL_image(gen).save(os.path.join(snapshot_dir,f"iter_{step:04d}.png"))
        if stale >= patience:
            print(F"早停啦，当前{step}轮")
            break
    print(f"finished 实际跑了{done}轮，最优在第 {best_step} 步，总损失 {best_total:.4f}")
    return best_gen,best_step,best_total,record,done       

# 主流程
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备 {device} 尺寸 {IMG_SIZE}")
    print("训练启动！！！")

    sets = discover_sets()
    snapshot_steps = get_snapshot_steps()
    vgg = load_vgg(device)
    print(f"此次共有{len(sets)}个set")

    for set_name,content_path,style_path in sets:
        print(f"对{set_name}进行生图")
        t0 = time.time()
        content = load_image(content_path,device)
        style = load_image(style_path,device)
        output_set_path = os.path.join(OUTPUT,set_name)
        output_set_snapshot_dir = os.path.join(output_set_path,"compare")
        if snapshot_steps:
            os.makedirs(output_set_snapshot_dir,exist_ok=True)
            to_PIL_image(to_tensor(content)).save(os.path.join(output_set_snapshot_dir,"iter_0000_content.png"))

        gen, best_step, step, record, done = transfer(vgg, content, style, output_set_snapshot_dir, snapshot_steps)
        os.makedirs(output_set_path,exist_ok=True)
        to_PIL_image(gen).save(os.path.join(output_set_path, "result.png"))
        save_loss_csv(record, os.path.join(output_set_path, "loss.csv"))
        print(f"  {set_name}用时{time.time() - t0:.2f}秒")

    print("全部完成")

if __name__ == "__main__":
    main()
    




