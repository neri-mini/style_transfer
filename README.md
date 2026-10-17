# 赛博画师 —— 风格迁移（Style Transfer）

用预训练 **VGG19** 当特征提取器，把「风格图」的画风迁移到「内容图」上

优化器用 **LBFGS**（二阶 + 强 Wolfe 线搜索）

---

## 目录结构

```
style_transfer/
├── src/                                   代码（4 个脚本）
│   ├── style_transfer_LBFGS_nomask.py     基础任务：单风格迁移
│   ├── style_transfer_ctrl.py             附加题 1：风格强度控制
│   ├── style_transfer_multi.py            附加题 2：多风格融合
│   └── style_transfer_mask.py             附加题 3：局部风格迁移
│
├── pictures/                              基础任务的输入（3 个 set）
│   ├── set1/{content_image, style_image}/     星空（油画）
│   ├── set2/{content_image, style_image}/     二次元动画
│   └── set3/{content_image, style_image}/     夕阳
├── output/                                基础任务的输出
│   └── setN/{result.png, loss.csv, compare/}
│
├── pictures_multi/                        附加题 2 的输入（1 内容 + 2 风格）
│   ├── content_image/绣球花.jpg
│   └── style_image/{01_二次元.jpg, 02_夕阳.jpeg}
├── output_multi/
│   ├── w_0.25/  w_0.50/  w_0.75/          三组不同权重
│   └── summary.csv
│
├── pictures_mask/                         附加题 3 的输入（mask）
│   └── set1/{content_image, style_image, mask/mask.png}
├── output_mask/
│   └── set1/{result.png, loss.csv}
│
├── docs/赛博画师.pdf                        题目解答
└── style_to.txt                            三套 set 对应的风格说明
```

---

## 环境

| 项 | 版本 |
|---|---|
| Python | 3.11 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| Pillow | 随 torchvision |

```bash
pip install -r requirements.txt
```

**首次运行会自动下载 VGG19 预训练权重（约 548 MB）**，存到 `~/.cache/torch/`，
不会提交到仓库

---

## 运行

所有脚本都**不需要命令行参数**，想改参数直接改文件开头的配置区

### 基础任务：单风格迁移

```bash
cd src
python style_transfer_LBFGS_nomask.py
```

扫描 `pictures/` 下的所有 set（每个 set 需要 `content_image/` 和 `style_image/` 各一张图）

逐个生成到 `output/setN/`

| 输出 | 内容 |
|---|---|
| `result.png` | 最终生成图 |
| `loss.csv` | 每步的内容/风格/TV 损失及加权值 |
| `compare/` | `iter_0000_content.png`（起点）+ 对数间隔的过程图 |

### 附加题 1：风格强度控制

```bash
python style_transfer_ctrl.py
```

用 `STRENGTH_LEVELS` 扫不同的风格强度。强度 `s ∈ [0,1]` 按**对数间隔**映射到权重：

$$
\beta = 10^{\,2 + 2s}
$$

输出 `output_ctrl/setN/strength_<s>/{result.png, loss.csv}` + `summary.csv`

### 附加题 2：多风格融合

```bash
python style_transfer_multi.py
```

同时使用**两张风格图**，风格损失拆成两项：

$$
L_{style} = \beta_1 L_{style}^{(1)} + \beta_2 L_{style}^{(2)},
\qquad
\beta_1 = \beta_{total}\cdot w,\quad \beta_2 = \beta_{total}\cdot(1-w)
$$

用 `WEIGHT_RATIOS` 扫三组权重

输出 `output_multi/w_<w>/{result.png, loss.csv}` + `summary.csv`

### 附加题 3：局部风格迁移

```bash
python style_transfer_mask.py
```

只让 mask 覆盖的区域（花朵）被风格化，**背景逐像素保持不变**

- 内容损失只在 mask 内算

- 风格损失：用 mask 对 Gram 的空间统计作线性加权，归一化用 mask 有效面积

- 输出时硬合成：`result = gen·mask + 原图·(1-mask)`

  

## Correctness and regression tests

- Loss CSV rows are evaluated on the accepted, clamped image after each LBFGS step.
  Early stopping and best-image selection use those same values; the initial image
  is also a candidate. Summary tables describe the saved best image.
- Masked transfer composites the fixed content background before every feature
  extraction and TV evaluation. Pixels where the input mask is zero receive no
  optimization gradient. Background preservation is relative to the resized,
  center-cropped content image used by the model.
- Feature-resolution masks may contain fractional values after average pooling.
  Masked Gram matrices use these values as linear area weights.

Run from the repository root after installing the requirements:

```bash
python -m unittest discover -s tests -v
```

The tests use a small CPU feature extractor, without downloading VGG weights.
They cover all four transfer modes, rejected line-search trials, fixed-background
gradients, soft-mask Gram weighting, and zero-iteration behavior.
