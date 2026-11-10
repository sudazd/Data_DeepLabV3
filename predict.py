"""
用训练好的权重做预测。

--split val / train（有标签）：
  保存可视化图并计算指标。每张输出图由 4 个面板横向拼成：
  [原图] [真值叠加(红)] [预测叠加(红)] [误差图：绿=预测正确的滑坡，红=误检(FP)，蓝=漏检(FN)]
  同时输出 per_image_iou.csv 和 <split>_metrics_recomputed.json。

--split test（Lushan.tif，无标签）：
  不计算任何精度，只做滑窗推理，输出
  Lushan_pred.png（0/255 二值图）、Lushan_prob.png（滑坡概率灰度图）、
  Lushan_pred.tif（若安装了 rasterio，则带原影像的坐标系，可直接叠加到 GIS）。

用法（--size 和三个 --no_* 开关必须与训练时一致）：
  python predict.py --ckpt runs\\full_256_s0\\best.pth --size 256 --split val  --out_dir Prediction\\val
  python predict.py --ckpt runs\\full_256_s0\\best.pth --size 256 --split test --out_dir Prediction\\test
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import LandslideDeepLab
from train import list_pairs, SegDS, Meter, MEAN, STD, DEFAULT_ROOT


def overlay(img, m, color=(255, 0, 0), a=0.5):
    out = img.astype(np.float32).copy()
    out[m == 1] = out[m == 1] * (1 - a) + np.array(color, np.float32) * a
    return out.astype(np.uint8)


# ------------------------------ 大影像读取 ------------------------------
def read_rgb(path):
    """
    读取影像为 uint8 (H,W,3)。优先用 rasterio（取前 3 个波段，非 8 位的按 2%-98% 拉伸），
    未安装则退回 PIL。返回 (数组, 地理信息字典或 None)。
    """
    Image.MAX_IMAGE_PIXELS = None
    try:
        import rasterio
    except ImportError:
        return np.asarray(Image.open(path).convert("RGB")), None

    with rasterio.open(path) as ds:
        n = min(ds.count, 3)
        arr = ds.read(list(range(1, n + 1)))  # (n,H,W)
        meta = {"crs": ds.crs, "transform": ds.transform}
    if n == 1:
        arr = np.repeat(arr, 3, axis=0)
    arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        out = np.empty(arr.shape, np.uint8)
        for c in range(3):
            band = arr[..., c].astype(np.float32)
            sub = band[::10, ::10]
            lo, hi = np.percentile(sub, 2), np.percentile(sub, 98)
            out[..., c] = np.clip((band - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        arr = out
    return arr, meta


def padded_len(L, tile, stride):
    """补边后的长度：>= tile，且 (长度-tile) 能被 stride 整除。"""
    if L <= tile:
        return tile
    return tile + int(np.ceil((L - tile) / stride)) * stride


@torch.no_grad()
def predict_large(model, img, dev, tile=256, overlap=32, bs=8):
    """对 uint8 (H,W,3) 大图滑窗推理，重叠区概率取平均，返回滑坡概率 (H,W) float32。"""
    model.eval()
    H, W, _ = img.shape
    stride = tile - overlap
    Hp, Wp = padded_len(H, tile, stride), padded_len(W, tile, stride)
    ph, pw = Hp - H, Wp - W
    mode = "reflect" if (ph < H and pw < W) else "edge"
    img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode=mode)

    prob = np.zeros((Hp, Wp), np.float32)
    cnt = np.zeros((Hp, Wp), np.float32)
    coords = [(y, x) for y in range(0, Hp - tile + 1, stride) for x in range(0, Wp - tile + 1, stride)]
    for i in range(0, len(coords), bs):
        batch = coords[i:i + bs]
        t = np.stack([((img[y:y + tile, x:x + tile].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
                      for y, x in batch])
        p = F.softmax(model(torch.from_numpy(t).to(dev)), 1)[:, 1].cpu().numpy()
        for (y, x), pi in zip(batch, p):
            prob[y:y + tile, x:x + tile] += pi
            cnt[y:y + tile, x:x + tile] += 1
        if (i // bs) % 20 == 0:
            print(f"  窗口 {min(i + bs, len(coords))}/{len(coords)}", flush=True)
    return (prob / np.maximum(cnt, 1))[:H, :W]


def run_test(args, model, dev):
    """Lushan.tif 无标签：只推理，不算精度。"""
    img_path = Path(args.data_root) / args.pred_img
    print("读取:", img_path)
    img, meta = read_rgb(str(img_path))
    H, W, _ = img.shape
    print(f"影像尺寸: {W} x {H}")

    if args.scale != 1.0:  # 把影像缩放到与训练数据相近的地物尺度再推理
        sw, sh = max(int(round(W * args.scale)), 1), max(int(round(H * args.scale)), 1)
        small = np.asarray(Image.fromarray(img).resize((sw, sh), Image.BILINEAR))
    else:
        small = img

    prob = predict_large(model, small, dev, args.size, args.overlap, args.bs)
    if prob.shape != (H, W):  # 概率图还原到原影像尺寸
        prob = np.asarray(Image.fromarray(prob, mode="F").resize((W, H), Image.BILINEAR))

    # 根据输入 TIFF 自动获取文件名（不含扩展名）
    stem = Path(img_path).stem

    mask = (prob > args.thr).astype(np.uint8)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 保存滑坡二值预测 PNG
    Image.fromarray(mask * 255).save(
        os.path.join(args.out_dir, f"{stem}_pred.png")
    )

    # 2. 保存滑坡概率 PNG
    Image.fromarray((prob * 255).astype(np.uint8)).save(
        os.path.join(args.out_dir, f"{stem}_prob.png")
    )

    # 3. 保存带地理坐标的预测 TIFF
    if meta is not None:
        import rasterio

        tif_path = os.path.join(
            args.out_dir,
            f"{stem}_pred.tif"
        )

        with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=H,
                width=W,
                count=1,
                dtype="uint8",
                crs=meta["crs"],
                transform=meta["transform"],
                compress="lzw"
        ) as dst:
            dst.write(mask, 1)

    print(f"预测完成：{stem}")
    print(f"滑坡像元占比：{mask.mean() * 100:.2f}%")
    print(f"结果保存位置：{args.out_dir}")


def run_labeled(args, model, dev):
    """train / val：有标签，保存可视化并计算指标。"""
    R = Path(args.data_root)
    if args.split == "train":
        pairs = list_pairs(R / args.train_img, R / args.train_msk)
    else:
        pairs = list_pairs(R / args.val_img, R / args.val_msk)
    print(f"{args.split}: {len(pairs)} 张")
    ds = SegDS(pairs, args.size, train=False)

    os.makedirs(args.out_dir, exist_ok=True)
    meter, rows = Meter(), []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            pred = model(x[None].to(dev)).argmax(1)[0].cpu()
            meter.update(pred, y)

            p, g = pred.numpy(), y.numpy()
            tp = int(((p == 1) & (g == 1)).sum())
            fp = int(((p == 1) & (g == 0)).sum())
            fn = int(((p == 0) & (g == 1)).sum())
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")

            ip = pairs[i][0]
            stem = os.path.splitext(os.path.basename(ip))[0]
            img = np.asarray(Image.open(ip).convert("RGB").resize((args.size, args.size), Image.BILINEAR))
            err = np.zeros((*p.shape, 3), np.uint8)
            err[(p == 1) & (g == 1)] = (0, 200, 0)
            err[(p == 1) & (g == 0)] = (255, 0, 0)
            err[(p == 0) & (g == 1)] = (0, 120, 255)
            canvas = np.concatenate([img, overlay(img, g), overlay(img, p), err], axis=1)
            Image.fromarray(canvas).save(os.path.join(args.out_dir, f"{i:03d}_{stem}.png"))
            rows.append((stem, iou, tp, fp, fn))

    # 逐张前景 IoU，从差到好排序，方便找出最失败的样本
    rows.sort(key=lambda r: 2 if r[1] != r[1] else r[1])
    with open(os.path.join(args.out_dir, "per_image_iou.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "IoU_landslide", "TP", "FP", "FN"])
        w.writerows(rows)

    res = meter.compute()
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(args.out_dir, f"{args.split}_metrics_recomputed.json"), "w"), indent=1)
    print("可视化图和 per_image_iou.csv 已保存到:", args.out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DEFAULT_ROOT)
    ap.add_argument("--train_img", default="train_data/train_data")
    ap.add_argument("--train_msk", default="train_data/train_label")
    ap.add_argument("--val_img", default="val_data/val_data")
    ap.add_argument("--val_msk", default="val_data/val_label")
    ap.add_argument("--pred_img", default=r"D:\Data\TestData\test_data\Lushan\Lushan_2.tif")  # 无标签测试影像
    ap.add_argument("--ckpt", default=r"D:\Data\DeepLabV3_PY\run\run100_256_1\best.pth")
    ap.add_argument("--out_dir", default=r"D:\Data\DeepLabV3_PY\Prediction\Lushan2")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--size", type=int, default=256)        # 训练输入尺寸，也是 test 的滑窗大小
    ap.add_argument("--overlap", type=int, default=32)      # test 滑窗重叠像素
    ap.add_argument("--bs", type=int, default=8)            # test 滑窗批大小
    ap.add_argument("--scale", type=float, default=1.0)     # test 推理前整体缩放比例（尺度不一致时调）
    ap.add_argument("--thr", type=float, default=0.5)       # test 滑坡概率阈值
    ap.add_argument("--no_ldaspp", action="store_true")
    ap.add_argument("--no_gam", action="store_true")
    ap.add_argument("--no_mrff", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LandslideDeepLab(2, False, not args.no_ldaspp, not args.no_gam, not args.no_mrff).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()

    if args.split == "test":
        run_test(args, model, dev)
    else:
        run_labeled(args, model, dev)


if __name__ == "__main__":
    main()
