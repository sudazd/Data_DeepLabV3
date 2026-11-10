"""
一键生成：
  val  —— 验证集的预测图 + 精度指标
  test —— Lushan.tif（无标签）的滑窗预测图，不输出精度
与 model.py、train.py、predict.py 放在同一个文件夹，直接在 PyCharm 里右键运行即可。
结果保存到 OUT\\val 和 OUT\\test。
"""
import os
import sys

import predict

# ======== 只需要改这几处 ========
DATA_ROOT = r"D:\Data"   # 其下应有 train_data / val_data / Test_data
CKPT = r"D:\Data\DeepLabV3_PY\run\run100_256_1\best.pth"
SIZE = 256        # 必须与该权重训练时的 --size 一致
EXTRA = []        # 训练时用过消融开关就写进来，例如 ["--no_gam"] 或 ["--no_ldaspp", "--no_gam", "--no_mrff"]
OUT = r"D:\Data\DeepLabV3_PY\Prediction\LuShan2"

PRED_IMG = r"D:\Data\TestData\test_data\Lushan\Lushan_2.tif"    # 相对 DATA_ROOT
OVERLAP = 32      # Lushan 滑窗重叠像素
SCALE = 1.0       # Lushan 推理前缩放比例；若地物尺度与训练数据差别大，可试 0.5 / 2.0
THR = 0.5         # 滑坡概率阈值
# ================================

for split in ("val", "test"):
    sys.argv = [
        "predict.py",
        "--data_root", DATA_ROOT,
        "--ckpt", CKPT,
        "--size", str(SIZE),
        "--split", split,
        "--out_dir", os.path.join(OUT, split),
        "--pred_img", PRED_IMG,
        "--overlap", str(OVERLAP),
        "--scale", str(SCALE),
        "--thr", str(THR),
    ] + EXTRA
    print("=" * 20, split, "=" * 20, flush=True)
    predict.main()
