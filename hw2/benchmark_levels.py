"""
消融实验：不同金字塔层数、不同掩码的融合质量与性能对比
输出：CSV（metrics）、曲线图（PNG）
"""

import csv
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from laplacian_pyramid_fusion import (
    LaplacianPyramid,
    create_vertical_split_mask,
    create_horizontal_split_mask,
    create_circular_mask,
)
from analyze_fusion import analyze


def make_pair(h=512, w=512):
    img1 = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(w):
        img1[:, i] = [i * 255 // w, 80, 255 - i * 255 // w]
    cv2.circle(img1, (w//3, h//2), 90, (0, 255, 255), -1)

    img2 = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        img2[i, :] = [255 - i * 255 // h, 255 * i // h, 80]
    cv2.rectangle(img2, (w//2-120, h//2-120), (w//2+120, h//2+120), (255, 200, 0), -1)
    return img1, img2


def run_benchmark(levels_list: List[int], masks: Dict[str, np.ndarray], out_prefix='bench'):
    img1, img2 = make_pair()
    records = []

    for name, m in masks.items():
        for L in levels_list:
            lp = LaplacianPyramid(levels=L)
            t0 = time.time()
            fused = lp.blend_images(img1, img2, m)
            fused = np.clip(fused, 0, 255).astype(np.uint8)
            t1 = time.time()
            dt = (t1 - t0) * 1000.0
            # 用相同分析函数产出一致的指标
            metrics = analyze(img1, img2, m.astype(np.float32), levels=L, save_prefix=f'{out_prefix}_{name}_L{L}')
            row = {'mask': name, 'levels': L, 'time_ms': dt}
            row.update(metrics)
            records.append(row)
            print(f"mask={name}, L={L}, time={dt:.2f} ms, SeamGrad_fused={metrics['SeamGrad_fused']:.4f}")

    # 保存CSV
    keys = list(records[0].keys()) if records else []
    with open(f'{out_prefix}_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"结果CSV已保存：{out_prefix}_results.csv")

    # 绘制部分曲线：不同L的SeamGrad_fused与time
    plt.figure(figsize=(10,4))
    for name in masks.keys():
        xs = [r['levels'] for r in records if r['mask']==name]
        ys = [r['SeamGrad_fused'] for r in records if r['mask']==name]
        plt.plot(xs, ys, marker='o', label=f'SeamGrad ({name})')
    plt.xlabel('Levels'); plt.ylabel('SeamGrad (lower better)'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(f'{out_prefix}_seamgrad_vs_levels.png', dpi=150)

    plt.figure(figsize=(10,4))
    for name in masks.keys():
        xs = [r['levels'] for r in records if r['mask']==name]
        ys = [r['time_ms'] for r in records if r['mask']==name]
        plt.plot(xs, ys, marker='o', label=f'Time ({name})')
    plt.xlabel('Levels'); plt.ylabel('Time (ms)'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(f'{out_prefix}_time_vs_levels.png', dpi=150)
    print(f"曲线已保存：{out_prefix}_seamgrad_vs_levels.png, {out_prefix}_time_vs_levels.png")


if __name__ == '__main__':
    H, W = 512, 512
    masks = {
        'vertical': create_vertical_split_mask((H, W), 0.5).astype(np.float32),
        'horizontal': create_horizontal_split_mask((H, W), 0.5).astype(np.float32),
        'circular': create_circular_mask((H, W), smooth=True).astype(np.float32),
    }
    levels_list = [2, 3, 4, 5, 6, 7]
    run_benchmark(levels_list, masks, out_prefix='benchmark')
