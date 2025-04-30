"""
The implementation of TreeRing is inspired by https://github.com/YuxinWenRick/tree-ring-watermark

The implementation of GaussianShading is inspired by https://github.com/zhaoyu-zhao/GaussianShading
"""

import random
import copy
import torch
import numpy as np
import pandas as pd
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def circle_mask(height=17117, width=44, r=10, x_offset=0, y_offset=0):
    x0 = width // 2
    y0 = height // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:height, :width]
    y = y[::-1]

    return ((x - x0)**2 + (y-y0)**2)<= r**2


def get_watermarking_mask(init_latents_w, args, device):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if args.w_mask_shape == 'circle':
        np_mask = circle_mask(height=init_latents_w.shape[-2], width=init_latents_w.shape[-1], r=args.w_radius)
        torch_mask = torch.tensor(np_mask).to(device)

        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, args.w_channel] = torch_mask
    elif args.w_mask_shape == 'square':
        anchor_p = init_latents_w.shape[-1] // 2
        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
        else:
            watermarking_mask[:, args.w_channel, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
    elif args.w_mask_shape == 'no':
        pass
    else:
        raise NotImplementedError(f'w_mask_shape: {args.w_mask_shape}')

    return watermarking_mask


def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
    if args.w_injection == 'complex':
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    elif args.w_injection == 'seed':
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        NotImplementedError(f'w_injection: {args.w_injection}')

    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

    return init_latents_w


def get_watermarking_pattern(args, device, shape, seed):
    set_random_seed(seed)
    gt_init = torch.randn(shape, device=device)

    if 'seed_ring' in args.w_pattern:
        gt_patch = gt_init

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-2], gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'seed_zeros' in args.w_pattern:
        gt_patch = gt_init * 0
    elif 'seed_rand' in args.w_pattern:
        gt_patch = gt_init
    elif 'rand' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'const' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        gt_patch += args.w_pattern_const
    elif 'ring' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-2], gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, i, 0].item()

    return gt_patch



def compute_z_score_sample(args):
    sample, mean_wo, std_wo, num_sample, z_threshold = args
    z_scores = (sample - mean_wo) / (std_wo / (num_sample ** 0.5))
    z_score_mean = z_scores.mean()  # Aggregate to scalar
    z_score_mean = max(z_score_mean, 0)  # Clamp negative to 0
    passed = z_score_mean > z_threshold
    return z_score_mean, passed


def get_zscore(data_wo, data_w, num_sample=1000, repeat_times=100, p_value=1e-3):
    mean_wo = data_wo.mean()
    std_wo = data_wo.std()
    z_threshold = norm.ppf(1 - p_value)

    if len(data_w) < num_sample:
        num_sample = len(data_w)
        print(f'num_sample is too large, set to {num_sample}, current results are not valid')

    args_list = [
        (data_w.sample(n=num_sample), mean_wo, std_wo, num_sample, z_threshold)
        for _ in range(repeat_times)
    ]

    z_score_samples = []
    cnt = 0
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(compute_z_score_sample, args_list))

    for z_score_sample, passed in results:
        z_score_samples.append(z_score_sample)
        if passed:
            cnt += 1

    z_score_samples = np.array(z_score_samples)
    mean_z = z_score_samples.mean()
    std_z = z_score_samples.std()
    tpr = cnt / repeat_times

    return mean_z, std_z, tpr

def get_zscore_TR(data_wo, data_w, p_value=1e-3):
    mean_wo = data_wo.mean()
    std_wo = data_wo.std()
    z_threshold = norm.ppf(1 - p_value)
    z_score_samples = []
    cnt = 0
    for i in range(len(data_w)):
        z_score = -(data_w[i] - mean_wo) / std_wo
        z_score = max(z_score, 0)
        if z_score > z_threshold:
            cnt += 1
        z_score_samples.append(z_score)
    z_score_samples = np.array(z_score_samples)
    mean_z = z_score_samples.mean()
    std_z = z_score_samples.std()
    tpr = cnt / len(data_w)
    return mean_z, std_z, tpr




