import os
import random
import copy
import torch
import numpy as np
import csv

import wandb
from matplotlib import pyplot as plt

from tabsyn.process_syn_dataset import preprocess_syn
from tabsyn.latent_utils import get_encoder_latent

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def circle_mask(height=17117, width=44, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
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


def eval_watermark(reversed_latents, watermarking_mask, gt_patch, args):
    reversed_latents = reversed_latents.unsqueeze(0).unsqueeze(0)

    if 'complex' in args.w_measurement:
        reversed_latents_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents), dim=(-1, -2))
        target_patch = gt_patch
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')
    if 'l1' in args.w_measurement:
        metric = torch.abs(reversed_latents_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')
    
    return metric
    


### Deletion attacks
def delete_random_rows(input_path, additional_path, output_path, delete_percentage=0.1):
    # attack: deleting the rows
    with open(input_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    
    header = data[0]  # Save the header row
    data = data[1:]
    num_rows_to_delete = int((len(data)) * delete_percentage)
    rows_to_delete = random.sample(range(len(data)), num_rows_to_delete)
    
    filtered_data = [header] + [row for idx, row in enumerate(data) if idx not in rows_to_delete]

    # detection: add additional rows from non-watermark data to complete detection
    with open(additional_path, 'r') as file:
        reader = csv.reader(file)
        additional_data = list(reader)
    additional_data = additional_data[1:]
    rows_to_add = random.sample(range(len(data)), num_rows_to_delete)
    add_data = [row for idx, row in enumerate(additional_data) if idx in rows_to_add]

    detection_data = filtered_data + add_data
    
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(detection_data)


def delete_random_values(input_path, additional_path, output_path, delete_percentage=0.1):
    with open(input_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    
    header = data[0]
    data = data[1:]

    # Calculate the total number of values in the data
    last_row = len(data) - 1
    last_col = len(data[0]) - 1
    total_values = last_row * last_col
    num_values_to_delete = int(total_values * delete_percentage)

    with open(additional_path, 'r') as file:
        reader = csv.reader(file)
        additional_data = list(reader)
    additional_data = np.array(additional_data[1:])
    
    for _ in range(num_values_to_delete):
        row_idx = random.randint(0, last_row)
        col_idx = random.randint(0, last_col)
        data[row_idx][col_idx] = np.random.choice(additional_data[:, col_idx])
    
    filtered_data = [header] + data
    
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(filtered_data)


def delete_random_columns(input_path, additional_path, output_path, delete_percentage=0.1):
    with open(input_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    
    # detection: add additional rows from non-watermark data to complete detection
    with open(additional_path, 'r') as file:
        reader = csv.reader(file)
        additional_data = list(reader)
    
    header = data[0]  # Save the header row
    data = np.array(data[1:])
    num_columns = len(data[0])

    additional_data = np.array(additional_data[1:])
    
    num_columns_to_distort = max(1, int(num_columns * delete_percentage))
    columns_to_distort = random.sample(range(num_columns), num_columns_to_distort)

    for idx in columns_to_distort:
        data[:, idx] = additional_data[:, idx]
    
    data = data.tolist()
    distored_data = [header] + data
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(distored_data)


def delete_random_latent_rows(args, info, save_dir, model, noise_scheduler, delete_percentage=0.1):
    X_num_w, X_cat_w = preprocess_syn(save_dir, info['task_type'], k='w')
    syn_latent_w = get_encoder_latent(X_num_w, X_cat_w, info, args.device).detach().cpu().numpy()

    X_num_no_w, X_cat_no_w = preprocess_syn(save_dir, info['task_type'], k='no-w')
    syn_latent_no_w = get_encoder_latent(X_num_no_w, X_cat_no_w, info, args.device).detach().cpu().numpy()
    num_rows = syn_latent_w.shape[0]
    num_rows_to_delete = int(num_rows * delete_percentage)
    rows_to_delete = random.sample(range(num_rows), num_rows_to_delete)
    rows_to_add = random.sample(range(num_rows), num_rows_to_delete)

    deleted_syn_latent_w = np.delete(syn_latent_w, rows_to_delete, axis=0)
    sampled_syn_latent_no_w = syn_latent_no_w[rows_to_add]

    syn_latent_w = np.concatenate((deleted_syn_latent_w, sampled_syn_latent_no_w), axis=0)
    syn_latent_w = torch.tensor(syn_latent_w, device=args.device)

    reversed_noise = noise_scheduler.gen_reverse(
            model.noise_fn,
            syn_latent_w,
            num_inference_steps=args.steps,
            eta=0.0)
    # Perform FFT
    # Perform FFT
    fft_result = np.fft.fft2(reversed_noise.detach().cpu().numpy())
    fft_shift = np.fft.fftshift(fft_result)  # Shift the zero frequency component to the center of the spectrum
    magnitude_spectrum = 20*np.log(np.abs(fft_shift) + 1)  # Log scaling for better visibility

    # Plot FFT as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(magnitude_spectrum, aspect='auto', extent=[0, magnitude_spectrum.shape[1], 0, magnitude_spectrum.shape[0]])
    plt.colorbar(label='Magnitude (dB)')
    plt.title('FFT Magnitude Spectrum')
    plt.xlabel('Frequency X')
    plt.ylabel('Frequency Y')

    # Upload plot to wandb
    wandb.log({"FFT Magnitude Spectrum": wandb.Image(plt)})

    plt.close()

    return reversed_noise



def delete_random_latent_columns(args, info, save_dir, model, noise_scheduler, delete_percentage=0.1):
    X_num_w, X_cat_w = preprocess_syn(save_dir, info['task_type'], k='w')
    syn_latent_w = get_encoder_latent(X_num_w, X_cat_w, info, args.device).detach().cpu().numpy()

    X_num_no_w, X_cat_no_w = preprocess_syn(save_dir, info['task_type'], k='no-w')
    syn_latent_no_w = get_encoder_latent(X_num_no_w, X_cat_no_w, info, args.device).detach().cpu().numpy()

    num_cols = syn_latent_w.shape[1]
    num_cols_to_delete = int(num_cols * delete_percentage)

    cols_to_delete = random.sample(range(num_cols), num_cols_to_delete)
    cols_to_add = random.sample(range(syn_latent_no_w.shape[1]), num_cols_to_delete)

    deleted_syn_latent_w = np.delete(syn_latent_w, cols_to_delete, axis=1)

    sampled_syn_latent_no_w = syn_latent_no_w[:, cols_to_add]

    syn_latent_w = np.concatenate((deleted_syn_latent_w, sampled_syn_latent_no_w), axis=1)
    syn_latent_w = torch.tensor(syn_latent_w, device=args.device)

    reversed_noise = noise_scheduler.gen_reverse(
        model.noise_fn,
        syn_latent_w,
        num_inference_steps=args.steps,
        eta=0.0)

    return reversed_noise
### Distortion attacks
def distort_random_rows(input_path, output_path, n_distance=10, delete_percentage=0.1):
    with open(input_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    header = data[0]  # Save the header row
    data = data[1:]
    num_rows = len(data)
    
    num_rows_to_distort = int(num_rows * delete_percentage)
    rows_to_distort = random.sample(range(num_rows), num_rows_to_distort)

    for idx in rows_to_distort:
        selected_row = data[idx]

        start_index = max(0, idx - n_distance)
        end_index = min(num_rows - 1, idx + n_distance)

        while True:
            random_idx = random.randint(start_index, end_index)
            if random_idx != idx:
                break

        chosen_neighbor = copy.deepcopy(data[random_idx])
        selected_row[:] = chosen_neighbor[:]
    
    distored_data = [header] + data
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(distored_data)


def distort_random_columns(input_path, output_path, n_distance=10, delete_percentage=0.1):
    with open(input_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    header = data[0]  # Save the header row
    data = np.array(data[1:])
    num_rows = len(data)
    num_columns = len(data[0])
    
    num_columns_to_distort = max(1, int(num_columns * delete_percentage))
    columns_to_distort = random.sample(range(num_columns), num_columns_to_distort)

    for idx in columns_to_distort:
        selected_column = data[:, idx]
        new_column = np.zeros_like(selected_column)

        for i in range(len(selected_column)):

            start_index = max(0, i - n_distance)
            end_index = min(num_rows - 1, i + n_distance)
            while True:
                random_idx = random.randint(start_index, end_index)
                if random_idx != i:
                    break

            new_column[i] = selected_column[random_idx]
        
        selected_column = new_column
    
    data = data.tolist()
    distored_data = [header] + data
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(distored_data)


def distort_random_values(input_path, output_path, n_distance=10, delete_percentage=0.1):
    with open(input_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    
    header = data[0]
    data = data[1:]

    # Calculate the total number of values in the data
    total_values = sum(len(row) for row in data)
    num_values_to_delete = int(total_values * delete_percentage)
    last_row = len(data) - 1
    last_col = len(data[0]) - 1

    for _ in range(num_values_to_delete):
        row_idx = random.randint(0, last_row)
        col_idx = random.randint(0, last_col)

        start_index = max(0, row_idx - n_distance)
        end_index = min(last_row - 1, row_idx + n_distance)
        neighbor_idx = random.randint(start_index, end_index)

        data[row_idx][col_idx] = data[neighbor_idx][col_idx]
    
    filtered_data = [header] + data
    
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(filtered_data)
    

def distort_random_latent_rows(args, info, save_dir, model, noise_scheduler, n_distance=10, delete_percentage=0.1):
    X_num_w, X_cat_w = preprocess_syn(save_dir, info['task_type'], k='w')
    syn_latent_w = get_encoder_latent(X_num_w, X_cat_w, info, args.device).detach().cpu().numpy()

    num_rows = syn_latent_w.shape[0]
    num_rows_to_distort = int(num_rows * delete_percentage)
    rows_to_distort = random.sample(range(num_rows), num_rows_to_distort)

    for idx in rows_to_distort:
        start_index = max(0, idx - n_distance)
        end_index = min(num_rows - 1, idx + n_distance)

        while True:
            random_idx = random.randint(start_index, end_index)
            if random_idx != idx:
                break

        chosen_neighbor = copy.deepcopy(syn_latent_w[random_idx])
        syn_latent_w[idx] = chosen_neighbor[:]

    syn_latent_w = torch.tensor(syn_latent_w, device=args.device)

    reversed_noise = noise_scheduler.gen_reverse(
            model.noise_fn,
            syn_latent_w,
            num_inference_steps=args.steps,
            eta=0.0)
    
    return reversed_noise




