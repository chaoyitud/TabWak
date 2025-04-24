import os

import pandas as pd
import torch
import wandb
import numpy as np

from tabsyn.model import MLPDiffusion, DDIMModel, DDIMScheduler
from tabsyn.latent_utils import get_input_generate, get_encoder_latent, get_decoder_latent
from watermark.watermark_utils import get_zscore, get_zscore_TR
from tabsyn.process_syn_dataset import process_data, preprocess_syn


def get_watermark_metric(
        args, dataname, data_path, save_dir, pre_k, info, model, noise_scheduler,
        watermarking_mask, gt_patch, mean=0, latents=None, X_num=None, X_cat=None,
        k=None, mask_col=None
):
    # Preprocess data
    process_data(name=dataname, data_path=data_path, save_dir=save_dir, k=pre_k)
    if X_num is None or X_cat is None:
        X_num, X_cat = preprocess_syn(save_dir, info['task_type'], k=pre_k)

    num_cols = X_num.shape[1]

    # Get the latent of the synthetic tabular from the vae encoder
    syn_latent_encoder = get_encoder_latent(X_num, X_cat, info, args.device)
    syn_latent = syn_latent_encoder
    #syn_latent = get_decoder_latent(X_num, X_cat, info, args.device, aux=latents, mask_col=mask_col)
    #syn_latent = latents # input saved latents for debugging

    mean = mean.to(args.device)

    syn_latent = (syn_latent - mean) / 2

    # Reverse the noise using DDIM-Inversion
    reversed_noise = noise_scheduler.gen_reverse(
        model.noise_fn, syn_latent, num_inference_steps=args.steps, eta=0.0
    )
    # save metric to experiment dir
    parent_dir = os.path.dirname(save_dir)
    metric_dir = os.path.join(parent_dir, args.exp_prefix)
    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir)
    metric = evaluate_watermark_methods(args, reversed_noise, watermarking_mask, gt_patch, k, num_cols=num_cols,
                                        save_dir=metric_dir)
    return metric


def evaluate_watermark_methods(args, reversed_noise, watermarking_mask, gt_patch, k=None, num_cols=None, save_dir=None):
    # Gaussian
    if args.with_w == 'GS':
        metric = eval_GS(reversed_noise, k=k, save_dir=save_dir)
    elif args.with_w == 'TabWak':
        metric = eval_TabWak(reversed_noise, k=k, num_cols=num_cols, token_dim=4, save_dir=save_dir)
    elif args.with_w == 'TabWak*':
        metric = eval_TabWak_star(reversed_noise, k=k, num_cols=num_cols, token_dim=4, save_dir=save_dir)
    elif args.with_w == 'treering':
        # TreeRing
        metric = eval_TR(reversed_noise, watermarking_mask, gt_patch, args, save_dir=save_dir, k=k)
    return metric


def eval_TabWak(reversed_noise, k=None, num_cols=None, token_dim=4, save_dir=None):
    cnt = 0
    correct = 0
    cnt_num = 0
    correct_num = 0
    cnt_cat = 0
    correct_cat = 0
    reversed_noise_num = reversed_noise[:, :num_cols * token_dim]
    reversed_noise_cat = reversed_noise[:, num_cols * token_dim:]
    for name, reversed_noise in zip(['num', 'cat'], [reversed_noise_num, reversed_noise_cat]):
        mid = torch.quantile(reversed_noise, 0.5)
        for i in range(reversed_noise.shape[0]):
            for j in range(reversed_noise.shape[1]):
                if reversed_noise[i][j] <= mid:
                    reversed_noise[i][j] = 0
                else:
                    reversed_noise[i][j] = 1

    for i in range(reversed_noise.shape[0]):
        cnt_row = 0
        correct_row = 0
        for name, reversed_noise_split in zip(['num', 'cat'], [reversed_noise_num, reversed_noise_cat]):
            bsz, seq_len = reversed_noise_split.shape
            torch.manual_seed(217)
            permutation = torch.randperm(seq_len)
            inverse_permutation = torch.argsort(permutation)
            row = reversed_noise_split[i]
            row = row[inverse_permutation]
            half_dim = seq_len // 2
            first_half = row[:half_dim]
            last_half = row[half_dim:]
            correct_row_split = (first_half == last_half).sum().item()
            cnt_row_split = half_dim
            cnt += cnt_row_split
            correct += correct_row_split
            cnt_row += cnt_row_split
            correct_row += correct_row_split
            if name == 'num':
                cnt_num += cnt_row_split
                correct_num += correct_row_split
            else:
                cnt_cat += cnt_row_split
                correct_cat += correct_row_split
        acc_bit_row = correct_row / cnt_row if cnt_row != 0 else -1
        acc_bit_num = correct_num / cnt_num if cnt_num != 0 else -1
        acc_bit_cat = correct_cat / cnt_cat if cnt_cat != 0 else -1
        if cnt_row != 0:
            wandb.log({f'{k}-acc_bit_row': acc_bit_row})
            with open(f'{save_dir}/{k}-detect.json', 'a') as f:
                f.write(f'{acc_bit_row}\n')
        if cnt_num != 0:
            wandb.log({f'{k}-acc_bit_num': acc_bit_num})
        if cnt_cat != 0:
            wandb.log({f'{k}-acc_bit_cat': acc_bit_cat})
    avg_bit_accuracy = correct / cnt if cnt != 0 else -1
    return avg_bit_accuracy


def eval_TabWak_star(reversed_noise, k=None, num_cols=None, token_dim=4, save_dir=None):
    cnt = 0
    correct = 0
    cnt_num = 0
    correct_num = 0
    cnt_cat = 0
    correct_cat = 0
    reversed_noise_num = reversed_noise[:, :num_cols * token_dim]
    reversed_noise_cat = reversed_noise[:, num_cols * token_dim:]
    for name, reversed_noise in zip(['num', 'cat'], [reversed_noise_num, reversed_noise_cat]):
        q1 = torch.quantile(reversed_noise, 0.25)
        q2 = torch.quantile(reversed_noise, 0.5)
        q3 = torch.quantile(reversed_noise, 0.75)
        for i in range(reversed_noise.shape[0]):
            for j in range(reversed_noise.shape[1]):
                if reversed_noise[i][j] <= q1:
                    reversed_noise[i][j] = 0
                elif reversed_noise[i][j] >= q3:
                    reversed_noise[i][j] = 1
                elif reversed_noise[i][j] > q1 and reversed_noise[i][j] < q2:
                    reversed_noise[i][j] = 2
                elif reversed_noise[i][j] >= q2 and reversed_noise[i][j] < q3:
                    reversed_noise[i][j] = 3

    for i in range(reversed_noise.shape[0]):
        cnt_row = 0
        correct_row = 0
        for name, reversed_noise_split in zip(['num', 'cat'], [reversed_noise_num, reversed_noise_cat]):
            bsz, seq_len = reversed_noise_split.shape
            torch.manual_seed(217)
            permutation = torch.randperm(seq_len)
            inverse_permutation = torch.argsort(permutation)
            row = reversed_noise_split[i]
            row = row[inverse_permutation]
            half_dim = seq_len // 2
            cnt_row_split = 0
            correct_row_split = 0
            first_half = row[:half_dim]
            last_half = row[half_dim:]
            for j in range(half_dim):
                if first_half[j] == 0 or first_half[j] == 1:
                    cnt_row_split += 1
                if first_half[j] == 0 and (last_half[j] == 0 or last_half[j] == 2):
                    correct_row_split += 1
                if first_half[j] == 1 and (last_half[j] == 1 or last_half[j] == 3):
                    correct_row_split += 1

            cnt += cnt_row_split
            correct += correct_row_split
            cnt_row += cnt_row_split
            correct_row += correct_row_split
            if name == 'num':
                cnt_num += cnt_row_split
                correct_num += correct_row_split
            else:
                cnt_cat += cnt_row_split
                correct_cat += correct_row_split
        acc_bit_row = correct_row / cnt_row if cnt_row != 0 else -1
        acc_bit_num = correct_num / cnt_num if cnt_num != 0 else -1
        acc_bit_cat = correct_cat / cnt_cat if cnt_cat != 0 else -1
        if cnt_row != 0:
            wandb.log({f'{k}-acc_bit_row': acc_bit_row})
            with open(f'{save_dir}/{k}-detect.json', 'a') as f:
                f.write(f'{acc_bit_num}\n')
        if cnt_num != 0:
            wandb.log({f'{k}-acc_bit_num': acc_bit_num})
        if cnt_cat != 0:
            wandb.log({f'{k}-acc_bit_cat': acc_bit_cat})
    avg_bit_accuracy = correct / cnt if cnt != 0 else -1
    return avg_bit_accuracy


def eval_TR(reversed_latents, watermarking_mask, gt_patch, args, save_dir=None, k=None):
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
    with open(f'{save_dir}/{k}-detect.json', 'a') as f:
        f.write(f'{metric}\n')
    return metric

def eval_GS(reversed_noise, k=None, save_dir=None):
    total_elements = reversed_noise.shape[0] * reversed_noise.shape[1]
    cnt = 0
    reversed_noise = (reversed_noise - reversed_noise.mean()) / reversed_noise.std()
    torch.manual_seed(217)
    latent_seed = torch.randint(0, 2, (reversed_noise.shape[1],), device=reversed_noise.device)
    for row in reversed_noise:
        sign_row = (row > 0).int()
        cnt_row = (sign_row == latent_seed).sum().item()
        cnt += cnt_row
        acc_bit_row = cnt_row / reversed_noise.shape[1]
        wandb.log({f'{k}-acc_bit_row': acc_bit_row})
        with open(f'{save_dir}/{k}-detect.json', 'a') as f:
            f.write(f'{acc_bit_row}\n')
    proportion = cnt / total_elements
    return proportion


def attack_numpy(attack_type, attack_percentage, X_num, X_cat, X_num_pre, X_cat_pre, args):
    mask_col = None
    print('Attack:', attack_type)
    if attack_type == 'rowdeletion':
        num_rows = X_num.shape[0]
        num_rows_delete = int(num_rows * attack_percentage)
        rows_delete = np.random.choice(num_rows, num_rows_delete, replace=False)
        X_num = np.delete(X_num, rows_delete, axis=0)
        X_cat = np.delete(X_cat, rows_delete, axis=0)
        if args.with_w == 'treering':
            rows_add = np.random.choice(X_num_pre.shape[0], num_rows_delete, replace=False)
            X_num = np.concatenate([X_num, X_num_pre[rows_add]], axis=0)
            X_cat = np.concatenate([X_cat, X_cat_pre[rows_add]], axis=0)
        else:
            raise ValueError('Attack type not supported')
    elif attack_type == 'coldeletion':
        num_cols = X_num.shape[1]
        num_cols_delete = 1 if attack_percentage == 0.05 else 2 if attack_percentage == 0.1 else 3
        if num_cols_delete > num_cols:
            raise ValueError('Number of columns to delete is greater than the number of columns')
        wandb.log({'num_cols_delete': num_cols_delete})
        wandb.log({'num_cols': num_cols})

        cols_delete = np.random.choice(num_cols, num_cols_delete, replace=False)
        mask_col = cols_delete
        X_num[:, cols_delete] = X_num_pre[:, cols_delete]
    elif attack_type == 'celldeletion':
        num_values = X_num.shape[0] * X_num.shape[1]
        num_values_delete = int(num_values * attack_percentage)
        values_delete = np.random.choice(num_values, num_values_delete, replace=False)
        rows_delete = values_delete // X_num.shape[1]
        cols_delete = values_delete % X_num.shape[1]
        X_num[rows_delete, cols_delete] = X_num_pre[rows_delete, cols_delete]
    elif attack_type == 'noise':
        multiplier = np.random.uniform(1 - attack_percentage, 1 + attack_percentage, X_num.shape)
        X_num = X_num * multiplier
    return X_num, X_cat, mask_col


def main(args, i):
    dataname = args.dataname
    device = args.device
    save_path_arg = args.save_path
    with_w = args.with_w
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    num_samples = args.num_samples
    args.w_radius = args.num_samples // 10

    if num_samples == -1:
        # sample as size of dataset
        save_dir = f'{curr_dir}/../{save_path_arg}/{with_w}/{i}'
    else:
        save_dir = f'{curr_dir}/../{save_path_arg}/{with_w}/{num_samples}/{i}'

    if not os.path.exists(save_dir):
        # If it doesn't exist, create it
        os.mkdir(save_dir)

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1]

    try:
        watermarking_mask = torch.tensor(np.load(f'{save_dir}/watermarking_mask.npy')).to(device)
    except:
        watermarking_mask = None

    try:
        gt_patch = torch.tensor(np.load(f'{save_dir}/gt_patch.npy')).to(device)
    except:
        gt_patch = None

    mean = train_z.mean(0)
    # Loading diffusion model for inverse process
    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    model = DDIMModel(denoise_fn).to(device)
    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    X_num = None
    X_cat = None
    mask_col = None
    # get the latent of the synthetic tabular from the vae encoder
    if args.mode == 'detect':
        pre_keys = ['no-w', 'w']
        for k in pre_keys:
            save_path = f'{save_dir}/{k}-{args.method}.csv'
            latents = torch.tensor(np.load(f'{save_dir}/{k}-{args.method}.npy')).to(device)

            # load X_num, X_cat from the disk
            X_num_pre = X_num if X_num is not None else None
            X_cat_pre = X_cat if X_cat is not None else None
            X_num = torch.tensor(np.load(f'{save_dir}/{k}-{args.method}-X_num.npy')).to(device)
            X_cat = torch.tensor(np.load(f'{save_dir}/{k}-{args.method}-X_cat.npy')).to(device)

            # attack
            if args.attack != 'none' and k == 'w':
                X_num, X_cat, mask_col = attack_numpy(args.attack, args.attack_percentage, X_num.cpu().numpy(),
                                                      X_cat.cpu().numpy(), X_num_pre.cpu().numpy(),
                                                      X_cat_pre.cpu().numpy(), args)
                X_num = torch.tensor(X_num).to(device)
                X_cat = torch.tensor(X_cat).to(device)

            metric = get_watermark_metric(args, dataname, save_path, save_dir, k, info, model, noise_scheduler,
                                          watermarking_mask, gt_patch, mean, latents, X_num, X_cat, k, mask_col)

            print(f'{k}:', metric)
            wandb.log({f'{k}': float(metric)})

        parent_dir = os.path.dirname(save_dir)
        no_w_metrics = pd.read_csv(f'{parent_dir}/{args.exp_prefix}/no-w-detect.json', header=None).iloc[:, 0]
        w_metrics = pd.read_csv(f'{parent_dir}/{args.exp_prefix}/w-detect.json', header=None).iloc[:, 0]

        p_value = 1e-4

        get_z_fn = get_zscore_TR if args.with_w == 'treering' else get_zscore
        mean_z, std_z, tpr = get_z_fn(no_w_metrics, w_metrics, p_value=p_value)

        box_width = 50
        title = f"RESULTS SUMMARY for Round {i}/{args.num_trials}"

        # Format values
        z_display = f"{mean_z:.4f} ± {std_z:.4f}"
        z_label = "Z-score (mean ± std):"
        z_line = f"{z_label} {z_display}"

        tpr_label = f"TPR @ FPR = {p_value:<6}:"
        tpr_display = f"{tpr:.4f}"
        tpr_line = f"{tpr_label} {tpr_display}"
        print("╔" + "═" * (box_width - 2) + "╗")
        print(f"║{title:^{box_width - 2}}║")
        print("╠" + "═" * (box_width - 2) + "╣")
        print(f"║ {z_line}{' ' * (box_width - 3 - len(z_line))}║")
        print(f"║ {tpr_line}{' ' * (box_width - 3 - len(tpr_line))}║")

        print("╚" + "═" * (box_width - 2) + "╝")

        # log z-score and TPR to wandb
        wandb.log({f'Z': mean_z})
        wandb.log({f'TPR': tpr})


if __name__ == '__main__':
    pass
