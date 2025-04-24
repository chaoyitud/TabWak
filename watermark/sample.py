import os
import torch

import argparse
import warnings
import time
import copy
import numpy as np
from matplotlib import pyplot as plt
import wandb
from tabsyn.model import MLPDiffusion, Model, DDIMModel, DDIMScheduler
#from tabsyn.model import BDIA_DDIMScheduler as DDIMScheduler
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target, get_encoder_latent
from watermark.watermark_utils import get_watermarking_mask, inject_watermark, get_watermarking_pattern
import numpy as np

warnings.filterwarnings('ignore')


def main(args, i):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    save_path_arg = args.save_path
    with_w = args.with_w

    num_samples = args.num_samples
    if num_samples == -1:
        # sample with the same number of training samples (using for evaluation)
        save_dir = f'{curr_dir}/../{save_path_arg}/{with_w}/{i}'
    else:
        save_dir = f'{curr_dir}/../{save_path_arg}/{with_w}/{num_samples}/{i}'
    if not os.path.exists(save_dir):
        # If it doesn't exist, create it
        os.makedirs(save_dir, exist_ok=True)

    args.w_radius = num_samples // 10


    device = args.device
    steps = args.steps

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse, d_num = get_input_generate(args, get_d_num=True)
    in_dim = train_z.shape[1]
    mean = train_z.mean(0)
    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)

    # DDIM
    model = DDIMModel(denoise_fn).to(device)
    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))
    start_time = time.time()

    num_samples = train_z.shape[0] if num_samples == -1 else num_samples
    sample_dim = in_dim
    torch.manual_seed(i)
    init_latents = torch.randn([num_samples, sample_dim], device=device)
    
    # watermarking
    if with_w == 'treering':
        # latents_1: no watermark, latents_2: with watermark
        latents_1 = init_latents.to(device)
        # change from two-dimensional table into watermark size [1, c, l, w]
        init_latents = init_latents.unsqueeze(0).unsqueeze(0)
        init_latent_w = copy.deepcopy(init_latents)
        gt_patch = get_watermarking_pattern(args, device, shape=init_latent_w.shape, seed=i)
        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latent_w, args, device)
        # inject watermark
        latents = inject_watermark(init_latent_w, watermarking_mask, gt_patch, args)
        latents = latents.squeeze(0).squeeze(0)
        latents_2 = latents.to(device)
        # saving gt_patch and watermarking_mask
        np_gt_patch = gt_patch.detach().cpu().numpy()
        np_watermarking_mask = watermarking_mask.detach().cpu().numpy()
        np.save(f'{save_dir}/gt_patch.npy', np_gt_patch)
        np.save(f'{save_dir}/watermarking_mask.npy', np_watermarking_mask)

    elif with_w == 'GS':
        latents_2 = torch.zeros_like(init_latents)

        torch.manual_seed(217)
        latent_seed = torch.randint(0, 2, (init_latents.shape[1],), device=init_latents.device)
        for i in range(init_latents.shape[0]):  # Loop through each sample
            for j in range(init_latents.shape[1]):  # Loop through each dimension
                if latent_seed[j] == 0:  # Even index, sample from the left half of the Gaussian distribution
                    while True:
                        sample = torch.randn(1)
                        if sample < 0:
                            latents_2[i, j] = sample
                            break
                else:
                    while True:
                        sample = torch.randn(1)
                        if sample >= 0:
                            latents_2[i, j] = sample
                            break
        latents_2 = latents_2.to(device)
        latents_1 = init_latents.to(device)

    elif with_w == 'TabWak*':
        def get_tabwakstar_patterns(init_latents):
            bit_string = init_latents.clone()
            condition_0 = bit_string <= -0.67449
            condition_1 = bit_string >= 0.67449
            condition_2 = (bit_string > -0.67449) & (bit_string < 0)
            condition_3 = (bit_string > 0) & (bit_string < 0.67449)

            # Apply conditions to update bit_string_4bits
            bit_string[condition_0] = 0
            bit_string[condition_1] = 1
            bit_string[condition_2] = 2
            bit_string[condition_3] = 3

            batch_size, latent_dim = bit_string.shape

            # Split the bit_string into two equal parts for each row
            split_dim = latent_dim // 2

            bit_string[:, split_dim:] = bit_string[:, :split_dim]

            torch.manual_seed(217)
            permutation = torch.randperm(latent_dim)  # Generate random permutation of indices

            # Apply the same permutation to all rows in the batch
            adjusted_bit_string = bit_string[:, permutation]

            num_samples, num_dimensions = adjusted_bit_string.shape

            # Initialize normal distribution
            normal_dist = torch.distributions.Normal(0, 1)

            # Sample based on adjusted_bit_string values
            random_samples = torch.rand(num_samples, num_dimensions, device=device)

            # Pre-allocate output tensor
            latents_2 = torch.empty_like(adjusted_bit_string, dtype=torch.float32)
            adjusted_bit_string = adjusted_bit_string
            # Generate quantiles in vectorized manner for each condition
            latents_2[adjusted_bit_string == 0] = normal_dist.icdf(random_samples[adjusted_bit_string == 0] * 0.25)
            latents_2[adjusted_bit_string == 1] = normal_dist.icdf(
                random_samples[adjusted_bit_string == 1] * 0.25 + 0.75)
            latents_2[adjusted_bit_string == 2] = normal_dist.icdf(
                random_samples[adjusted_bit_string == 2] * 0.25 + 0.25)
            latents_2[adjusted_bit_string == 3] = normal_dist.icdf(
                random_samples[adjusted_bit_string == 3] * 0.25 + 0.5)
            latents_2 = latents_2.to(device)
            return latents_2

        latents_2_num = get_tabwakstar_patterns(init_latents[:, :d_num * info['token_dim']])
        latents_2_cat = get_tabwakstar_patterns(init_latents[:, d_num * info['token_dim']:])
        latents_2 = torch.cat([latents_2_num, latents_2_cat], dim=1)
        latents_1 = init_latents.to(device)

    elif with_w == 'TabWak':
        def get_tabwak_patterns(init_latents):
            bit_string = init_latents.clone()
            condition_0 = bit_string <= 0
            condition_1 = bit_string > 0

            bit_string[condition_0] = 0
            bit_string[condition_1] = 1

            batch_size, latent_dim = bit_string.shape

            # Split the bit_string into two equal parts for each row
            split_dim = latent_dim // 2

            bit_string[:, split_dim:] = bit_string[:, :split_dim]

            torch.manual_seed(217)
            permutation = torch.randperm(latent_dim)

            adjusted_bit_string = bit_string[:, permutation]

            num_samples, num_dimensions = adjusted_bit_string.shape

            # Initialize normal distribution
            normal_dist = torch.distributions.Normal(0, 1)

            # Sample based on adjusted_bit_string values
            random_samples = torch.rand(num_samples, num_dimensions, device=device)

            # Pre-allocate output tensor
            latents_2 = torch.empty_like(adjusted_bit_string, dtype=torch.float32)
            adjusted_bit_string = adjusted_bit_string
            # Generate quantiles in vectorized manner for each condition
            latents_2[adjusted_bit_string == 0] = normal_dist.icdf(random_samples[adjusted_bit_string == 0] * 0.5)
            latents_2[adjusted_bit_string == 1] = normal_dist.icdf(random_samples[adjusted_bit_string == 1] * 0.5 + 0.5)
            latents_2 = latents_2.to(device)
            return latents_2

        latents_2_num = get_tabwak_patterns(init_latents[:, :d_num * info['token_dim']])
        latents_2_cat = get_tabwak_patterns(init_latents[:, d_num * info['token_dim']:])
        latents_2 = torch.cat([latents_2_num, latents_2_cat], dim=1)
        latents_1 = init_latents.to(device)
    else:
        latents_1 = init_latents.to(device)


    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    x_next_1 = noise_scheduler.generate(
            model.noise_fn,
            latents_1,
            num_inference_steps=steps,
            eta=0.0)
    
    if with_w != 'none':
        # DDIM watermark before diffusion
        x_next_2 = noise_scheduler.generate(
                model.noise_fn,
                latents_2,
                num_inference_steps=steps,
                eta=0.0)


    # Saving the synthetic csv
    x_next_dict = {'no-w': x_next_1, 'w': x_next_2} if with_w != 'none' else {'no-w': x_next_1}

    for k in x_next_dict.keys():
        save_path = f'{save_dir}/{k}-{args.method}.csv'
        save_path_latent = f'{save_dir}/{k}-{args.method}.npy'
        x_next = x_next_dict[k]

        x_next = x_next * 2 + mean.to(device)
        np.save(save_path_latent, x_next.cpu().detach().numpy())
        syn_data = x_next.float().cpu().numpy()
        syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device)
        syn_data = syn_data.reshape(syn_data.shape[0], -1, info['token_dim'])
        decoder = info['pre_decoder']
        recon = decoder(torch.tensor(syn_data))
        X_num, X_cat = recon[0], recon[1]


        X_cat_lst = []
        for pred in X_cat:
            X_cat_lst.append(pred.argmax(dim=-1))
        X_cat = torch.stack(X_cat_lst).t()

        np.save(f'{save_dir}/{k}-{args.method}-X_num.npy', X_num.cpu().detach().numpy())
        np.save(f'{save_dir}/{k}-{args.method}-X_cat.npy', X_cat.cpu().detach().numpy())
        syn_df = recover_data(syn_num, syn_cat, syn_target, info)

        idx_name_mapping = info['idx_name_mapping']
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

        syn_df.rename(columns = idx_name_mapping, inplace=True)
        syn_df.to_csv(save_path, index = False)

        end_time = time.time()
        print('Time:', end_time - start_time)
        print('Saving sampled data to {}'.format(save_path))
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch.')
    parser.add_argument('--steps', type=int, default=None, help='Number of function evaluations.')
    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'