import os
import torch
import wandb
import numpy as np
from scipy.stats import binomtest

from tabsyn.model import MLPDiffusion, Model, DDIMModel, DDIMScheduler
#from tabsyn.model import BDIA_DDIMScheduler as DDIMScheduler
from tabsyn.latent_utils import get_input_generate, get_encoder_latent, get_decoder_latent
from tabsyn.watermark_utils import eval_watermark, delete_random_rows, distort_random_rows, delete_random_latent_rows, delete_random_latent_columns, distort_random_values, distort_random_columns, delete_random_columns, delete_random_values, distort_random_latent_rows
from tabsyn.process_syn_dataset import process_data, preprocess_syn

def get_watermark_metric(args, dataname, data_path, save_dir, pre_k, info, model, noise_scheduler, watermarking_mask, gt_patch, mean=0, latents=None, X_num=None, X_cat=None,k=None, mask_col=None):

    process_data(name=dataname, data_path=data_path, save_dir=save_dir, k=pre_k)
    if X_num is None or X_cat is None:
        X_num, X_cat = preprocess_syn(save_dir, info['task_type'], k=pre_k)
    else:
        X_num, X_cat = X_num, X_cat
    print('X_cat shape:', X_cat.shape)
    print('X_cat_sample:', X_cat[:5])
    syn_latent_encoder = get_encoder_latent(X_num, X_cat, info, args.device)
    syn_latent = get_decoder_latent(X_num, X_cat, info, args.device, aux=latents, mask_col=mask_col)
    num_cols = X_num.shape[1]
    #syn_latent = get_encoder_latent(X_num, X_cat, info, args.device)
    #syn_latent = latents
    mean = mean.to(args.device)
    #print('Mean shape:', mean.shape)
    #syn_latent_test = syn_latent[:,:24]
    #latents_test = latents[:,:24]
    dis_latent = torch.norm(syn_latent[:,::num_cols*4] - latents[:,::num_cols*4], dim=1)
    dis_encoder = torch.norm(syn_latent_encoder[:,::num_cols*4] - latents[:,::num_cols*4], dim=1)
    print('Distance between synthetic and latents:', dis_latent.mean())
    #syn_latent = latents
    wandb.log({f'dis-latent':dis_latent.mean()})
    wandb.log({f'dis-encoder':dis_encoder.mean()})
    syn_latent = (syn_latent - mean) / 2

    # check the distance between the synthetic latent and the latents
    #dis_latent = torch.norm(syn_latent - latents, dim=1)
    print('Synthetic latent shape:', syn_latent.shape)
    if args.with_w == 'VAEtreering':
        metric = eval_watermark(syn_latent, watermarking_mask, gt_patch, args)
    else:
        # DDIM reverse to get the initial noise latent used for synthesizing
        reversed_noise = noise_scheduler.gen_reverse(
                model.noise_fn,
                syn_latent,
                num_inference_steps=args.steps,
                eta=0.0)
        # check mean and std of the reversed noise
        print('Reversed noise mean:', reversed_noise.mean())
        print('Reversed noise std:', reversed_noise.std())
        print('num_cols:', num_cols)
        # Evaluate watermarking metric
        '''
        if args.with_w == 'GS':
            metric, _ = eval_GS(reversed_noise[:,:num_cols*4],k=k)
        elif args.with_w == 'GS+':
            metric = eval_GS_plus(reversed_noise[:,:num_cols*4],k=k)
        elif args.with_w == 'GS++':
            metric = eval_GS_plus_plus(reversed_noise[:,:num_cols*4],k=k)
        else:
           metric = eval_watermark(reversed_noise[:,:num_cols*4], watermarking_mask, gt_patch, args)
        '''
        if args.with_w == 'GS':
            metric, _ = eval_GS(reversed_noise,k=k)
        elif args.with_w == 'GS+':
            metric = eval_GS_plus(reversed_noise,k=k)
        elif args.with_w == 'GS++':
            metric = eval_GS_plus_plus(reversed_noise,k=k)
        else:
           metric = eval_watermark(reversed_noise, watermarking_mask, gt_patch, args)
    return metric

def eval_GS_plus(reversed_noise, bit_dim=4,k=None):
    total_elements = reversed_noise.numel()  # Total number of elements in the noise
    cnt = 0  # Counter for correct bit predictions

    # bound the noise to [-1, 1]
    #reversed_noise = torch.clamp(reversed_noise, -4, 4)
    # Normalize the noise to mean 0 and std 1
    reversed_noise = (reversed_noise - reversed_noise.mean()) / reversed_noise.std()
    # Get the bit sign of the noise
    reversed_noise_sign = (reversed_noise > 0).int()
    shape = reversed_noise_sign.shape
    # Reshape the noise into [number_of_sequences, bit_dim]
    num_sequences = reversed_noise.numel() // bit_dim
    if reversed_noise.numel() % bit_dim != 0:
        print("Warning: Total number of bits is not perfectly divisible by bit_dim, trimming excess...")
        reversed_noise_sign = reversed_noise_sign[:num_sequences * bit_dim]

    reversed_noise_sign = reversed_noise_sign.view(shape[0],-1, bit_dim)

    # Define the mapping from the first three bits to two bits
    mapping = {
        (0, 0): [1, 1],
        (0, 1): [1, 0],
        (1, 1): [0, 0],
        (1, 0): [0, 1],
    }
    for row in reversed_noise_sign:
        cnt_row = 0
        for bits in row:
            #print(bits)
            first_three_bits = tuple(bits[:2].tolist())
            # print(first_three_bits)
            actual_last_two_bits = bits[-2:].tolist()
            generated_last_two_bits = mapping[first_three_bits]

            # Increment the counter for each correct bit
            for gen_bit, act_bit in zip(generated_last_two_bits, actual_last_two_bits):
                if gen_bit == act_bit:
                    cnt_row += 1
        cnt += cnt_row
        portion_row = cnt_row / (2 * row.shape[0])
        wandb.log({f'{k}-portion_row':portion_row})



    # Calculate the average bit accuracy
    total_bits = num_sequences * 2  # Total number of last two bits across all sequences
    avg_bit_accuracy = cnt / total_bits

    return avg_bit_accuracy


def eval_GS_plus_plus_plus(reversed_noise, bit_dim=4,k=None):
    cnt = 0  # Counter for correct bit predictions
    correct = 0
    #reversed_noise = (reversed_noise - reversed_noise.mean()) / reversed_noise.std()
    # get quantile of the noise 25% and 50%, 75%, torch
    q1 = torch.quantile(reversed_noise, 0.25)
    print(q1)
    q2 = torch.quantile(reversed_noise, 0.5)
    print(q2)
    q3 = torch.quantile(reversed_noise, 0.75)
    print(q3)
    shape = reversed_noise.shape
    for i in range(reversed_noise.shape[0]):
        q1 = torch.quantile(reversed_noise[i], 0.25)
        q2 = torch.quantile(reversed_noise[i], 0.5)
        q3 = torch.quantile(reversed_noise[i], 0.75)
        for j in range(reversed_noise.shape[1]):
            if reversed_noise[i][j] <= q1:
                reversed_noise[i][j] = 0
            elif reversed_noise[i][j] >= q3:
                reversed_noise[i][j] = 1
            elif reversed_noise[i][j] > q1 and reversed_noise[i][j] < q2:
                reversed_noise[i][j] = 2
            elif reversed_noise[i][j] >=q2 and reversed_noise[i][j] < q3:
                reversed_noise[i][j] = 3

    bsz, seq_len = reversed_noise.shape
    torch.manual_seed(217)
    permutation = torch.randperm(seq_len)
    inverse_permutation = torch.argsort(permutation)

    reversed_noise = reversed_noise[:, inverse_permutation]
    half_dim = seq_len // 2
    for row in reversed_noise:
        cnt_row = 0
        correct_row = 0
        first_two_bits = row[:half_dim]
        actual_last_two_bits = row[half_dim:]
        for i in range(half_dim):
            if first_two_bits[i] == 0 or first_two_bits[i] == 1:
                cnt_row += 1
            if first_two_bits[i] == 0 and (actual_last_two_bits[i] == 0 or actual_last_two_bits[i] == 2):
                correct_row += 1
            if first_two_bits[i] == 1 and (actual_last_two_bits[i] == 1 or actual_last_two_bits[i] == 3):
                correct_row += 1

        cnt += cnt_row
        correct += correct_row

        portion_row = correct_row / cnt_row if cnt_row != 0 else 0.5
        wandb.log({f'{k}-portion_row':portion_row})

    avg_bit_accuracy = correct / cnt

    return avg_bit_accuracy

def eval_GS_plus_plus(reversed_noise, bit_dim=4,k=None):
    cnt = 0  # Counter for correct bit predictions
    correct = 0
    #reversed_noise = (reversed_noise - reversed_noise.mean()) / reversed_noise.std()
    # get quantile of the noise 25% and 50%, 75%, torch
    q1 = torch.quantile(reversed_noise, 0.25)
    print(q1)
    q2 = torch.quantile(reversed_noise, 0.5)
    print(q2)
    q3 = torch.quantile(reversed_noise, 0.75)
    print(q3)
    shape = reversed_noise.shape
    for i in range(reversed_noise.shape[0]):
        q1 = torch.quantile(reversed_noise[i], 0.25)
        q2 = torch.quantile(reversed_noise[i], 0.5)
        q3 = torch.quantile(reversed_noise[i], 0.75)
        for j in range(reversed_noise.shape[1]):
            if reversed_noise[i][j] <= q1:
                reversed_noise[i][j] = 0
            elif reversed_noise[i][j] >= q3:
                reversed_noise[i][j] = 1
            elif reversed_noise[i][j] > q1 and reversed_noise[i][j] < q2:
                reversed_noise[i][j] = 2
            elif reversed_noise[i][j] >=q2 and reversed_noise[i][j] < q3:
                reversed_noise[i][j] = 3

    # shuffle the reversed_noise through the -1 axis
    shape = reversed_noise.shape
    #permuted_indices = torch.randperm(shape[1])
    #reversed_noise = reversed_noise[:, permuted_indices]
    reversed_noise_sign = reversed_noise.view(shape[0],-1, bit_dim)
    # shuffle the bits
    for row in reversed_noise_sign:
        cnt_row = 0
        correct_row = 0
        for bits in row:
            #print(bits)
            first_two_bits = tuple(bits[:2].tolist())
            # print(first_three_bits)
            actual_last_two_bits = bits[-2:].tolist()
            for i in range(2):
                #print(first_two_bits[i])
                if first_two_bits[i] == 0 or first_two_bits[i] == 1:
                    cnt_row += 1
                if first_two_bits[i] == 0 and (actual_last_two_bits[i] == 0 or actual_last_two_bits[i] == 2):
                    correct_row += 1
                if first_two_bits[i] == 1 and (actual_last_two_bits[i] == 1 or actual_last_two_bits[i] == 3):
                    correct_row += 1

        cnt += cnt_row
        correct += correct_row

        portion_row = correct_row / cnt_row if cnt_row != 0 else 0.5
        wandb.log({f'{k}-portion_row':portion_row})

    avg_bit_accuracy = correct / cnt

    return avg_bit_accuracy












def eval_GS(reversed_noise,k=None):
    total_elements = reversed_noise.shape[0]*reversed_noise.shape[1] # Total number of elements in the noise
    cnt = 0

    # for each row, normalize the noise into gaussian distribution with mean 0 and std 1
    reversed_noise = (reversed_noise - reversed_noise.mean()) / reversed_noise.std()

    for i in range(reversed_noise.shape[0]):
        cnt_row = 0
        for j in range(reversed_noise.shape[1]):
            if j % 2 == 0:
                # Check if the element is greater than the median
                if reversed_noise[i][j] < 0:
                    cnt_row += 1
            if j % 2 == 1:
                # Check if the element is less than the median
                if reversed_noise[i][j] > 0:
                    cnt_row += 1
        cnt += cnt_row
        portion_row = cnt_row / reversed_noise.shape[1]
        wandb.log({f'{k}-portion_row':portion_row})
        

    # Calculate the proportion of counts relative to the total number of elements
    proportion = cnt / total_elements
    #proportion = cnt
    print(f'Proportion: {proportion}')
    print(f'Count: {cnt}')
    print(f'Total elements: {total_elements}')

    # Under the null hypothesis, the expected probability of success (meeting criteria) is 0.5
    # Calculate the p-value using a binomial test
    #p_value = binomtest(cnt, total_elements, 0.5).pvalue
    p_value = 0
    return proportion, p_value


def attack_numpy(attack_type, attack_percentage, X_num, X_cat, X_num_pre, X_cat_pre, args):
    mask_col = None
    print('Attack:', attack_type)
    if attack_type == 'rdeletion':
        # select rows to delete
        num_rows = X_num.shape[0]
        num_rows_delete = int(num_rows * attack_percentage)
        rows_delete = np.random.choice(num_rows, num_rows_delete, replace=False)
        # delete rows
        X_num = np.delete(X_num, rows_delete, axis=0)
        X_cat = np.delete(X_cat, rows_delete, axis=0)
        if args.with_w == 'treering':
            # add the random rows from X_num_pre, X_cat_pre to X_num, X_cat to keep the same number of rows
            rows_add = np.random.choice(X_num_pre.shape[0], num_rows_delete, replace=False)
            X_num = np.concatenate([X_num, X_num_pre[rows_add]], axis=0)
            X_cat = np.concatenate([X_cat, X_cat_pre[rows_add]], axis=0)
        else:
            # Throw an error
            raise ValueError('Attack type not supported')

    elif attack_type == 'cdeletion':
        # select columns to delete
        print("test")
        num_cols = X_num.shape[1]
        # percentage = 0.05 , delete, 1, percentage=0.1, delete 2, percentage=0.2, delete 3
        num_cols_delete = 1 if attack_percentage == 0.05 else 2 if attack_percentage == 0.1 else 3
        # check if the number of columns to delete is greater than the number of columns
        if num_cols_delete > num_cols:
            raise ValueError('Number of columns to delete is greater than the number of columns')
        wandb.log({'num_cols_delete':num_cols_delete})
        wandb.log({'num_cols':num_cols})

        cols_delete = np.random.choice(num_cols, num_cols_delete, replace=False)
        mask_col = cols_delete
        print('Masked columns:', mask_col)
        # replace the columns with the no w columns
        X_num[:, cols_delete] = X_num_pre[:, cols_delete]

    elif attack_type == 'ascdeletion':
        #
        catcols = X_cat.shape[1]
        # percentage = 0.05 , delete, 1, percentage=0.1, delete 2, percentage=0.2, delete 3
        num_cols_delete = 1 if attack_percentage == 0.05 else 2 if attack_percentage == 0.1 else 3
        # check if the number of columns to delete is greater than the number of columns
        if num_cols_delete > catcols:
            raise ValueError('Number of columns to delete is greater than the number of columns')
        wandb.log({'num_cols_delete':num_cols_delete})
        wandb.log({'num_cols':catcols})

        cols_delete = np.random.choice(catcols, num_cols_delete, replace=False)
        mask_col = cols_delete
        print('Masked columns:', mask_col)
        # replace the columns with the no w columns
        X_cat[:, cols_delete] = X_cat_pre[:, cols_delete]
    elif attack_type == 'vdeletion':
        # select values to delete (Cell Deletion)
        num_values = X_num.shape[0] * X_num.shape[1]
        num_values_delete = int(num_values * attack_percentage)
        values_delete = np.random.choice(num_values, num_values_delete, replace=False)
        # replace the values with the no w values
        rows_delete = values_delete // X_num.shape[1]
        cols_delete = values_delete % X_num.shape[1]
        X_num[rows_delete, cols_delete] = X_num_pre[rows_delete, cols_delete]
        #X_cat[rows_delete, cols_delete] = X_cat_pre[rows_delete, cols_delete]
    elif attack_type == 'noise':
        # get uniform mutipiplier from (1-attack_percentage, 1+attack_percentage)
        multiplier = np.random.uniform(1-attack_percentage, 1+attack_percentage, X_num.shape)
        # dot product with the original data
        X_num = X_num * multiplier
    return X_num, X_cat, mask_col

def main(args,i):
    X_num_pre = None
    X_cat_pre = None
    for i in range(10,60):
        if i == 10:
            X_num_pre, X_cat_pre = loop(args, i, None, None)
        else:
            X_num_pre, X_cat_pre = loop(args, i, X_num_pre, X_cat_pre)
def loop(args, i, X_num_pre=None, X_cat_pre=None):
    dataname = args.dataname
    device = args.device
    save_path_arg = args.save_path
    w_radius = args.w_radius
    with_w = args.with_w

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # num_samples = args.num_samples
    num_samples = 5000
    if with_w == 'VAEtreering':
        save_dir = f'{curr_dir}/../{save_path_arg}/VAE/{i}'
    elif num_samples == -1:
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
    if args.mode == 'watermark':
        pre_keys = ['no-w', 'w']
        if i in [j for j in range(10,60)]:
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
                    X_num, X_cat, mask_col = attack_numpy(args.attack, args.attack_percentage, X_num.cpu().numpy(), X_cat.cpu().numpy(), X_num_pre.cpu().numpy(), X_cat_pre.cpu().numpy(), args)
                    X_num = torch.tensor(X_num).to(device)
                    X_cat = torch.tensor(X_cat).to(device)
                metric = get_watermark_metric(args, dataname, save_path, save_dir, k, info, model, noise_scheduler, watermarking_mask, gt_patch, mean, latents, X_num, X_cat,k, mask_col)
                print(f'{k}:', metric)
                wandb.log({f'{k}':float(metric)})
        else:
            k = 'w'
            save_path = f'{save_dir}/{k}-{args.method}.csv'
            latents = torch.tensor(np.load(f'{save_dir}/{k}-{args.method}.npy')).to(device)
            # load X_num, X_cat from the disk
            X_num_pre = X_num_pre
            X_cat_pre = X_cat_pre
            X_num = torch.tensor(np.load(f'{save_dir}/{k}-{args.method}-X_num.npy')).to(device)
            X_cat = torch.tensor(np.load(f'{save_dir}/{k}-{args.method}-X_cat.npy')).to(device)
            # attack
            if args.attack != 'none' and k == 'w':
                X_num, X_cat, mask_col = attack_numpy(args.attack, args.attack_percentage, X_num.cpu().numpy(), X_cat.cpu().numpy(), X_num_pre.cpu().numpy(), X_cat_pre.cpu().numpy(), args)
                X_num = torch.tensor(X_num).to(device)
                X_cat = torch.tensor(X_cat).to(device)
            metric = get_watermark_metric(args, dataname, save_path, save_dir, k, info, model, noise_scheduler, watermarking_mask, gt_patch, mean, latents, X_num, X_cat,k, mask_col)
            print(f'{k}:', metric)
            wandb.log({f'{k}':float(metric)})
        return X_num, X_cat
    if args.mode =='attack':
        w_path=f'{save_dir}/w-{args.method}.csv'
        no_w_path=f'{save_dir}/no-w-{args.method}.csv'

        if args.attack == 'rdeletion':
            k = f'w-rdel-{args.attack_percentage}'
            save_path = f'{save_dir}/{k}-{args.method}.csv'
            delete_random_rows(input_path=w_path, additional_path=no_w_path, output_path=save_path, delete_percentage=args.attack_percentage)
            metric = get_watermark_metric(args, dataname, save_path, save_dir, k, info, model, noise_scheduler, watermarking_mask, gt_patch)
            print(f'{k}:', metric)
            wandb.log({f'{k}':float(metric)})
        
        if args.attack == 'cdeletion':
            k = f'w-cdel-{args.attack_percentage}'
            save_path = f'{save_dir}/{k}-{args.method}.csv'
            delete_random_columns(input_path=w_path, additional_path=no_w_path, output_path=save_path, delete_percentage=args.attack_percentage)
            metric = get_watermark_metric(args, dataname, save_path, save_dir, k, info, model, noise_scheduler, watermarking_mask, gt_patch)
            print(f'{k}:', metric)
            wandb.log({f'{k}':float(metric)})

        if args.attack == 'vdeletion':
            k = f'w-vdel-{args.attack_percentage}'
            save_path = f'{save_dir}/{k}-{args.method}.csv'
            delete_random_values(input_path=w_path, additional_path=no_w_path, output_path=save_path, delete_percentage=args.attack_percentage)
            metric = get_watermark_metric(args, dataname, save_path, save_dir, k, info, model, noise_scheduler, watermarking_mask, gt_patch)
            print(f'{k}:', metric)
            wandb.log({f'{k}':float(metric)})
        
        if args.attack == 'rdeleLatent':
            k = f'w-rdeleLatent-{args.attack_percentage}'
            attacked_reversed_noise = delete_random_latent_rows(args, info, save_dir, model, noise_scheduler, delete_percentage=args.attack_percentage)
            metric = eval_watermark(attacked_reversed_noise, watermarking_mask, gt_patch, args)
            print(f'{k}:', metric)
            wandb.log({f'{k}':float(metric)})

        if args.attack == 'cdisLatent':
            k = f'w-cdeleLatent-{args.attack_percentage}'
            attacked_reversed_noise = delete_random_latent_columns(args, info, save_dir, model, noise_scheduler, delete_percentage=args.attack_percentage)
            metric = eval_watermark(attacked_reversed_noise, watermarking_mask, gt_patch, args)
            print(f'{k}:', metric)
            wandb.log({f'{k}':float(metric)})
        if args.attack == 'rdistortion':
            k = f'w-rdis-{args.attack_percentage}'
            save_path = f'{save_dir}/{k}-{args.method}.csv'
            distort_random_rows(input_path=w_path, output_path=save_path, delete_percentage=args.attack_percentage)
            metric = get_watermark_metric(args, dataname, save_path, save_dir, k, info, model, noise_scheduler, watermarking_mask, gt_patch)
            print(f'{k}:', metric)
            wandb.log({f'{k}':float(metric)})

        if args.attack == 'vdistortion':
            k = f'w-vdis-{args.attack_percentage}'
            save_path = f'{save_dir}/{k}-{args.method}.csv'
            distort_random_values(input_path=w_path, output_path=save_path, delete_percentage=args.attack_percentage)
            metric = get_watermark_metric(args, dataname, save_path, save_dir, k, info, model, noise_scheduler, watermarking_mask, gt_patch)
            print(f'{k}:', metric)
            wandb.log({f'{k}':float(metric)})
        
        if args.attack == 'cdistortion':
            k = f'w-cdis-{args.attack_percentage}'
            save_path = f'{save_dir}/{k}-{args.method}.csv'
            distort_random_columns(input_path=w_path, output_path=save_path, delete_percentage=args.attack_percentage)
            metric = get_watermark_metric(args, dataname, save_path, save_dir, k, info, model, noise_scheduler, watermarking_mask, gt_patch)
            print(f'{k}:', metric)
            wandb.log({f'{k}':float(metric)})

        if args.attack == 'rdisLatent':
            k = f'w-rdisLatent-{args.attack_percentage}'
            attacked_reversed_noise = distort_random_latent_rows(args, info, save_dir, model, noise_scheduler, delete_percentage=args.attack_percentage)
            metric = eval_watermark(attacked_reversed_noise, watermarking_mask, gt_patch, args)
            print(f'{k}:', metric)
            wandb.log({f'{k}':float(metric)})
        
        else:
            print('Attack not implemented')


if __name__ == '__main__':
    pass