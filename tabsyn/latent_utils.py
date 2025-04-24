import os
import json
import numpy as np
import pandas as pd
import torch
from torch import nn

from utils_train import preprocess
from tabsyn.vae.model import Decoder_model, Encoder_model, Model_VAE
import wandb

def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mask_col=None):
    ce_loss_fn = nn.CrossEntropyLoss()
    if mask_col is None:
        mse_loss = (X_num - Recon_X_num).pow(2).mean()
    else:
        # remove the masked columns to calculate the mse loss
        non_mask_col = [i for i in range(X_num.size(1)) if i not in mask_col]
        mse_loss = (X_num[:, non_mask_col] - Recon_X_num[:, non_mask_col]).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim=-1)
        acc += (x_hat == X_cat[:, idx]).float().sum()
        total_num += x_hat.shape[0]
    wandb.log({'acc': acc.item() / total_num})
    ce_loss /= (idx + 1)
    wandb.log({'ce_loss': ce_loss})
    wandb.log({'mse_loss': mse_loss.item()})
    loss = mse_loss + ce_loss
    return loss

def get_decoder_latent_train_VAE(X_num, X_cat, info, device, mask_col=None):

    with torch.no_grad():
        vae_mdeol = info['model'].to(device).eval()
        X_num = torch.tensor(X_num).float().to(device)
        X_cat = torch.tensor(X_cat).to(device)
        _, mu_z, logvar_z = vae_mdeol.VAE(X_num, X_cat)
        # get latent shape
        latent = mu_z[:, 1:, :].clone().detach()
        mu_z = mu_z[:, 1:, :].detach()
        logvar_z = logvar_z[:, 1:, :].detach()
        std_z = torch.exp(0.5 * logvar_z)

    latent.requires_grad = True
    optimizer = torch.optim.AdamW([latent], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)
    decoder = info['pre_decoder'].to(device).eval()
    for i in range(1000):
        optimizer.zero_grad()
        recon = decoder(latent)
        loss = compute_loss(X_num, X_cat, recon[0], recon[1], mask_col=mask_col)

        # Regularization term to enforce mean and std
        #mean_loss = torch.nn.functional.mse_loss(latent.mean(0), mean.to(device))
        #std_loss = torch.nn.functional.mse_loss(latent.std(0), std.to(device))
        mean_loss = torch.nn.functional.mse_loss(latent, mu_z)

        punish_loss = torch.mean(torch.relu(latent - (mu_z + 3*std_z))) + torch.mean(torch.relu((mu_z - 3*std_z) - latent))

        kl_loss = -0.5 * torch.mean(1 + 2*logvar_z - latent**2 - std_z**2)

        wandb.log({'mean_loss': mean_loss.item()})

        wandb.log({'loss_kld': kl_loss.item()})

        loss += punish_loss
        wandb.log({'loss': loss.item()})
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
    return latent

def get_decoder_latent_train(X_num, X_cat, info, device, aux=None, mask_col=None):
    # get the encoder latent at first step
    aux = None
    with torch.no_grad():
        pre_encoder = info['pre_encoder'].to(device)
        X_num = torch.tensor(X_num).float().to(device)
        X_cat = torch.tensor(X_cat).to(device)
        latent = pre_encoder(X_num, X_cat)
        # get latent shape
        latent = latent[:, 1:, :]
        ("latent shape", latent.size())
    latent.requires_grad = True
    optimizer = torch.optim.Adam([latent], lr=1e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    decoder = info['pre_decoder'].to(device)
    for i in range(1000):
        optimizer.zero_grad()
        recon = decoder(latent)
        loss = compute_loss(X_num, X_cat, recon[0], recon[1], mask_col=mask_col)
        wandb.log({'loss': loss.item()})
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
    return latent


def get_encoder_latent(X_num, X_cat, info, device):

    pre_encoder = info['pre_encoder'].to(device)
    X_num = torch.tensor(X_num).float().to(device)
    X_cat = torch.tensor(X_cat).to(device)

    latent = pre_encoder(X_num, X_cat)

    latent = latent[:, 1:, :]

    B, num_tokens, token_dim = latent.size()
    in_dim = num_tokens * token_dim
    
    latent = latent.view(B, in_dim)

    return latent

def get_decoder_latent(X_num, X_cat, info, device, aux=None, mask_col=None, mean=None, std=None):
    latent = get_decoder_latent_train_VAE(X_num, X_cat, info, device, mask_col=mask_col)
    #latent = latent[:, , :]

    B, num_tokens, token_dim = latent.size()
    in_dim = num_tokens * token_dim

    latent = latent.view(B, in_dim)

    return latent

    

def get_input_train(args):
    dataname = args.dataname

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # dataset_dir = f'data/{dataname}'
    dataset_dir = f'{curr_dir}/../data/{dataname}/'

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    ckpt_dir = f'{curr_dir}/ckpt/{dataname}/'
    embedding_save_path = f'{curr_dir}/vae/ckpt/{dataname}/train_z.npy'
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    
    train_z = train_z.view(B, in_dim)

    return train_z, curr_dir, dataset_dir, ckpt_dir, info


def get_input_generate(args, get_d_num=False):
    dataname = args.dataname

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = f'{curr_dir}/../data/{dataname}/'

    ckpt_dir = f'{curr_dir}/ckpt/{dataname}'

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']

    _, _, categories, d_numerical, num_inverse, cat_inverse = preprocess(dataset_dir, task_type=task_type, inverse=True)

    embedding_save_path = f'{curr_dir}/vae/ckpt/{dataname}/train_z.npy'
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :]

    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim

    train_z = train_z.view(B, in_dim)
    pre_decoder = Decoder_model(2, d_numerical, categories, 4, n_head=1, factor=32)
    pre_encoder = Encoder_model(2, d_numerical, categories, 4, n_head=1, factor=32)
    model = Model_VAE(2, d_numerical, categories, 4, n_head=1, factor=32)
    decoder_save_path = f'{curr_dir}/vae/ckpt/{dataname}/decoder.pt'
    pre_decoder.load_state_dict(torch.load(decoder_save_path))
    encoder_save_path = f'{curr_dir}/vae/ckpt/{dataname}/encoder.pt'
    pre_encoder.load_state_dict(torch.load(encoder_save_path))
    model_save_path = f'{curr_dir}/vae/ckpt/{dataname}/model.pt'
    model.load_state_dict(torch.load(model_save_path))

    info['pre_decoder'] = pre_decoder
    info['pre_encoder'] = pre_encoder
    info['token_dim'] = token_dim
    info['model'] = model
    if get_d_num:
        return train_z, curr_dir, dataset_dir, ckpt_dir, info, num_inverse, cat_inverse, d_numerical
    return train_z, curr_dir, dataset_dir, ckpt_dir, info, num_inverse, cat_inverse


 
@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse, device):
    task_type = info['task_type']

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == 'regression':
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

  
    pre_decoder = info['pre_decoder']
    token_dim = info['token_dim']

    syn_data = syn_data.reshape(syn_data.shape[0], -1, token_dim)
    
    norm_input = pre_decoder(torch.tensor(syn_data))
    x_hat_num, x_hat_cat = norm_input

    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim = -1))

    syn_num = x_hat_num.cpu().numpy()
    syn_cat = torch.stack(syn_cat).t().cpu().numpy()

    syn_num = num_inverse(syn_num)
    syn_cat = cat_inverse(syn_cat)

    if info['task_type'] == 'regression':
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):]
    
    else:
        syn_target = syn_cat[:, :len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx):]

    return syn_num, syn_cat, syn_target

def recover_data(syn_num, syn_cat, syn_target, info):

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']


    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info['task_type'] == 'regression':
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]] 
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]


    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df
    

def process_invalid_id(syn_cat, min_cat, max_cat):
    syn_cat = np.clip(syn_cat, min_cat, max_cat)

    return syn_cat

