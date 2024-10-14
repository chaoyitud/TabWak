import os
import torch

import argparse
import warnings
import time
import copy
import numpy as np

import wandb

from tabsyn.model import MLPDiffusion, Model, DDIMModel, DDIMScheduler, BDIA_DDIMScheduler
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target, get_encoder_latent
from tabsyn.watermark_utils import get_watermarking_mask, inject_watermark, get_watermarking_pattern, eval_watermark
from tabsyn.process_syn_dataset import process_data, preprocess_syn

warnings.filterwarnings('ignore')


def main(args, i):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    save_path_arg = args.save_path
    # TREERING
    save_dir = f'{curr_dir}/../{save_path_arg}/treering/{i}'
    if not os.path.exists(save_dir):
        # If it doesn't exist, create it
        os.makedirs(save_dir, exist_ok=True)

    dataname = args.dataname
    device = args.device
    steps = args.steps
    
    with_w = args.with_w

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1] 

    mean = train_z.mean(0)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    
    # Score-based
    # model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    # DDIM
    model = DDIMModel(denoise_fn).to(device)

    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))

    '''
        Generating samples    
    '''
    start_time = time.time()

    num_samples = 100
    sample_dim = in_dim
    torch.manual_seed(i)
    init_latents = torch.randn([num_samples, sample_dim], device=device)

    latents_1 = init_latents.to(device)

    init_latents_2 = torch.randn([num_samples, sample_dim], device=device)
    latents_2 = init_latents_2.to(device)

    # distance between latents_1 and latents_2
    distance = torch.norm(latents_1 - latents_2)
    print(f'distance between latents_1 and latents_2: {distance}')
    wandb.log({'distance_two_random_latents': float(distance)}, step=i)
    # Score-based
    # x_next = sample(model.denoise_fn_D, latents, args=args)

    # DDIM no watermark
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    x_next_1 = noise_scheduler.generate(
            model.noise_fn,
            latents_1,
            num_inference_steps=steps,
            eta=0.0)
    # save x_next_1
    for i, x_next in enumerate(x_next_1):
        print()
        x_next = x_next * 2 + mean.to(device)
        syn_data = x_next.float().cpu().numpy()
        syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device)
        save_path = f'{save_dir}/animation-{i}.csv'
        syn_df = recover_data(syn_num, syn_cat, syn_target, info)

        idx_name_mapping = info['idx_name_mapping']
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
        syn_df.rename(columns=idx_name_mapping, inplace=True)
        syn_df.to_csv(save_path, index=False)

    end_time = time.time()
    print('Time:', end_time - start_time)
    print('Saving sampled data to {}'.format(save_path))

    x_next_1 = torch.stack(x_next_1).cpu().detach().numpy()
    # save x_next_1 to file
    np.save(f'{save_dir}/no-w-DDIM.npy', x_next_1)

    return
    # DDIM watermark
    BDIA_noise_scheduler = BDIA_DDIMScheduler(num_train_timesteps=1000)
    x_next_2, x_next_2_aux = BDIA_noise_scheduler.generate(
            model.noise_fn,
            latents_1,
            num_inference_steps=steps,
            eta=0.0)

    x_next_3 = noise_scheduler.generate(
            model.noise_fn,
            latents_2,
            num_inference_steps=steps,
            eta=0.0)
    # log the distance between x_next_1 and x_next_2
    distance = torch.norm(x_next_1 - x_next_2)
    print(f'distance between x_next_1 and x_next_2: {distance}')
    wandb.log({'distance': float(distance)}, step=i)
    # log the distance between x_next_1 and x_next_3
    distance = torch.norm(x_next_1 - x_next_3)
    print(f'distance between x_next_1 and x_next_3: {distance}')
    wandb.log({'distance_random': float(distance)}, step=i)
    # do inverse process
    recovered_latent_1 = noise_scheduler.gen_reverse(
            model.noise_fn,
            x_next_1,
            num_inference_steps=steps,
            eta=0.0
    )
    # get the distance between the original latent and the recovered latent
    distance_latent = torch.norm(latents_1 - recovered_latent_1)
    print(f'distance_latent_DDIM: {distance_latent}')
    wandb.log({'distance_latent_DDIM': float(distance_latent)}, step=i)
    recovered_latent_2 = BDIA_noise_scheduler.gen_reverse(
            model.noise_fn,
            x_next_2,
            x_next_2_aux,
            num_inference_steps=steps,
            eta=0.0
    )
    # get the distance between the original latent and the recovered latent
    distance_latent = torch.norm(latents_1 - recovered_latent_2)
    print(f'distance_latent_BDIA: {distance_latent}')
    wandb.log({'distance_latent_BDIA': float(distance_latent)}, step=i)

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

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            t_scale = t / schedulers[0].num_train_timesteps

            for latent_i in range(2):
                if run_baseline and latent_i == 1: continue  # just have one sequence for baseline

                latent_model_input = new_latent
                if reverse:
                    current_timestep = t - schedulers[latent_i].config.num_train_timesteps / schedulers[
                        latent_i].num_inference_steps
                else:
                    current_timestep = t
                # Predict the unconditional noise residual
                noise_pred_uncond = unet(latent_model_input, current_timestep,
                                         encoder_hidden_states=embedding_unconditional).sample

                # Prepare the Cross-Attention layers
                if prompt_edit is not None:
                    save_last_tokens_attention()
                    save_last_self_attention()
                else:
                    # Use weights on non-edited prompt when edit is None
                    use_last_tokens_attention_weights()

                # Predict the conditional noise residual and save the cross-attention layer activations
                noise_pred_cond = unet(latent_model_input, current_timestep,
                                       encoder_hidden_states=embedding_conditional).sample

                # Edit the Cross-Attention layer activations
                if prompt_edit is not None:
                    t_scale = t / schedulers[0].num_train_timesteps
                    if t_scale >= prompt_edit_tokens_start and t_scale <= prompt_edit_tokens_end:
                        use_last_tokens_attention()
                    if t_scale >= prompt_edit_spatial_start and t_scale <= prompt_edit_spatial_end:
                        use_last_self_attention()

                    # Use weights on edited prompt
                    use_last_tokens_attention_weights()

                    # Predict the edited conditional noise residual using the cross-attention masks
                    noise_pred_cond = unet(latent_model_input,
                                           t,
                                           encoder_hidden_states=embedding_conditional_edit).sample

                # Perform guidance
                grad = (noise_pred_cond - noise_pred_uncond)
                noise_pred = noise_pred_uncond + guidance_scale * grad

                if reverse and t_prev == None:
                    step_call = reverse_step if reverse else forward_step
                    # new_latent = step_call(schedulers[latent_i],
                    #                          noise_pred,
                    #                            t,
                    #                            latent_base)# .prev_sample
                    current_timestep = t - schedulers[latent_i].config.num_train_timesteps / schedulers[
                        latent_i].num_inference_steps

                    alpha_prod_t_next, beta_prod_t_next = get_alpha_and_beta(t, schedulers[latent_i])
                    alpha_prod_t, beta_prod_t = get_alpha_and_beta(current_timestep, schedulers[latent_i])
                    alpha_quotient = ((alpha_prod_t_next / alpha_prod_t) ** 0.5)
                    first_term = alpha_quotient * latent_model_input
                    second_term = ((beta_prod_t_next) ** 0.5) * noise_pred
                    third_term = alpha_quotient * ((1 - alpha_prod_t) ** 0.5) * noise_pred

                    new_latent = first_term + second_term - third_term
                    new_latent = new_latent.to(latent_model_input.dtype)
                    t_prev = current_timestep
                    latent_prev = latent_model_input
                elif reverse and t_prev != None:
                    current_timestep = t - schedulers[latent_i].config.num_train_timesteps / schedulers[
                        latent_i].num_inference_steps

                    step_call = reverse_step if reverse else forward_step
                    new_latent = step_call(schedulers[latent_i],
                                           noise_pred,
                                           t,
                                           latent_model_input)  # .prev_sample
                    new_latent = new_latent.to(latent_model_input.dtype)
                    latent_backward = forward_step(schedulers[latent_i],
                                                   noise_pred,
                                                   current_timestep,
                                                   latent_model_input)  # .prev_sample
                    latent_backward = latent_backward.to(latent_model_input.dtype)
                    '''
                    new_latent = (latent_prev 
                                  -(1-gamma)*(latent_prev-latent_model_input) 
                                  - gamma*(latent_backward-latent_model_input)
                                  + (new_latent-latent_model_input) 
                              )
                    '''
                    new_latent = (latent_prev / gamma
                                  - latent_backward / gamma
                                  + new_latent
                                  )

                    t_prev = current_timestep
                    latent_prev = latent_model_input

                elif not reverse and t_prev == None:
                    step_call = reverse_step if reverse else forward_step
                    new_latent = step_call(schedulers[latent_i],
                                           noise_pred,
                                           t,
                                           latent_model_input)  # .prev_sample
                    new_latent = new_latent.to(latent_model_input.dtype)
                    # current_timestep = t - schedulers[latent_i].config.num_train_timesteps / schedulers[latent_i].num_inference_steps
                    t_prev = t
                    latent_prev = latent_model_input
                elif not reverse and t_prev != None:
                    # prev_timestep = t + schedulers[latent_i].config.num_train_timesteps / schedulers[latent_i].num_inference_steps

                    step_call = reverse_step if reverse else forward_step
                    new_latent = step_call(schedulers[latent_i],
                                           noise_pred,
                                           t,
                                           latent_model_input)  # .prev_sample
                    new_latent = new_latent.to(latent_model_input.dtype)
                    # latent_backward = reverse_step(schedulers[latent_i],
                    #                            noise_pred,
                    #                            t_prev,
                    #                            latent_base)# .prev_sample
                    alpha_prod_t_prev, beta_prod_t_prev = get_alpha_and_beta(t_prev, schedulers[latent_i])
                    alpha_prod_t, beta_prod_t = get_alpha_and_beta(t, schedulers[latent_i])
                    alpha_quotient = ((alpha_prod_t_prev / alpha_prod_t) ** 0.5)
                    first_term = alpha_quotient * latent_model_input
                    second_term = ((beta_prod_t_prev) ** 0.5) * noise_pred
                    third_term = alpha_quotient * ((1 - alpha_prod_t) ** 0.5) * noise_pred
                    latent_backward = first_term + second_term - third_term
                    latent_backward = latent_backward.to(latent_model_input.dtype)

                    new_latent = (latent_prev
                                  - (1 - gamma) * (latent_prev - latent_model_input)
                                  - gamma * (latent_backward - latent_model_input)
                                  + (new_latent - latent_model_input)
                                  )
                    '''
                    new_latent = (latent_prev/gamma 
                                  - latent_backward/gamma
                                  + new_latent
                              )
                    '''
                    t_prev = t
                    latent_prev = latent_model_input

                latent_pair[latent_i] = new_latent

            if (not reverse) and (not run_baseline):
                # Mixing layer (contraction) during generative process
                new_latents = [l.clone() for l in latent_pair]
                new_latents[0] = (mix_weight * new_latents[0] + (1 - mix_weight) * new_latents[1]).clone()
                new_latents[1] = ((1 - mix_weight) * new_latents[0] + (mix_weight) * new_latents[1]).clone()
                latent_pair = new_latents

        latent_pair[0] = new_latent
        latent_pair[1] = latent_model_input