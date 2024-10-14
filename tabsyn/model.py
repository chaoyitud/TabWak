from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tabsyn.diffusion_utils import EDMLoss
from tqdm import tqdm

ModuleType = Union[str, Callable[..., nn.Module]]


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t=512):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

    def forward(self, x, noise_labels, class_labels=None):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        emb = self.time_embed(emb)

        x = self.proj(x) + emb
        return self.mlp(x)


class Precond(nn.Module):
    def __init__(self,
                 denoise_fn,
                 hid_dim,
                 sigma_min=0,  # Minimum supported noise level.
                 sigma_max=float('inf'),  # Maximum supported noise level.
                 sigma_data=0.5,  # Expected standard deviation of the training data.
                 ):
        super().__init__()

        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        ###########
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma):
        x = x.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.denoise_fn_F((x_in).to(dtype), c_noise.flatten())

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class Model(nn.Module):
    def __init__(self, denoise_fn, hid_dim, P_mean=-1.2, P_std=1.2, sigma_data=0.5, gamma=5, opts=None, pfgmpp=False):
        super().__init__()

        self.denoise_fn_D = Precond(denoise_fn, hid_dim)
        self.loss_fn = EDMLoss(P_mean, P_std, sigma_data, hid_dim=hid_dim, gamma=5, opts=None)

    def forward(self, x):
        loss = self.loss_fn(self.denoise_fn_D, x)
        return loss.mean(-1).mean()


############################### DDIM ###############################
def cosine_beta_schedule(timesteps, beta_start=0.0, beta_end=0.999, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, beta_start, beta_end)


class DDIMScheduler:
    def __init__(self,
                 num_train_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 clip_sample=True,
                 set_alpha_to_one=True):

        self.betas = cosine_beta_schedule(num_train_timesteps,
                                          beta_start=beta_start,
                                          beta_end=beta_end)

        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        self.final_alpha_cumprod = np.array(
            1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy()

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev /
                    beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def set_timesteps(self, num_inference_steps, offset=0):
        self.num_inference_steps = num_inference_steps
        self.timesteps = np.arange(0, 1000,
                                   1000 // num_inference_steps)[::-1].copy()
        self.timesteps += offset

    def step(
            self,
            model_output: Union[torch.FloatTensor, np.ndarray],
            timestep: int,
            sample: Union[torch.FloatTensor, np.ndarray],
            eta: float = 0.0,
            generator=None,
    ):
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (sample - beta_prod_t **
                                (0.5) * model_output) / alpha_prod_t ** (0.5)

        # 5/6. compute "direction pointing to x_t" without additional noise
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * model_output

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (
            0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            device = model_output.device if torch.is_tensor(
                model_output) else "cpu"
            noise = torch.randn(model_output.shape,
                                generator=generator).to(device)
            variance = self._get_variance(timestep,
                                          prev_timestep) ** (0.5) * eta * noise

            if not torch.is_tensor(model_output):
                variance = variance.numpy()

            prev_sample = prev_sample + variance

        return prev_sample

    def inverse_step(
            self,
            model_output: Union[torch.FloatTensor, np.ndarray],
            timestep: int,
            sample: Union[torch.FloatTensor, np.ndarray],
    ):

        # 1. get current step value (=t+1)
        current_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        next_timestep = timestep

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[current_timestep]
        alpha_prod_t_next = self.alphas_cumprod[
            next_timestep] if next_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # 3. Inverted update step 
        # re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (sample - beta_prod_t.sqrt() * model_output) * (alpha_prod_t_next.sqrt() / alpha_prod_t.sqrt()) + (
                    1 - alpha_prod_t_next).sqrt() * model_output

        return latents

    def add_noise(self, original_samples, noise, timesteps):
        timesteps = timesteps.cpu()
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = match_shape(sqrt_alpha_prod, original_samples)
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = match_shape(sqrt_one_minus_alpha_prod,
                                                original_samples)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    @torch.no_grad()
    def generate(self,
                 model,
                 latents,
                 eta=0.0,
                 num_inference_steps=50,
                 device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        x_next = latents.to(device)
        N = x_next.shape[0]
        self.set_timesteps(num_inference_steps)
        x_nexts = [x_next]
        for t in tqdm(self.timesteps):
            # 1. predict noise model_output
            timesteps = torch.full((N,), t, dtype=torch.long, device=device)
            model_output = model(x_next, timesteps)

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta = 0.0 for deterministic ddim
            # do x_t -> x_t-1
            x_next = self.step(model_output,
                               t,
                               x_next,
                               eta)
            # save x_next for each timestep
            x_nexts.append(x_next)
        return x_next

    @torch.no_grad()
    def gen_reverse(self,
                    model,
                    latents,
                    eta=0.0,
                    num_inference_steps=50,
                    device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        x_next = latents.to(device)
        #mean = x_next.mean(0)
        #x_next = x_next / 2 - mean
        N = x_next.shape[0]
        self.set_timesteps(num_inference_steps)
        reverse_timesteps = self.timesteps[::-1]

        for t in tqdm(reverse_timesteps[1:]):
            # 1. predict noise model_output
            timesteps = torch.full((N,), t, dtype=torch.long, device=device)
            model_output = model(x_next, timesteps)

            # 2. predict previous mean of noisy image x_t and eta = 0.0 for deterministic ddim
            # do x_t-1 -> x_t
            x_next = self.inverse_step(model_output,
                                       t,
                                       x_next)

        return x_next

    def __len__(self):
        return self.num_train_timesteps


class BDIA_DDIMScheduler:
    def __init__(self,
                 num_train_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 clip_sample=True,
                 set_alpha_to_one=True):

        self.betas = cosine_beta_schedule(num_train_timesteps,
                                          beta_start=beta_start,
                                          beta_end=beta_end)

        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        self.final_alpha_cumprod = np.array(
            1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy()

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev /
                    beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def set_timesteps(self, num_inference_steps, offset=0):
        self.num_inference_steps = num_inference_steps
        self.timesteps = np.arange(0, 1000,
                                   1000 // num_inference_steps)[::-1].copy()
        self.timesteps += offset

    def inverse_step(
            self,
            model_output: Union[torch.FloatTensor, np.ndarray],
            timestep: int,
            sample: Union[torch.FloatTensor, np.ndarray],
    ):

        # 1. get current step value (=t+1)
        current_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        next_timestep = timestep

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[current_timestep]
        alpha_prod_t_next = self.alphas_cumprod[
            next_timestep] if next_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # 3. Inverted update step
        # re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (sample - beta_prod_t.sqrt() * model_output) * (alpha_prod_t_next.sqrt() / alpha_prod_t.sqrt()) + (
                1 - alpha_prod_t_next).sqrt() * model_output

        return latents

    def step(
            self,
            model_output: Union[torch.FloatTensor, np.ndarray],
            timestep: int,
            sample: Union[torch.FloatTensor, np.ndarray, List],
            eta: float = 0.0,
            generator=None,
    ):
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # 5/6. compute "direction pointing to x_t" without additional noise
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        # Calculate coefficients a and b
        a = (alpha_prod_t_prev ** 0.5) / (alpha_prod_t ** 0.5)
        b = (1 - alpha_prod_t_prev) ** 0.5 - ((alpha_prod_t_prev ** 0.5) * (beta_prod_t ** 0.5)) / (alpha_prod_t ** 0.5)

        prev_sample_alt = a * sample + b * model_output
        return prev_sample
    def step_BDIA(
            self,
            model_output: Union[torch.FloatTensor, np.ndarray],
            timestep: int,
            sample: Union[torch.FloatTensor, np.ndarray, List],
            eta: float = 0.0,
            generator=None,
    ):
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # Calculate coefficients a and b
        a_t = (alpha_prod_t_prev ** 0.5) / (alpha_prod_t ** 0.5)
        b_t = (1 - alpha_prod_t_prev) ** 0.5 - ((alpha_prod_t_prev ** 0.5) * (beta_prod_t ** 0.5)) / (alpha_prod_t ** 0.5)

        next_timestep = timestep + self.num_train_timesteps // self.num_inference_steps
        alpha_prod_t_next = self.alphas_cumprod[next_timestep]
        beta_prod_t_next = 1 - alpha_prod_t_next

        a_t_plus_1 = (alpha_prod_t ** 0.5) / (alpha_prod_t_next ** 0.5)
        b_t_plus_1 = (1 - alpha_prod_t) ** 0.5 - ((alpha_prod_t ** 0.5) * (beta_prod_t_next ** 0.5)) / (alpha_prod_t_next ** 0.5)

        first_term = sample[0]
        second_term = (1/a_t_plus_1-1) * sample[1] - b_t_plus_1/a_t_plus_1 * model_output
        third_term = (a_t-1)*sample[1] + b_t*model_output
        prev_sample = first_term - second_term + third_term
        return prev_sample

    def reverse_step_BDIA(
            self,
            model_output: Union[torch.FloatTensor, np.ndarray],
            timestep: int,
            sample: Union[torch.FloatTensor, np.ndarray, List],
            eta: float = 0.0,
            generator=None,
    ):
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # Calculate coefficients a and b
        a_t = (alpha_prod_t_prev ** 0.5) / (alpha_prod_t ** 0.5)
        b_t = (1 - alpha_prod_t_prev) ** 0.5 - ((alpha_prod_t_prev ** 0.5) * (beta_prod_t ** 0.5)) / (
                    alpha_prod_t ** 0.5)

        next_timestep = timestep + self.num_train_timesteps // self.num_inference_steps
        alpha_prod_t_next = self.alphas_cumprod[next_timestep]
        beta_prod_t_next = 1 - alpha_prod_t_next

        a_t_plus_1 = (alpha_prod_t ** 0.5) / (alpha_prod_t_next ** 0.5)
        b_t_plus_1 = (1 - alpha_prod_t) ** 0.5 - ((alpha_prod_t ** 0.5) * (beta_prod_t_next ** 0.5)) / (
                    alpha_prod_t_next ** 0.5)

        first_term = sample[0]
        second_term = a_t*sample[1] + b_t*model_output
        third_term = (1/a_t_plus_1)*sample[1] - b_t_plus_1/a_t_plus_1 * model_output
        next_sample = first_term - second_term + third_term
        return next_sample

    def add_noise(self, original_samples, noise, timesteps):
        timesteps = timesteps.cpu()
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = match_shape(sqrt_alpha_prod, original_samples)
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = match_shape(sqrt_one_minus_alpha_prod,
                                                original_samples)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    @torch.no_grad()
    def generate(self,
                 model,
                 latents,
                 eta=0.0,
                 num_inference_steps=50,
                 device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        x_next = latents.to(device)
        N = x_next.shape[0]
        x_pairs = [x_next, x_next]  # [x_t+1, x_t]
        self.set_timesteps(num_inference_steps)
        for t in tqdm(self.timesteps):
            timesteps = torch.full((N,), t, dtype=torch.long, device=device)
            model_output = model(x_pairs[1], timesteps)
            if t == self.timesteps[0]:  # t = T [x_T, x_T-1]
                x_T_minus_1 = self.step(model_output, t, x_pairs[0], eta)
                x_pairs[1] = x_T_minus_1
            else:
                x_t_minus_1 = self.step_BDIA(model_output, t, x_pairs)
                x_pairs[0] = x_pairs[1]
                x_pairs[1] = x_t_minus_1
        return x_pairs[0]

    @torch.no_grad()
    def gen_reverse(self,
                    model,
                    latents,
                    eta=0.0,
                    num_inference_steps=50,
                    device=None,
                    latents_aux=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        x_next = latents.to(device)
        #mean = x_next.mean(0)
        #x_next = x_next / 2 - mean
        if latents_aux is None:
            latents_aux = x_next
        N = x_next.shape[0]
        self.set_timesteps(num_inference_steps)
        reverse_timesteps = self.timesteps[::-1]
        x_pairs = [x_next, x_next]  # [x_t-1, x_t]
        for t in tqdm(reverse_timesteps[:-1]):
            timesteps = torch.full((N,), t, dtype=torch.long, device=device)
            if t == reverse_timesteps[0]:
                x_pairs = [latents_aux, x_next]
            else:
                # t = 2, x0, x1
                model_output = model(x_pairs[1], timesteps)
                x_t_plus_1 = self.reverse_step_BDIA(model_output, t, x_pairs)
                x_pairs[0] = x_pairs[1]
                x_pairs[1] = x_t_plus_1
        return x_pairs[1]

    def __len__(self):
        return self.num_train_timesteps


class DDIMModel(nn.Module):
    def __init__(self, noise_fn):
        super().__init__()

        self.noise_fn = noise_fn

    def forward(self, noise, noisy_images, timesteps):
        noise_pred = self.noise_fn(noisy_images, timesteps)
        loss = F.l1_loss(noise_pred, noise)

        return loss.mean(-1).mean()


################### DDIM utils ###################

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def match_shape(values, broadcast_array, tensor_format="pt"):
    values = values.flatten()

    while len(values.shape) < len(broadcast_array.shape):
        values = values[..., None]
    if tensor_format == "pt":
        values = values.to(broadcast_array.device)

    return values


def clip(tensor, min_value=None, max_value=None):
    if isinstance(tensor, np.ndarray):
        return np.clip(tensor, min_value, max_value)
    elif isinstance(tensor, torch.Tensor):
        return torch.clamp(tensor, min_value, max_value)

    raise ValueError("Tensor format is not valid is not valid - " \
                     f"should be numpy array or torch tensor. Got {type(tensor)}.")
