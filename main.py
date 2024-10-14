import torch
import wandb
import numpy as np
import os
from utils import execute_function, get_args

if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    if not args.save_path:
        args.save_path = f'synthetic/{args.dataname}'

    if args.attack != 'none':
        run_name = f'{args.with_w}-{args.attack}-{args.attack_percentage}'
    else:
        run_name = f'{args.mode}-{args.with_w}-{args.w_radius}'

    # wandb initilizationf
    wandb.init(
        project=f'Watermark-Eval-{args.dataname}',
        name=run_name)
    main_fn = execute_function(args.method, args.mode)
    if args.mode in ['sample','detect']:
        # generating 100 tables and Evaluate
        main_fn(args, 100)
    else:
        main_fn(args)
