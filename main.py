import hashlib
import time

import torch
import wandb
import numpy as np
import os
from utils import execute_function, get_args
from tqdm import tqdm

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
        run_name = f'{args.mode}-{args.with_w}'

    wandb.init(
        project=f'Watermark-Eval-{args.dataname}',
        name=run_name)

    if args.exp_prefix is None:
        unique_str = str(time.time()).encode('utf-8')
        args.exp_prefix = hashlib.md5(unique_str).hexdigest()[:8]
        print("Current experiment ID:", args.exp_prefix)

    main_fn = execute_function(args.method, args.mode)
    if args.mode in ['sample', 'detect']:
        for i in tqdm(range(args.num_trials), desc="Running trials"):
            main_fn(args, i)
    else:
        main_fn(args)
