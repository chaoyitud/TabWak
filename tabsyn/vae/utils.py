import os
import numpy as np

if __name__ == '__main__':
    for dataset in ['shoppers', 'magic', 'default', 'adult']:
        path = f'./ckpt/{dataset}/train_z.npy'
        if os.path.exists(path):
            train_z = np.load(path)
            print(f"{dataset}: ", train_z.shape)
        else:
            print('File does not exist')