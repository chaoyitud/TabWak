import numpy as np
import pandas as pd
import os
import src
import sys
import json
import argparse
import sklearn
from sklearn.pipeline import make_pipeline
from category_encoders import LeaveOneOutEncoder
from copy import deepcopy
from typing import cast

TYPE_TRANSFORM ={
    'float', np.float32,
    'str', str,
    'int', int
}
curr_dir = os.path.dirname(os.path.abspath(__file__))
INFO_PATH = f'{curr_dir}/../data'
CAT_MISSING_VALUE = 'nan'
CAT_RARE_VALUE = '__rare__'


def preprocess_beijing():
    with open(f'{INFO_PATH}/beijing.json', 'r') as f:
        info = json.load(f)
    
    data_path = info['raw_data_path']

    data_df = pd.read_csv(data_path)
    columns = data_df.columns

    data_df = data_df[columns[1:]]


    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(info['data_path'], index = False)

def preprocess_news():
    with open(f'{INFO_PATH}/news.json', 'r') as f:
        info = json.load(f)

    data_path = info['raw_data_path']
    data_df = pd.read_csv(data_path)
    data_df = data_df.drop('url', axis=1)

    columns = np.array(data_df.columns.tolist())

    cat_columns1 = columns[list(range(12,18))]
    cat_columns2 = columns[list(range(30,38))]

    cat_col1 = data_df[cat_columns1].astype(int).to_numpy().argmax(axis = 1)
    cat_col2 = data_df[cat_columns2].astype(int).to_numpy().argmax(axis = 1)

    data_df = data_df.drop(cat_columns2, axis=1)
    data_df = data_df.drop(cat_columns1, axis=1)

    data_df['data_channel'] = cat_col1
    data_df['weekday'] = cat_col2
    
    data_save_path = 'data/news/news.csv'
    data_df.to_csv(f'{data_save_path}', index = False)

    columns = np.array(data_df.columns.tolist())
    num_columns = columns[list(range(45))]
    cat_columns = ['data_channel', 'weekday']
    target_columns = columns[[45]]

    info['num_col_idx'] = list(range(45))
    info['cat_col_idx'] = [46, 47]
    info['target_col_idx'] = [45]
    info['data_path'] = data_save_path
    
    name = 'news'
    with open(f'{INFO_PATH}/{name}.json', 'w') as file:
        json.dump(info, file, indent=4)


def get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names = None):
    
    if not column_names:
        column_names = np.array(data_df.columns.tolist())
    

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):

        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1


    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k
        
    idx_name_mapping = {}
    
    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping



def process_data(name, data_path, save_dir, k):

    if name == 'news':
        preprocess_news()
    elif name == 'beijing':
        preprocess_beijing()

    with open(f'{INFO_PATH}/{name}/info.json', 'r') as f:
        info = json.load(f)

    data_df = pd.read_csv(data_path, header = info['header'])

    num_data = data_df.shape[0]

    column_names = info['column_names'] if info['column_names'] else data_df.columns.tolist()
 
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names)

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]
    print(data_df.shape)

    data_df.rename(columns = idx_name_mapping, inplace=True)

    for col in num_columns:
        data_df.loc[data_df[col] == '?', col] = np.nan
    for col in cat_columns:
        data_df.loc[data_df[col] == '?', col] = 'nan'

    X_num_train = data_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = data_df[cat_columns].to_numpy()
    y_train = data_df[target_columns].to_numpy()

    np.save(f'{save_dir}/{k}-X_num.npy', X_num_train)
    np.save(f'{save_dir}/{k}-X_cat.npy', X_cat_train)
    np.save(f'{save_dir}/{k}-y.npy', y_train)

    print('Numerical', X_num_train.shape)
    print('Categorical', X_cat_train.shape)


def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def make_dataset(
    data_path: str,
    T: src.Transformations,
    task_type,
    concat = True,
    k=''
):  
    X_cat = {} if os.path.exists(os.path.join(data_path, f'{k}-X_cat.npy'))  else None
    X_num = {} if os.path.exists(os.path.join(data_path, f'{k}-X_num.npy')) else None
    y = {} if os.path.exists(os.path.join(data_path, f'{k}-y.npy')) else None

    X_num_t = np.load(os.path.join(data_path, f'{k}-X_num.npy'), allow_pickle=True)
    X_cat_t = np.load(os.path.join(data_path, f'{k}-X_cat.npy'), allow_pickle=True)
    y_t = np.load(os.path.join(data_path, f'{k}-y.npy'), allow_pickle=True)

    # classification
    if task_type == 'binclass' or task_type == 'multiclass':
        if X_num is not None:
                X_num = X_num_t
        if X_cat is not None:
            if concat:
                X_cat_t = concat_y_to_X(X_cat_t, y_t)
            X_cat = X_cat_t
    else:
        # regression
        if X_num is not None:
            if concat:
                X_num_t = concat_y_to_X(X_num_t, y_t)
            X_num = X_num_t
        if X_cat is not None:
            X_cat = X_cat_t
        

    return X_num, X_cat
        

def preprocess_syn(dataset_path, task_type = 'binclass', k='', cat_encoding = None, concat = True):
    
    T_dict = {}

    T_dict['normalization'] = "quantile"
    T_dict['num_nan_policy'] = 'mean'
    T_dict['cat_nan_policy'] =  None
    T_dict['cat_min_frequency'] = None
    T_dict['cat_encoding'] = cat_encoding
    T_dict['y_policy'] = "default"

    T = src.Transformations(**T_dict)

    X_num_np, X_cat_np = make_dataset(
        data_path = dataset_path,
        T = T,
        task_type = task_type,
        concat = concat,
        k=k
    )

    X_num = {'train': X_num_np}
    X_cat = {'train': X_cat_np}

    if X_num is not None:
        X_num = num_process_nans(X_num, T.num_nan_policy)
    
    if X_num is not None and T.normalization is not None:
        X_num, num_transform = normalize(
            X_num,
            T.normalization,
            T.seed,
            return_normalizer=True
        )

    if X_cat is None:
        assert T.cat_nan_policy is None
        assert T.cat_min_frequency is None
        # assert transformations.cat_encoding is None
        X_cat = None
    else:
        X_cat = cat_process_nans(X_cat, T.cat_nan_policy)
   
        X_cat, is_num, cat_transform = cat_encode(
            X_cat,
            T.cat_encoding,
            return_encoder=True
        )

    return X_num['train'], X_cat['train']


def num_process_nans(X_num, policy):
    assert X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in X_num.items()}
    if not any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        # assert policy is None
        print('No NaNs in numerical features, skipping')
        return X_num

    assert policy is not None

    if policy == 'mean':
        new_values = np.nanmean(X_num, axis=0)
        X_num = deepcopy(X_num)
        for k, v in X_num.items():
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        
    return X_num

def normalize(X, normalization, seed, return_normalizer=False):
    X_train = X['train']
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'minmax':
        normalizer = sklearn.preprocessing.MinMaxScaler()
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X['train'].shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
    else:
        print('Unkown normalization', normalization)

    normalizer.fit(X_train)
    if return_normalizer:
        return {k: normalizer.transform(v) for k, v in X.items()}, normalizer
    return {k: normalizer.transform(v) for k, v in X.items()}

def cat_process_nans(X, policy):
    assert X is not None
    nan_masks = {k: v == CAT_MISSING_VALUE for k, v in X.items()}
    if any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        if policy is None:
            X_new = X
        elif policy == 'most_frequent':
            imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy)  # type: ignore[code]
            imputer.fit(X['train'])
            X_new = {k: cast(np.ndarray, imputer.transform(v)) for k, v in X.items()}
        else:
            print('Unknown categorical NaN policy', policy)
    else:
        assert policy is None
        X_new = X
    return X_new

def cat_encode(X, encoding, return_encoder=False):  # (X, is_converted_to_numerical)

    if encoding is None:
        unknown_value = np.iinfo('int64').max - 3
        oe = sklearn.preprocessing.OrdinalEncoder(
            handle_unknown='use_encoded_value',  # type: ignore[code]
            unknown_value=unknown_value,  # type: ignore[code]
            dtype='int64',  # type: ignore[code]
        ).fit(X['train'])
        encoder = make_pipeline(oe)
        encoder.fit(X['train'])
        X = {k: encoder.transform(v) for k, v in X.items()}
        max_values = X['train'].max(axis=0)
        for part in X.keys():
            if part == 'train': continue
            for column_idx in range(X[part].shape[1]):
                X[part][X[part][:, column_idx] == unknown_value, column_idx] = (
                    max_values[column_idx] + 1
                )
        if return_encoder:
            return (X, False, encoder)
        return (X, False)
    else:
        print('Unknown encoding', encoding)
    
    if return_encoder:
        return X, True, encoder # type: ignore[code]
    return (X, True)




