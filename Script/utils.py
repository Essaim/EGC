
import torch, shutil, yaml, os, pickle, h5py, json
import torch.nn as nn
import numpy as np
import pandas as pd
from .normalization import standard



def get_node_fea(data_set, train_num=0.7):
    if data_set == 'METR-LA':
        path = 'Data/metr-la.h5'
    elif data_set == 'PEMS-BAY':
        path = 'Data/pems-bay.h5'
    elif data_set == 'PEMS03':
        path = 'Data/PEMS03.h5'
    elif data_set == 'PEMS04':
        path = 'Data/PEMS04.h5'
    elif data_set == 'PEMS08':
        path = 'Data/PEMS08.h5'
    elif data_set == 'solar-energy':
        path = 'Data/solar-energy.h5'
    elif data_set == 'electricity':
        path = 'Data/electricity.h5'
    elif data_set == 'exchange-rate':
        path = 'Data/exchange-rate.h5'
    elif data_set == 'ECG':
        path = 'Data/ECG.h5'
    elif data_set == 'NYCBike':
        path = 'Data/bike_data.h5'
    elif data_set == 'NYCTaxi':
        path = 'Data/taxi_data.h5'
    else:
        raise ('No such dataset........................................')


    # df = x['df']['block0_values'][:]
    # num_samples = df.shape[0]
    # num_train = round(num_samples * train_num)
    # df = df[:num_train]
    # scaler = standard(df)
    # train_feas = scaler.transform(df)

    if data_set == 'NYCBike' or data_set== 'NYCTaxi':
        x = h5py.File(path, 'r')
        data = list()
        for key in x.keys():
            data.append(x[key][:])
        data = np.stack(data, axis=1)
        num_samples = data.shape[0]
        num_train = round(num_samples * train_num)
        df = data[:num_train]
        scaler = standard(df)
        train_feas = scaler.transform(df).reshape([-1,df.shape[2]])
    else:
        x = pd.read_hdf(path)
        data = x.values
        print(x.shape)
        num_samples = data.shape[0]
        num_train = round(num_samples * train_num)
        df = data[:num_train]
        print(df.shape)
        scaler = standard(df)
        train_feas = scaler.transform(df)
    return train_feas



def get_optimizer(id, model_parameter, **kwargs):
    return getattr(torch.optim, id)(model_parameter, **kwargs)


def get_scheduler(id, optimizer, **kwargs):
    return getattr(torch.optim.lr_scheduler, id)(optimizer, **kwargs)


def get_saverun(save_folder: str, run_folder: str, config:dict, test: bool = False, stage1 = True):
    shutil.rmtree(run_folder, ignore_errors=True)
    os.makedirs(run_folder)
    if not test and stage1:
        shutil.rmtree(save_folder, ignore_errors=True)
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.yaml'), 'w+') as f:
        yaml.safe_dump(config, f)

    return save_folder, run_folder

def get_number_parameters(model:nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def save_model(path, best_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(best_dict, path)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def np_softmax(x, axis):

    mx = np.max(x, axis=axis, keepdims=True)
    numerator = np.exp(x - mx)
    denominator = np.sum(numerator, axis=axis, keepdims=True)
    return numerator / denominator