import torch
from numpy import ndarray
import torch.nn as nn


from .egc_stage1 import EGC_Stage1
from .egc_stage2 import EGC_Stage2
from .egc_stage1_single import EGC_Stage1_Single
from .egc_stage2_single import EGC_Stage2_Single



def create_model(config_name: str, loss_function: nn.Module, device: torch.device, config: dict,
                 node_fea: ndarray, pre_dict: dict = None, out_catagory: str = None):

    if out_catagory == 'multi':
        if config_name == 'EGC_Stage1':
            model = EGC_Stage1(**config, node_fea=torch.from_numpy(node_fea).float().to(device))
            trainer = DCRNNTrainer(model=model, loss=loss_function)
        elif config_name == 'EGC_Stage2':
            trainer_par = config.pop('trainer')
            model = EGC_Stage2(**config, node_fea=torch.from_numpy(node_fea).float().to(device))
            model_dict = model.state_dict()
            pretrain_dict = {k:v for k,v in pre_dict.items() if 'node_encode' in k}
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)
            trainer = DCRNN_regularTrainer(model=model, loss=loss_function, **trainer_par)
        else:
            raise ValueError(f'not such model: {config_name}')
    else:
        if config_name == 'EGC_Stage1':
            model = EGC_Stage1_Single(**config, node_fea=torch.from_numpy(node_fea).float().to(device))
            trainer = DCRNNTrainer(model=model, loss=loss_function)
        elif config_name == 'EGC_Stage2':
            trainer_par = config.pop('trainer')
            model = EGC_Stage2_Single(**config, node_fea=torch.from_numpy(node_fea).float().to(device))
            model_dict = model.state_dict()
            pretrain_dict = {k:v for k,v in pre_dict.items() if 'node_encode' in k}
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)
            trainer = DCRNN_regularTrainer(model=model, loss=loss_function, **trainer_par)
        else:
            raise ValueError(f'not such model: {config_name}')

    return model, trainer


class Trainer:
    def __init__(self, model, loss_function):
        self.loss = loss_function
        self.model = model

    def train(self, input: torch.Tensor, targets: torch.Tensor, phase: str = None):
        output, hook = self.model(input)
        loss = self.loss(output, targets)
        return output, loss, hook

    def load_dict(self, dict):
        print(dict['epoch'])
        self.model.load_state_dict(dict['model_state_dict'])

    def load_model(self, dict):
        print(dict['epoch'])
        self.model = dict['model']

    def set_eval(self):
        self.model.eval()



class DCRNNTrainer(Trainer):
    def __init__(self, model, loss):
        super(DCRNNTrainer, self).__init__(model, loss)
        self.train_batch_seen: int = 0

    def train(self, inputs: torch.Tensor, targets: torch.Tensor, phase: str):
        if phase == 'train':
            self.train_batch_seen += 1
        i_targets = targets if phase == 'train' else None
        outputs= self.model(inputs, i_targets, self.train_batch_seen)
        loss2 = self.loss(outputs, targets)
        # loss1 = self.loss(y_, y)
        # loss = loss2 + loss1 * 0.01

        return outputs, loss2, loss2, f''


class groupl2loss(nn.Module):
    def __init__(self):
        super(groupl2loss, self).__init__()

    def forward(self, inputs, labels):
        return torch.norm(inputs-labels, p=2)

class infityloss(nn.Module):
    def __init__(self):
        super(infityloss, self).__init__()

    def forward(self, inputs, labels):
        x = torch.norm(inputs - labels, p=float('inf'), dim=3)
        return torch.mean(x)

class DCRNN_regularTrainer(Trainer):
    def __init__(self, model, loss, alfa):
        super(DCRNN_regularTrainer, self).__init__(model, loss)
        self.train_batch_seen: int = 0
        self.alfa = alfa

        # self.loss2 = nn.L1Loss()
        # self.loss2 = groupl2loss()
        # self.loss2 = infityloss()

    def train(self, inputs: torch.Tensor, targets: torch.Tensor, phase: str):
        if phase == 'train':
            self.train_batch_seen += 1
        i_targets = targets if phase == 'train' else None
        outputs, regular_term, regular_label = self.model(inputs, i_targets, targets, self.train_batch_seen)
        loss1 = self.loss(outputs, targets)
        # loss2 = self.loss2(regular_term, regular_label)
        loss2 = self.loss(regular_term, regular_label)
        loss = loss1 + self.alfa * loss2
        return outputs, loss, loss1, f'normal:{loss1:5},   regular:{loss2:5}'



