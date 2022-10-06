import os, torch, time, copy, json

from tensorboardX import SummaryWriter
import torch.nn as nn
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from torch.optim.optimizer import Optimizer
from torch.optim import lr_scheduler

from Script.utils import get_number_parameters, save_model
from Script.evaluate import _evaluate
from Model.model import Trainer
from .utils import MyEncoder


def train_model(model: nn.Module,
                trainer: Trainer,
                data_loader: dict,
                optimizer: Optimizer,
                scheduler: lr_scheduler,
                save_folder: str,
                run_folder: str,
                mask: bool,
                out_catagory: str,
                device: torch.device,
                end_epoch: int,
                max_grad_norm: float,
                early_stop_steps: float):
    best_savepath = os.path.join(save_folder, 'best.pkl')
    if os.path.exists(best_savepath):
        best_dict = torch.load(best_savepath)
        model.load_state_dict(best_dict['model_state_dict'])
        optimizer.load_state_dict(best_dict['model_state_dict'])

        best_val_loss = best_dict['best_val_loss']
        begin_epoch = best_dict['epoch'] + 1
    else:
        best_dict = dict()
        best_val_loss = float('inf')
        begin_epoch = 0
        best_dict.update(model=copy.deepcopy(model),
                         epoch=0,
                         best_val_loss=best_val_loss,
                         optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))

    stages = ['train', 'val', 'test']
    begin_time = time.clock()

    writer = SummaryWriter(run_folder)
    model.to(device)
    print(model)
    print(f'Number of trainable parameters: {get_number_parameters(model)}.')

    try:
        for epoch in range(begin_epoch, end_epoch):

            for stage in stages:
                model.train() if stage=='train' else model.eval()
                steps, running_loss, output_list, ground_truth_list, metrics = 0, 0, list(), list(), defaultdict()

                runloss1 = 0
                tqdm_loader = tqdm(data_loader[stage])
                for input, ground_truth in tqdm_loader:
                    ground_truth_list.append(ground_truth.numpy())

                    input = input.to(device)
                    ground_truth = ground_truth.to(device)

                    with torch.set_grad_enabled(stage == 'train'):

                        output, loss, loss1, hook = trainer.train(input, ground_truth, stage)

                        if stage == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            if max_grad_norm is not None:
                                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()

                    output_list.append(output.detach().cpu().numpy())
                    running_loss += loss * len(ground_truth)

                    runloss1 += loss1 * len(ground_truth)

                    steps += len(ground_truth)
                    tqdm_loader.set_description(
                        f'{stage:5} epoch: {epoch:3}, {stage:5} loss: {running_loss / steps:3.6}, hook:{hook}')

                output_list, ground_truth_list = np.concatenate(output_list, axis=0), np.concatenate(ground_truth_list,
                                                                                                     axis=0)
                metrics = _evaluate(data_loader['scaler'].inverse_transform(output_list),
                                    data_loader['scaler'].inverse_transform(ground_truth_list), mask, out_catagory)
                for metric in metrics.keys():
                    print('')
                    print('')
                    for key, val in metrics[metric].items():
                        print(f'{metric}/{key}: {val:3.6}', end="    ")
                        writer.add_scalars(f'{metric}/{key}', {f'{stage}': val}, global_step=epoch)

                if stage == 'train':
                    scheduler.step(running_loss)
                elif stage == 'val':
                    if running_loss <= best_val_loss:
                        best_val_loss = running_loss
                        best_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                         epoch=epoch,
                                         best_val_loss=best_val_loss,
                                         optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                        print(f'Better model at epoch {epoch} recorded.')
                    elif early_stop_steps is not None and epoch - best_dict['epoch'] > early_stop_steps:
                        raise ValueError('Early stopped.')
                elif stage == 'test':
                    if epoch == best_dict['epoch']:
                        best_dict.update(best_test_loss=running_loss)
                        print(f'Best loss on test is {running_loss / steps:.6}')
        raise ValueError('Model have not converged............')


    except (ValueError, KeyboardInterrupt) as e:
        print(e)
        print(f'cost {time.clock() - begin_time} seconds, hook: {hook}')
        model.load_state_dict(best_dict['model_state_dict'])

        save_model(best_savepath, best_dict)


        print(
            f'model of epoch {best_dict["epoch"]}, best loss on test {best_dict["best_test_loss"]}, loss successfully saved at `{best_savepath}` ')

    return model


def test_model(folder: str,
               data_loader,
               trainer,
               model,
               mask,
               device,
               out_catagory):
    save_path = os.path.join(folder, 'best.pkl')
    save_dict = torch.load(save_path)
    trainer.load_dict(save_dict)

    trainer.set_eval()
    trainer.model.to(device)

    steps, predictions, running_targets = 0, list(), list()
    tqdm_loader = tqdm(enumerate(data_loader['test']))
    for step, (inputs, targets) in tqdm_loader:
        running_targets.append(targets.numpy())

        with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets.to(device)
            out = trainer.train(inputs, targets, 'test')
            output = out[0]
            predictions.append(output.cpu().numpy())

    running_targets, predictions = np.concatenate(running_targets, axis=0), np.concatenate(predictions, axis=0)

    scores = _evaluate(data_loader['scaler'].inverse_transform(predictions),
                       data_loader['scaler'].inverse_transform(running_targets), mask,out_catagory)

    print('test results:')
    print(json.dumps(scores, cls=MyEncoder, indent=4))
    with open(os.path.join(folder, 'test-scores.json'), 'w+') as f:
        json.dump(scores, f, cls=MyEncoder, indent=4)


    np.savez(os.path.join(folder, 'test-results.npz'), predictions=predictions, targets=running_targets)
