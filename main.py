import argparse, os
import torch,  yaml, json

from Model.model import create_model
from Script.utils import get_optimizer, get_scheduler, get_saverun, get_node_fea
from Script.loss import get_loss
from Script.data_container import get_dataloader
from Script.train import train_model, test_model


def str2bool(str):
    return True if str.lower() == 'true' else False


def main(config, test: bool = False, stage1: bool = False):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['device'])
    device = torch.device(0)

    optimizer_name, scheduler_name, data_set, mask, out_catagory = config['optimizer']['name'], config['scheduler'][
        'name'], config['data']['dataset'], config['loss'].pop('mask'), config.pop('out_catagory')

    loss_function = get_loss(**config['loss'])
    loss_function.to(device)

    node_fea = get_node_fea(data_set, config['data'][data_set].pop('node_fea_num'))

    model_name = 'EGC_Stage1'
    save_folder, run_folder = get_saverun(os.path.join('Save', data_set, f"device{config['device']}", model_name),
                                          os.path.join('Run', data_set, f"device{config['device']}", model_name),
                                          config, test, stage1)
    if stage1:
        dataloader = get_dataloader(dataset_id=data_set, **config['data'][data_set][model_name], out_catagory=out_catagory)
        model_1, trainer_1 = create_model(model_name, loss_function, device, config['model'][data_set][model_name],
                                          node_fea, out_catagory=out_catagory)
        optimizer_1 = get_optimizer(id=optimizer_name, model_parameter=model_1.parameters(),
                                    **config['optimizer'][optimizer_name])
        scheduler_1 = get_scheduler(id=scheduler_name, optimizer=optimizer_1, **config['scheduler'][scheduler_name])
        model_1 = train_model(model=model_1,
                              trainer=trainer_1,
                              data_loader=dataloader,
                              optimizer=optimizer_1,
                              scheduler=scheduler_1,
                              save_folder=save_folder,
                              run_folder=run_folder,
                              device=device,
                              mask=mask,
                              out_catagory = out_catagory,
                              **config['train'])
        test_model(folder=save_folder,
                   trainer=trainer_1,
                   model=model_1,
                   data_loader=dataloader,
                   device=device,
                   mask=mask,
                   out_catagory=out_catagory
                   )
    else:
        print("skip the step 1................")
    par_stage1 = torch.load(os.path.join(save_folder, 'best.pkl'))['model_state_dict']

    model_name = 'DSGRN_Stage2'
    save_folder, run_folder = get_saverun(os.path.join('Save', data_set, f"device{config['device']}", model_name),
                                          os.path.join('Run', data_set, f"device{config['device']}", model_name),
                                          config, test)
    dataloader = get_dataloader(dataset_id=data_set, **config['data'][data_set][model_name], out_catagory=out_catagory)
    model_2, trainer_2 = create_model(model_name, loss_function, device, config['model'][data_set][model_name],
                                      node_fea, par_stage1, out_catagory=out_catagory)
    optimizer_2 = get_optimizer(id=optimizer_name, model_parameter=model_2.parameters(),
                                **config['optimizer'][optimizer_name])
    scheduler_2 = get_scheduler(id=scheduler_name, optimizer=optimizer_2, **config['scheduler'][scheduler_name])

    if not test:
        model_2 = train_model(model=model_2,
                              trainer=trainer_2,
                              data_loader=dataloader,
                              optimizer=optimizer_2,
                              scheduler=scheduler_2,
                              save_folder=save_folder,
                              run_folder=run_folder,
                              device=device,
                              mask=mask,
                              out_catagory = out_catagory,
                              **config['train'])
    test_model(folder=save_folder,
               trainer=trainer_2,
               model=model_2,
               data_loader=dataloader,
               mask=mask,
               device=device,
               out_catagory=out_catagory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, default='egc-config',
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--test', required=False, type=str2bool, default='False',
                        help='Test.')
    parser.add_argument('--stage', required=True, type=str2bool, default='True',
                        help='Stage.')

    args = parser.parse_args()
    with open(os.path.join('Config', f'{args.config}.yaml')) as f:
        config = yaml.safe_load(f)
        print(json.dumps(config, indent=4))

    main(config, args.test, args.stage)
