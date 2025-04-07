# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics

import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def train_parser():
    
    """
    Initializes and returns a command-line argument parser for the training process.

    Usage examples:
        python train.py --hypes_yaml [path_to_yaml]
        python train.py --hypes_yaml [path_to_yaml] --half
    
    Resturn:
        opt: It will be something like 
        Namespace(dist_url='env://', half=False, hypes_yaml='opencood/hypes_yaml/point_pillar_intermediate_V2VAM.yaml', model_dir='')
    
    done
    """
    # Create an argument parser with a brief description
    parser = argparse.ArgumentParser(description="Parser for training with synthetic data")

    # Required: Path to the YAML configuration file for hyperparameters
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help="Path to the YAML configuration file for training")

    # Optional: Directory for saving or loading model checkpoints
    parser.add_argument('--model_dir', default='',
                        help="Directory to load/save the model checkpoint")

    # Optional: Enable mixed-precision training (half-precision)
    parser.add_argument("--half", action='store_true',
                        help="Use half-precision training to reduce memory consumption")

    # Optional: Distributed training initialization URL (default uses environment variable)
    parser.add_argument('--dist_url', default='env://',
                        help="URL for initializing distributed training")

    # Parse and return the command-line arguments
    opt = parser.parse_args()
    print(f"-----------------Command Line Arguments Summary------------------\n{opt}")
    return opt



def main():
    opt = train_parser()

    print("-----------------Yaml file Reading------------------")
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    
    print('-----------------Multi GPU Checking------------------',end="\n")
    multi_gpu_utils.init_distributed_mode(opt)
    opencood_train_dataset = build_dataset(dataset_cfg=hypes, visualize=False, train=True)
    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(dataset_cfg=hypes, visualize=False, train=True)
   
    opencood_validate_dataset = build_dataset(dataset_cfg=hypes, visualize=False, train=False)
    if opt.distributed:
        print('-----------------DataLoader Creating------------------')
        print('Since Distributed training environment detected. Initializing DistributedSampler for data parallelization is done.')
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=0, # These were 8, due to error I set them to 0
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=0,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        print('-----------------DataLoader Creating------------------')
        print('Since Distributed training environment not detected. Initializing DataLoader for data parallelization is done.')
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=0,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=0,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Model will run ({device}) device.')
    # if we want to train from last checkpoint.
    if opt.model_dir:
        print(f'Model directory {opt.model_dir} detected. Initializing model from saved checkpoint.')
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    print('---------------Creating Loss Function------------------')
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    print('---------------Creating Optimizer------------------')
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('---------------Training start---------------------')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])


            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
