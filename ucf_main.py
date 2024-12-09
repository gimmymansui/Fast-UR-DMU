import pdb
import numpy as np
import torch.utils.data as data
from torch.utils.data import Subset
import utils
from options import *
from config import *
from train import *
from ucf_test import test
from model import *
from utils import Visualizer
import os
from dataset_loader import *
from tqdm import tqdm
import copy
from prune import ModelPruner

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()

    config = Config(args)
    worker_init_fn = None
    gpus = [0]
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # Set the device
    torch.cuda.set_device(device if device.startswith('cuda') else 0)

    config.len_feature = 1024
    net = WSAD(config.len_feature, flag = "Train", a_nums = 60, n_nums = 60)
    net = net.to(device)

    pruner = ModelPruner(net, prune_amount=0.2)
    best_auc = 0
    best_state_dict = None

    # Calculate 10% of the dataset size
    total_normal_samples = len(UCF_crime(root_dir=config.root_dir, mode='Train', modal=config.modal, num_segments=200, len_feature=config.len_feature, is_normal=True))
    total_abnormal_samples = len(UCF_crime(root_dir=config.root_dir, mode='Train', modal=config.modal, num_segments=200, len_feature=config.len_feature, is_normal=False))
    
    normal_subset_size = int(0.1 * total_normal_samples)
    abnormal_subset_size = int(0.1 * total_abnormal_samples)

    # Create subsets
    normal_indices = np.random.choice(total_normal_samples, normal_subset_size, replace=False)
    abnormal_indices = np.random.choice(total_abnormal_samples, abnormal_subset_size, replace=False)

    normal_train_dataset = UCF_crime(root_dir=config.root_dir, mode='Train', modal=config.modal, num_segments=200, len_feature=config.len_feature, is_normal=True)
    abnormal_train_dataset = UCF_crime(root_dir=config.root_dir, mode='Train', modal=config.modal, num_segments=200, len_feature=config.len_feature, is_normal=False)

    normal_train_subset = Subset(normal_train_dataset, normal_indices)
    abnormal_train_subset = Subset(abnormal_train_dataset, abnormal_indices)

    # Update DataLoaders to use subsets
    normal_train_loader = data.DataLoader(
        normal_train_subset,
        batch_size=64,
        shuffle=True, num_workers=config.num_workers,
        worker_init_fn=worker_init_fn, drop_last=True)

    abnormal_train_loader = data.DataLoader(
        abnormal_train_subset,
        batch_size=64,
        shuffle=True, num_workers=config.num_workers,
        worker_init_fn=worker_init_fn, drop_last=True)

    test_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Test', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature),
            batch_size = 1,
            shuffle = False, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn)

    test_info = {"step": [], "auc": [],"ap":[],"ac":[]}
    
    best_auc = 0

    criterion = AD_Loss()
    
    optimizer = torch.optim.Adam(net.parameters(), lr = config.lr[0],
        betas = (0.9, 0.999), weight_decay = 0.00005)

    wind = Visualizer(env = 'UCF_URDMU', port = "2022", use_incoming_socket = False)
    test(net, config, wind, test_loader, test_info, 0)
    for step in tqdm(
            range(1, config.num_iters + 1),
            total = config.num_iters,
            dynamic_ncols = True
        ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]
        if (step - 1) % len(normal_train_loader) == 0:
            normal_loader_iter = iter(normal_train_loader)

        if (step - 1) % len(abnormal_train_loader) == 0:
            abnormal_loader_iter = iter(abnormal_train_loader)
        train(net, normal_loader_iter,abnormal_loader_iter, optimizer, criterion, wind, step)
        if step % 10 == 0 and step > 10:
            test(net, config, wind, test_loader, test_info, step)
            if test_info["auc"][-1] > best_auc:
                best_auc = test_info["auc"][-1]
                best_state_dict = copy.deepcopy(net.state_dict())
                
                pruner.prune_model()
                pruner.reset_weights()
                
                torch.save({
                    'state_dict': net.state_dict(),
                    'mask': {name + '.weight_mask': module.weight_mask 
                            for name, module in net.named_modules() 
                            if hasattr(module, 'weight_mask')}
                }, os.path.join(args.model_path, f"pruned_ucf_trans_{config.seed}.pkl"))
            if step == config.num_iters:
                torch.save(net.state_dict(), os.path.join(args.model_path, \
                    "ucf_trans_{}.pkl".format(step)))

