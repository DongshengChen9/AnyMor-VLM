"""Autoencoder plus seeded K-means baseline for UrbanForm evaluation."""

import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam 

from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds_v4
from config import exp_root,get_dir
from model import ContrastiveLearningViewGenerator
from torchvision import transforms

from sklearn.metrics import pairwise_distances
import json
import os

class AEencoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        ## 512 * 512 * 1 -> 256 * 256 * 15 -> 128 * 128 * 15 -> 64 * 64 * 15 -> 32 * 32 * 10 -> 16 * 16 * 10 = 2560   
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 15, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(15, 15, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(15, 15, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(15, 10, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(10, 10, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(16 * 16 * 10, 1024),
            nn.Linear(1024, encoded_space_dim)
        )
    def forward(self, x):
        x = self.encoder_cnn(x)
       # print(x.size())
        x = self.flatten(x)
       # print(x.size())
        x = self.encoder_lin(x)
        #print(x.size())
        return x
    
class AEdecoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 1024),
            nn.Linear(1024, 16 * 16 * 10)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(10, 16, 16))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(10, 10, 3, stride=2, 
            padding=1, output_padding=1),
            nn.ConvTranspose2d(10, 15, 3, stride=2, 
            padding=1, output_padding=1),
            nn.ConvTranspose2d(15, 15, 3, stride=2, 
            padding=1, output_padding=1),
            nn.ConvTranspose2d(15, 15, 3, stride=2, 
            padding=1, output_padding=1),
            nn.ConvTranspose2d(15, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )
    def forward(self, x):
        x = self.decoder_lin(x)
        #print(x.size())
        x = self.unflatten(x)
        #print(x.size())
        x = self.decoder_conv(x)
        #print(x.size())
        x = torch.sigmoid(x)
        #print(x.size())
        return x

def kmeanspp_pick_centroids(X_unlabeled, existing_centroids, n_to_pick, random_state=None):
    rng = np.random.RandomState(random_state)
    n_samples, dim = X_unlabeled.shape

    if n_to_pick <= 0:
        return np.zeros((0, dim))

    centroids = []

    if existing_centroids is None or existing_centroids.shape[0] == 0:
        first_idx = rng.randint(0, n_samples)
        centroids.append(X_unlabeled[first_idx].copy())
        n_existing = 1
    else:
        centroids = [c.copy() for c in existing_centroids]
        n_existing = len(centroids)
        

    target_total = n_existing + n_to_pick

    while len(centroids) < target_total:

        C = np.array(centroids)
        d2 = np.sum((X_unlabeled[:, None, :] - C[None, :, :])**2, axis=2)
        nearest_d2 = np.min(d2, axis=1)
        min_allowed_dist = 0.3 * np.median(d2)
        if existing_centroids is not None and existing_centroids.shape[0] > 0:
            d2_existing = np.sum(
                (X_unlabeled[:, None, :] - existing_centroids[None, :, :])**2, axis=2
            )
            min_d2_existing = np.min(d2_existing, axis=1)
            
            # If a point is too close, set its probability nearly to zero
            mask_too_close = (min_d2_existing < min_allowed_dist)
            nearest_d2 = nearest_d2.copy()
            nearest_d2[mask_too_close] = 1e-12  # almost zero probability

        # If distances all zero (degenerate), pick random
        if np.all(nearest_d2 <= 1e-12):
            idx = rng.randint(0, n_samples)
            centroids.append(X_unlabeled[idx].copy())
            continue

        # Standard K-means++ probability
        probs = nearest_d2 / np.sum(nearest_d2)
        idx = rng.choice(n_samples, p=probs)
        centroids.append(X_unlabeled[idx].copy())

    # Return only the new centroids
    return np.array(centroids[n_existing:target_total])


def seeded_kmeans(X, centroids_seen, K, max_iter=300, tol=1e-4, random_state=0):
    rng = np.random.RandomState(random_state)
    n, d = X.shape
    seen_classes = list(centroids_seen.keys())
    n_seen = len(seen_classes)

    # initial centroids for seen classes
    if K < n_seen:
        raise ValueError(f'K must >= seen classes({n_seen})')
    centroids = np.zeros((K, d))
    for idx, cidx in enumerate(seen_classes):
        centroids[idx] = centroids_seen[cidx]

    # initial centroids for unseen classes
    n_to_pick = K - n_seen
    if n_to_pick > 0:
        new_cs = kmeanspp_pick_centroids(X, centroids[:n_seen], n_to_pick, random_state=random_state)
        centroids[n_seen:] = new_cs

    assignments = np.full(n, -1, dtype=int)

    for it in range(max_iter):
        dists = pairwise_distances(X, centroids, metric='euclidean')
        # assign unlabeled
        assignments = np.argmin(dists, axis=1)

        # recompute centroids
        new_centroids = np.zeros_like(centroids)
        changed = False
        for k in range(K):
            assigned_idx = np.where(assignments == k)[0]
            if len(assigned_idx) == 0:
                # empty cluster: reinit to random unlabeled point
                new_centroids[k] = X[rng.randint(0, n)]
            else:
                new_centroids[k] = np.mean(X[assigned_idx], axis=0)
        shift = np.linalg.norm(centroids - new_centroids, axis=1).max()
        centroids = new_centroids
        if shift <= tol:
            break
    return assignments, centroids


def seeded_semi_supervised_kmeans(X, labels, mask_lab, seen_classes, K, max_iter=300, tol=1e-4, random_state=0):
    rng = np.random.RandomState(random_state)
    n, d = X.shape
    labels = np.asarray(labels)
    mask_lab = np.asarray(mask_lab, dtype=bool)
    seen_classes = np.asarray(seen_classes, dtype=int)

    if np.any(seen_classes < 0) or np.any(seen_classes >= K):
        raise ValueError(f'seen class ids must be in [0, {K - 1}], got {seen_classes.tolist()}')

    unlabeled_idx = np.where(~mask_lab)[0]
    seen_set = set(seen_classes.tolist())
    unseen_slots = np.asarray([k for k in range(K) if k not in seen_set], dtype=int)

    n_seen = len(seen_classes)
    if K < n_seen:
        raise ValueError(f'K must >= seen classes({n_seen})')

    centroids = np.zeros((K, d), dtype=X.dtype)
    fixed_assign = np.full(n, -1, dtype=int)

    # Put each labelled class centroid in the slot corresponding to its class id.
    for cidx in seen_classes:
        idxs = np.where((labels == cidx) & mask_lab)[0]
        if len(idxs) == 0:
            raise ValueError(f'no training data in seen class ({cidx})')
        centroids[cidx] = np.mean(X[idxs], axis=0)
        fixed_assign[idxs] = cidx

    # Initialize only the truly unseen slots. The original code used centroids[:n_seen]
    # and centroids[n_seen:], which is wrong when seen class ids are not 0..n_seen-1.
    n_to_pick = len(unseen_slots)
    if n_to_pick > 0:
        if len(unlabeled_idx) == 0:
            for slot in unseen_slots:
                source_class = seen_classes[rng.randint(0, n_seen)]
                centroids[slot] = centroids[source_class]
        else:
            existing = centroids[seen_classes]
            new_cs = kmeanspp_pick_centroids(
                X[unlabeled_idx], existing, n_to_pick, random_state=random_state
            )
            centroids[unseen_slots] = new_cs

    assignments = np.full(n, -1, dtype=int)

    for _ in range(max_iter):
        dists = pairwise_distances(X, centroids, metric='euclidean')
        new_assign = np.argmin(dists, axis=1)
        new_assign[fixed_assign >= 0] = fixed_assign[fixed_assign >= 0]

        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            assigned_idx = np.where(new_assign == k)[0]
            if len(assigned_idx) == 0:
                new_centroids[k] = X[rng.randint(0, n)]
            else:
                new_centroids[k] = np.mean(X[assigned_idx], axis=0)

        shift = np.linalg.norm(centroids - new_centroids, axis=1).max()
        centroids = new_centroids
        assignments = new_assign
        if shift <= tol:
            break

    return assignments, centroids

def train_AutoEncoder(encoder, decoder, train_loader, train_eval_loader, test_loader, save_model_path, args, device):
    loss_fn = nn.BCELoss()
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]
    optimizer = Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
    )

    best_train_acc_all = 0
    best_train_acc_new = 0
    best_train_acc_self = 0
    best_train_acc_seen = 0

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        encoder.train()
        decoder.train()
        train_loss = []
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True,device=device), mask_lab.cuda(non_blocking=True,device=device).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True,device=device)
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                image_features = encoder(images).float()
                decoded_data = decoder(image_features)
                loss = loss_fn(decoded_data, images)
                pstr = f'training_loss: {loss.item():.4f} '


            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))
                
        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        with torch.no_grad():
            results = test_Kmeans(encoder, train_eval_loader, test_loader, epoch=epoch, save_name='Test ACC', args=args, device=device)
        
        (all_acc, old_acc, self_acc, new_acc, 
        all_ari, old_ari, self_ari, new_ari, 
        all_nmi, old_nmi, self_nmi, new_nmi) = results

        args.logger.info('Test Acc: All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_acc, old_acc, self_acc, new_acc))
        args.logger.info('Test ARI: All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_ari, old_ari, self_ari, new_ari))
        args.logger.info('Test NMI: All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_nmi, old_nmi, self_nmi, new_nmi))
        
        # Step schedule
        exp_lr_scheduler.step()
        save_dict = {
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        #torch.save(save_dict, args.model_path)
        #args.logger.info("full model saved to {}.".format(args.model_path))

        if (all_acc) > (best_train_acc_all):

            args.logger.info('Best Acc: All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_acc, old_acc, self_acc, new_acc))
            args.logger.info('Best ARI: All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_ari, old_ari, self_ari, new_ari))
            args.logger.info('Best NMI: All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_nmi, old_nmi, self_nmi, new_nmi))

            torch.save(save_dict, save_model_path)
            args.logger.info("Best model saved to {}.".format(save_model_path))
            
            best_train_acc_seen = old_acc
            best_train_acc_self = self_acc
            best_train_acc_new = new_acc
            best_train_acc_all = all_acc
        
        #args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_seen:.4f} Semi: {best_train_acc_self:.4f} New: {best_train_acc_new:.4f}')
        args.logger.info(f'-----------------------------------------------')

def test_Kmeans(encoder, train_eval_loader, test_loader, epoch, save_name, args, device):
    encoder.eval()
    seen_classes = args.train_labelled_classes
    X_lab, y_lab, X_unlab = [], [], []

    # Use a deterministic evaluation loader here. This function supports either
    # a single tensor image or a list/tuple of views.
    for batch in train_eval_loader:
        images, class_labels, uq_idxs, mask_lab = batch
        if mask_lab.ndim > 1:
            mask_lab = mask_lab[:, 0]

        class_labels = class_labels.to(device, non_blocking=True)
        mask_lab = mask_lab.to(device, non_blocking=True).bool()

        with torch.no_grad():
            if isinstance(images, (list, tuple)):
                view_features = [
                    encoder(view.to(device, non_blocking=True)).float()
                    for view in images
                ]
                sup_image_features = torch.cat(
                    [features[mask_lab] for features in view_features], dim=0
                )
                sup_labels = class_labels[mask_lab].repeat(len(view_features))
            else:
                image_features = encoder(images.to(device, non_blocking=True)).float()
                sup_image_features = image_features[mask_lab]
                sup_labels = class_labels[mask_lab]

        if len(sup_labels) > 0:
            X_lab.append(sup_image_features.cpu().numpy())
            y_lab.append(sup_labels.cpu().numpy())

    if not X_lab:
        raise ValueError('No labelled examples were found in train_eval_loader')

    X_lab = np.concatenate(X_lab, axis=0)
    y_lab = np.concatenate(y_lab, axis=0)

    all_uq_idxs = []
    all_labels = []
    mask_seen_parts = []
    mask_new_parts = []

    for images, label, uq_idxs in test_loader:
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            image_features = encoder(images).float()

        X_unlab.append(image_features.cpu().numpy())
        all_uq_idxs.append(uq_idxs.cpu().numpy())
        label_np = label.cpu().numpy()
        all_labels.append(label_np)
        mask_new_parts.append(~np.isin(label_np, args.train_classes))
        mask_seen_parts.append(np.isin(label_np, args.train_labelled_classes))

    X_unlab = np.concatenate(X_unlab, axis=0)
    all_uq_idxs = np.concatenate(all_uq_idxs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    mask_seen = np.concatenate(mask_seen_parts, axis=0).astype(bool)
    mask_new = np.concatenate(mask_new_parts, axis=0).astype(bool)

    X_all = np.vstack([X_lab, X_unlab])
    n_lab = len(X_lab)
    n_unlab = len(X_unlab)

    masked_labels = np.concatenate([y_lab, -np.ones(n_unlab, dtype=int)])
    is_lab = np.concatenate([
        np.ones(n_lab, dtype=bool),
        np.zeros(n_unlab, dtype=bool),
    ])

    kmeans_seed = 0 if args.seed is None else args.seed
    assignments, centroids = seeded_semi_supervised_kmeans(
        X_all,
        masked_labels,
        is_lab,
        seen_classes,
        args.mlp_out_dim,
        max_iter=300,
        random_state=kmeans_seed,
    )
    assign_unlab = assignments[n_lab:]

    results = log_accs_from_preds_v4(
        y_true=all_labels,
        y_pred=assign_unlab,
        mask_seen=mask_seen,
        mask_new=mask_new,
        T=epoch,
        eval_funcs=args.eval_funcs,
        save_name=save_name,
        print_output=False,
        args=args,
    )
    return results

def test_Final_model(encoder, train_loader, test_loader, args, device):
    encoder.to(device).eval()

    with torch.no_grad():
        results = test_Kmeans(encoder, train_loader, test_loader, epoch=0,
                       save_name='Final Test ACC', args=args, device=device)

    (all_acc, old_acc, self_acc, new_acc, 
     all_ari, old_ari, self_ari, new_ari, 
     all_nmi, old_nmi, self_nmi, new_nmi) = results

    args.logger.info('Final Test Acc: All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_acc, old_acc, self_acc, new_acc))
    args.logger.info('Final Test ARI: All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_ari, old_ari, self_ari, new_ari))
    args.logger.info('Final Test NMI: All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_nmi, old_nmi, self_nmi, new_nmi))

    return results
    


def set_random_seed(seed: int) -> None:
    import random,os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_class_split_info(args):
      
    # loading class split file if exist
    if not args.random_class_split:
        assert os.path.exists(args.class_split_file)
        
        with open(args.class_split_file, 'r') as f:
            split_info = json.load(f)
        args.train_classes = split_info['train_classes']
        args.unlabeled_classes = split_info['unlabeled_classes']
        args.train_labelled_classes = split_info['train_labelled_classes']
        args.train_unlabelled_classes = split_info['train_unlabelled_classes']

        return args

    # random generate class split
    np.random.seed(args.seed)

    all_classes = np.arange(args.num_classes4test)
    np.random.shuffle(all_classes)
    
    num_train_classes = int(args.num_classes4test/3*2)
    num_train_labeled_classes = int(num_train_classes/2)

    train_classes = np.sort(all_classes[:num_train_classes])
    unlabeled_classes = np.sort(all_classes[num_train_classes:])

    labelled_classes = np.sort(np.random.choice(train_classes,size=num_train_labeled_classes,replace=False))
    unlabelled_train_classes = np.sort(np.setdiff1d(train_classes,labelled_classes))
    unlabeled_classes = sorted(unlabeled_classes.tolist()+unlabelled_train_classes.tolist())

    split_info = {
        "train_classes":
            train_classes.tolist(),
        "unlabeled_classes":
            unlabeled_classes,
        "train_labelled_classes":
            labelled_classes.tolist(),
        "train_unlabelled_classes":
            unlabelled_train_classes.tolist()
    }
    
    base_name = "class_split_info_run"
    extension = ".json"
    counter = 1
    while True:
        class_split_path = f"{base_name}{counter}{extension}"
        class_split_path = os.path.join('data/', class_split_path)
        if not os.path.exists(class_split_path):
            break  
        counter += 1
        
    with open(class_split_path, 'w') as f:
        json.dump(split_info, f, indent=2)
        
    args.train_classes = split_info['train_classes']
    args.unlabeled_classes = split_info['unlabeled_classes']
    args.train_labelled_classes = split_info['train_labelled_classes']
    args.train_unlabelled_classes = split_info['train_unlabelled_classes']
    print(args.train_classes)
    return args

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v4'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='urbanform', help='options: urbanform, allform')
    parser.add_argument('--prop_train_labels', type=float, default=0.8)
    parser.add_argument('--use_ssb_splits', action='store_true', default=False)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='debug', type=str)
    parser.add_argument('--exp_id', default='debug', type=str)
    
    parser.add_argument('--num_words', default=7, type=int)
    parser.add_argument('--words_drop_ratio', default=0.2, type=float)
    parser.add_argument('--word_dim', default=512, type=int)   
    parser.add_argument('--lamC', default=1.0, type=float)
    parser.add_argument('--TES_model_dir', type=str, default=None)
    
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')
    
    parser.add_argument('--num_classes4test', type=int, default=6, help='number of classes for testing')
    parser.add_argument('--class_split_file', type=str, default='data/class_split_info_run00.json', help='file path to split classes')
    parser.add_argument('--random_class_split', action='store_true', default=False, help='if true, randomly split Labeled-Unlabeled-Novel equally. if false, load --class_split_file')
    
    
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    
    if args.seed is None:
        torch.backends.cudnn.benchmark = True
    else: set_random_seed(args.seed)
    
    device = torch.device(f"cuda:{args.cuda_dev}" if torch.cuda.is_available() else "cpu")

    args = get_class_splits(args)

    #args.num_labeled_classes = len(args.train_classes)
    #args.num_unlabeled_classes = len(args.unlabeled_classes)
    
    
    init_experiment(args, runner_name=['Autoencoder'], exp_id=args.exp_id)
    
    args.interpolation = 3
    args.crop_pct = 0.875

    args.image_size = 512
    args.feat_dim = 512
    args.num_mlp_layers = 3
    args = get_class_split_info(args)    
    class_names_path = os.path.join(get_dir, f"dataset_class_name/{args.dataset_name}_name.npy")
    if not os.path.exists(class_names_path):
        print(f"generate class_names_path for {args.dataset_name}")
        from dataset_class_name import gen_classnames
        gen_classnames.gen(args.dataset_name, class_names_path) 
    class_names = np.load(class_names_path)
    args.base_names = class_names[args.train_classes]
    
    #print("train_classes:", class_names[args.train_classes])
    #print("train_labelled_classes:", class_names[args.train_labelled_classes])
    #print("train_unlabelled_classes:", class_names[args.train_unlabelled_classes])
    #print("novel_classes:", class_names[list(set(args.unlabeled_classes)-set(args.train_unlabelled_classes))])
    
    args.mlp_out_dim = args.num_classes4test # output number of classes
    
    
    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
    ])
    # train_transform, test_transform = preprocess,preprocess
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets,train_examples_test, full_dataset = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    
    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=False)
    train_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    # Deterministic loader for computing labelled centroids during evaluation.
    # train_examples_test is expected to use test_transform and return
    # (image, class_label, uq_idx, mask_lab).
    train_eval_loader = DataLoader(train_examples_test, num_workers=args.num_workers,
                                   batch_size=256, shuffle=False, drop_last=False,
                                   pin_memory=False)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                         batch_size=256, shuffle=False, pin_memory=False)
    
    
    # load model
    encoder = AEencoder(encoded_space_dim=args.feat_dim).to(device)
    decoder = AEdecoder(encoded_space_dim=args.feat_dim).to(device)

    args.logger.info(f'Training AutoEncoder......')
    save_model_path ="saved_models/AutoEncoder_222_"+args.class_split_file[-10:-5]+".pt"
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    train_AutoEncoder(
        encoder, decoder, train_loader, train_eval_loader, test_loader,
        save_model_path, args, device
    )

    # IMPORTANT: evaluate the saved best checkpoint, not the last in-memory epoch.
    checkpoint = torch.load(save_model_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    args.logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}: {save_model_path}")

    args.logger.info(f'Accuracy of Semi-supervised K-Means ++ for AutoEncoder representations:')
    test_Final_model(encoder, train_eval_loader, test_loader, args, device)
    
    
    
        
    
    
  
