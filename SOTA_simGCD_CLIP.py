"""SimGCD-CLIP state-of-the-art comparison implementation."""

import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader


from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment,distill_crit
from util.cluster_and_log_utils import log_accs_from_preds_v4
from config import exp_root,get_dir
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, JointProjector, TES

from clip import clip
import torch.nn.functional as F
import json


def train_simGCD(backbone, projector, model_tes, ln_t, jointer, train_loader, test_loader, unlabelled_train_loader, save_model_path, args,device):

    optimizer = SGD(list(projector.parameters())+list(backbone.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )


    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )


    best_train_acc_all = 0
    best_train_acc_new = 0
    best_train_acc_self = 0
    best_train_acc_seen = 0
    for epoch in range(args.epochs):
            
        loss_record = AverageMeter()
        backbone.train()
        projector.train()
        #ln_t.train()
        #jointer.train()
        #model_tes.eval()
        
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True,device=device), mask_lab.cuda(non_blocking=True,device=device).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True,device=device)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                image_features = backbone.encode_image(images).float()

                #caption_features = backbone.encode_text(caption_tokens).float()
                #caption_features_ln = ln_t(caption_features)
                #caption_features = F.normalize(caption_features, dim=-1)

                #text_features = model_tes(images)
                #text_features_ln = ln_t(text_features)

                #joint_features = jointer([image_features, text_features_ln])
                
                ## -----visual_branch-----
                student_proj, student_out = projector(image_features)
                teacher_out = student_out.detach()

                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # clustering, unsup
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs))) # mean entropy reg for visual
                cluster_loss += args.memax_weight * me_max_loss

                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj,device=device)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels,device=device)
              
                loss_v = 0
                loss_v += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                loss_v += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
              
    
              
                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
                
                    
                loss = loss_v
  

            # Train acc
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

        args.logger.info('Testing on unlabelled examples in the training data...')
        (all_acc_mix, old_acc_mix, self_acc_mix, new_acc_mix, 
        all_ari_mix, old_ari_mix, self_ari_mix, new_ari_mix, 
        all_nmi_mix, old_nmi_mix, self_nmi_mix, new_nmi_mix) = test(backbone, projector, model_tes,ln_t, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args,device=device)
        args.logger.info('Train Acc(mix): All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_acc_mix, old_acc_mix, self_acc_mix, new_acc_mix))
        args.logger.info('Train ARI(mix): All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_ari_mix, old_ari_mix, self_ari_mix, new_ari_mix))
        args.logger.info('Train NMI(mix): All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_nmi_mix, old_nmi_mix, self_nmi_mix, new_nmi_mix))
        
        with torch.no_grad():
            results = test(backbone, projector, model_tes, ln_t, test_loader, epoch=0,
                        save_name='Final Test ACC', args=args, device=device)
        
        (all_acc_mix, old_acc_mix, self_acc_mix, new_acc_mix, 
        all_ari_mix, old_ari_mix, self_ari_mix, new_ari_mix, 
        all_nmi_mix, old_nmi_mix, self_nmi_mix, new_nmi_mix) = results

        args.logger.info('Test Acc(mix): All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_acc_mix, old_acc_mix, self_acc_mix, new_acc_mix))
        args.logger.info('Test ARI(mix): All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_ari_mix, old_ari_mix, self_ari_mix, new_ari_mix))
        args.logger.info('Test NMI(mix): All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_nmi_mix, old_nmi_mix, self_nmi_mix, new_nmi_mix))
        
        # Step schedule
        exp_lr_scheduler.step()

        

        save_dict = {
            'backbone': backbone.state_dict(),
            #'model_tes_projector': model_tes.projector.state_dict(),
            #'ln_t': ln_t.state_dict(),
            #'jointer': jointer.state_dict(),
            'proj': projector.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        #torch.save(save_dict, args.model_path)
        #args.logger.info("full model saved to {}.".format(args.model_path))

        if (all_acc_mix + 0.3 * new_acc_mix) > (best_train_acc_all + 0.3 * best_train_acc_new):

            args.logger.info('Best Acc(mix): All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_acc_mix, old_acc_mix, self_acc_mix, new_acc_mix))
            args.logger.info('Best ARI(mix): All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_ari_mix, old_ari_mix, self_ari_mix, new_ari_mix))
            args.logger.info('Best NMI(mix): All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_nmi_mix, old_nmi_mix, self_nmi_mix, new_nmi_mix))

            torch.save(save_dict, save_model_path)
            args.logger.info("Best model saved to {}.".format(save_model_path))
            
            best_train_acc_seen = old_acc_mix
            best_train_acc_self = self_acc_mix
            best_train_acc_new = new_acc_mix
            best_train_acc_all = all_acc_mix
        
        #args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_seen:.4f} Semi: {best_train_acc_self:.4f} New: {best_train_acc_new:.4f}')
        args.logger.info(f'-----------------------------------------------')

def test(backbone, projector, model_tes,ln_t, test_loader, epoch, save_name, args,device):
    backbone.eval()
    projector.eval()
    #model_tes.eval()
    #ln_t.eval()
    #jointer.eval()

    preds_img, preds_text, targets = [], [],[]
    preds_mix = []
    mask_seen = np.array([])
    mask_new = np.array([])
    for batch_idx, (images, label, uq_idxs) in enumerate(test_loader):
        images = images.cuda(non_blocking=True,device=device)

        with torch.no_grad():
            image_features = backbone.encode_image(images).float()
            #text_features = model_tes(images)
            #text_features_ln = ln_t(text_features)
            #caption_features = backbone.encode_text(caption_tokens).float()
            #joint_features = jointer([image_features, text_features_ln])
            #_, logits_text = projector(text_features_ln)
            _, logits_img = projector(image_features)
            #_, logits_mix = projector(joint_features)
            
            preds_img.append(logits_img.argmax(1).cpu().numpy())
            #preds_text.append(logits_text.argmax(1).cpu().numpy())
            #preds_mix.append(logits_mix.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask_new = np.append(mask_new, np.array([True if x.item() not in args.train_classes else False for x in label]))
            mask_seen = np.append(mask_seen, np.array([True if x.item() in args.train_labelled_classes else False for x in label]))

    preds_img = np.concatenate(preds_img)
    #preds_text = np.concatenate(preds_text)
    #preds_mix = np.concatenate(preds_mix)
    
    targets = np.concatenate(targets)

    all_acc_mix, old_acc_mix, self_acc_mix, new_acc_mix, all_ari_mix, old_ari_mix, self_ari_mix, new_ari_mix, all_nmi_mix, old_nmi_mix, self_nmi_mix, new_nmi_mix  = log_accs_from_preds_v4(
                                                    y_true=targets, y_pred=preds_img, mask_seen=mask_seen, mask_new=mask_new,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc_mix, old_acc_mix, self_acc_mix, new_acc_mix, all_ari_mix, old_ari_mix, self_ari_mix, new_ari_mix, all_nmi_mix, old_nmi_mix, self_nmi_mix, new_nmi_mix

def test_simGCD(backbone, projector, model_tes, ln_t, jointer, test_loader, ckpt_path, args, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt['backbone'])
    projector.load_state_dict(ckpt['proj'])
    #ln_t.load_state_dict(ckpt['ln_t'])
    #jointer.load_state_dict(ckpt['jointer'])
    #model_tes.projector.load_state_dict(ckpt['model_tes_projector'])

    backbone.to(device).eval()
    projector.to(device).eval()
    #ln_t.to(device).eval()
    #jointer.to(device).eval()
    #model_tes.to(device).eval()

    with torch.no_grad():
        results = test(backbone, projector, model_tes, ln_t, test_loader, epoch=0,
                       save_name='Final Test ACC', args=args, device=device)

    (all_acc_mix, old_acc_mix, self_acc_mix, new_acc_mix, 
     all_ari_mix, old_ari_mix, self_ari_mix, new_ari_mix, 
     all_nmi_mix, old_nmi_mix, self_nmi_mix, new_nmi_mix) = results

    args.logger.info('Final Test Acc(mix): All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_acc_mix, old_acc_mix, self_acc_mix, new_acc_mix))
    args.logger.info('Final Test ARI(mix): All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_ari_mix, old_ari_mix, self_ari_mix, new_ari_mix))
    args.logger.info('Final Test NMI(mix): All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_nmi_mix, old_nmi_mix, self_nmi_mix, new_nmi_mix))

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
    
    parser.add_argument('--seed', default=None, type=int)
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
    
    
    init_experiment(args, runner_name=['simGCD-CLIP'], exp_id=args.exp_id)
    
    args.interpolation = 3
    args.crop_pct = 0.875

    args.image_size = 224
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
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
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
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                         batch_size=256, shuffle=False, pin_memory=False)
    
    
    # load model

    model_tes = None
    
    model_clip, preprocess = clip.load("ViT-B/16",device)

        
    for m in model_clip.parameters():
        m.requires_grad = False

    for name, m in model_clip.named_parameters():
        # if 'visual.proj' in name:
        #     m.requires_grad =True
        if 'visual.transformer.resblocks' in name:
            block_num = int(name.split('.')[3])
            if block_num >= args.grad_from_block:
                m.requires_grad = True
    
    model_clip.visual.use_proj = False 
    
    

    
    
    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=768, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers).to(device)
    
    
    ln_t = None
    jointer = None
    args.logger.info(f'Training simGCD CLIP......')
    save_model_path ="saved_models/simGCD_222_"+args.class_split_file[-10:-5]+".pt"
    train_simGCD(model_clip, projector, model_tes, ln_t, jointer, train_loader, test_loader, train_loader_unlabelled, save_model_path,args,device)

    test_simGCD(model_clip, projector, model_tes, ln_t, jointer, test_loader, save_model_path, args, device)
    
    
    
        
    
    
  
