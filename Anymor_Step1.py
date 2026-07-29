import argparse
import os
import numpy as np
import json
import torch
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader


from data.augmentations import get_transform
from data.get_datasets import get_datasets

from util.general_utils import AverageMeter, init_experiment,distill_crit
from config import exp_root,get_dir
from model import ContrastiveLearningViewGenerator, TES

from clip import clip
import torch.nn.functional as F

from config import urbanform_root

    
def train_tes(model_clip, model_tes,train_loader, args,device):
    

    optimizer = SGD(list(model_tes.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    mseloss=torch.nn.MSELoss(reduction='sum')
    
    # text_features of base classes
    if args.dataset_name == 'urbanform':
        class_descriptions = np.load('./dataset_class_name/urbanform_descriptions.npy')
        text_descriptions = [f"In urban morphology, this is a figure-ground diagram of urban form of {label} with {descrip}" for label, descrip in zip(args.base_names, class_descriptions)]
        print(text_descriptions)
    else:
        text_descriptions = [f"a photo of a {label}" for label in args.base_names]
    
    base_Tfeats = model_tes.text_encoder(text_descriptions)
    base_Tfeats = base_Tfeats.float().detach()
    
    logit_scale = model_clip.logit_scale.exp()
    logit_scale =  logit_scale.detach()
    # # inductive
    # best_test_acc_lab = 0
    # # transductive
    # best_train_acc_lab = 0
    # best_train_acc_ubl = 0 
    # best_train_acc_all = 0

    drop_or_not = args.words_drop
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
            
        model_clip.eval()
        model_tes.train()

        
     
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True,device=device), mask_lab.cuda(non_blocking=True,device=device).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True,device=device)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                img_feats = model_clip.encode_image(images).float()
            
                pseudo_text_feats = model_tes(images, drop_or_not)
                # text_feats_ln = ln_text(text_feats)
        
                
                # align loss
                align_logits_img = logit_scale *  F.normalize(img_feats,dim=-1) @ F.normalize(pseudo_text_feats,dim=-1).t()
                align_logits_text = logit_scale *  F.normalize(pseudo_text_feats,dim=-1) @ F.normalize(img_feats,dim=-1).t()
            
                align_labels_text = torch.arange(pseudo_text_feats.shape[0]).to(device)
                align_labels_img = torch.arange(img_feats.shape[0]).to(device)
            
                align_loss_text = F.cross_entropy(align_logits_text, align_labels_text)
                align_loss_image = F.cross_entropy(align_logits_img, align_labels_img)
            
                align_loss = align_loss_text+align_loss_image
                
                # distill_loss
                mask_twoView = torch.cat([mask_lab,mask_lab])
                pseudo_text_feats_base = pseudo_text_feats[mask_twoView]
                pseudo_text_feats_base = F.normalize(pseudo_text_feats_base, dim=-1)
                label_base = torch.cat([class_labels[mask_lab], class_labels[mask_lab]])
            
                distill_logits, distill_labels = distill_crit(stu_feats=pseudo_text_feats_base,tea_feats=base_Tfeats,labels=label_base, args=args, device=device)
           
                distill_loss = torch.nn.CrossEntropyLoss()(distill_logits, distill_labels) + \
                                mseloss(pseudo_text_feats_base, base_Tfeats[distill_labels]) / pseudo_text_feats_base.shape[0]
                pstr = ''
    
                pstr += f'align_loss: {align_loss.item():.4f} '
                pstr += f'distill_loss: {distill_loss.item():.4f} '


                loss = 0
                loss += align_loss + distill_loss

                
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

    
        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model_tes_projector': model_tes.projector.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        
        torch.save(save_dict, args.tes_model_path)
        args.logger.info("TES model saved to {}.".format(args.tes_model_path))


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

    train_classes = all_classes[:num_train_classes]
    unlabeled_classes = all_classes[num_train_classes:]

    labelled_classes = np.random.choice(train_classes,size=num_train_labeled_classes,replace=False)
    unlabelled_train_classes = np.setdiff1d(train_classes,labelled_classes)
    
    split_info = {
        "seed": args.seed,
        "train_classes":
            train_classes.tolist(),
        "unlabeled_classes":
            unlabeled_classes.tolist()+unlabelled_train_classes.tolist(),
        "train_labelled_classes":
            labelled_classes.tolist(),
        "train_unlabelled_classes":
            unlabelled_train_classes.tolist()
    }
    
    base_name = "class_split_info_run"
    extension = ".json"
    counter = 1
    while True:
        class_split_path = f"{base_name}{counter:02d}{extension}"
        class_split_path = os.path.join(urbanform_root, class_split_path)
        if not os.path.exists(class_split_path):
            break  
        counter += 1
        
    with open(class_split_path, 'w') as f:
        json.dump(split_info, f, indent=2)
        
    args.train_classes = split_info['train_classes']
    args.unlabeled_classes = split_info['unlabeled_classes']
    args.train_labelled_classes = split_info['train_labelled_classes']
    args.train_unlabelled_classes = split_info['train_unlabelled_classes']
    args.class_split_file = class_split_path

    return args
    
if __name__ == "__main__":
    
    

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='urbanform', help='options: urbanform, allform')
    parser.add_argument('--prop_train_labels', type=float, default=0.8)
    parser.add_argument('--use_ssb_splits', action='store_true', default=False)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='debug', type=str)
    parser.add_argument('--exp_id', default='None', type=str)
    
    parser.add_argument('--num_words', default=7, type=int)
    parser.add_argument('--word_dim', default=512, type=int)   
    parser.add_argument('--words_drop_ratio', default=0.5, type=float)
    parser.add_argument('--words_drop',  action='store_true', default=False)
    
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

    args = get_class_split_info(args)    
    
    # get classnames
    class_names_path = os.path.join(get_dir, f"dataset_class_name/{args.dataset_name}_name.npy")
    if not os.path.exists(class_names_path):
        print(f"generate class_names_path for {args.dataset_name}")
        from dataset_class_name import gen_classnames
        gen_classnames.gen(args.dataset_name, class_names_path) 
        
    class_names = np.load(class_names_path)
    
    args.base_names = class_names[args.train_labelled_classes]
    
    #print("train_classes:", class_names[args.train_classes])
    #print("train_labelled_classes:", class_names[args.train_labelled_classes])
    #print("train_unlabelled_classes:", class_names[args.train_unlabelled_classes])
    #print("novel_classes:", class_names[list(set(args.unlabeled_classes)-set(args.train_unlabelled_classes))])
    
    init_experiment(args, runner_name=['TES'], exp_id=args.exp_id)
    
    
    args.interpolation = 3
    args.crop_pct = 0.875

    args.image_size = 224
    args.feat_dim = 512
    args.num_mlp_layers = 3

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
    #print(set(train_dataset.labelled_dataset.targets))
    #print(set(train_dataset.unlabelled_dataset.targets))
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
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    # train_test_loader = DataLoader(train_examples_test, num_workers=args.num_workers,
    #                                     batch_size=256, shuffle=False, pin_memory=False)
    
    
    

    # load model
    model_tes = TES(device=device, num_words=args.num_words, 
                              word_dim=args.word_dim, words_drop_ratio=args.words_drop_ratio)
    
    model_clip, preprocess = clip.load("ViT-B/16",device)


    for m in model_clip.parameters():
        m.requires_grad = False

    
    # train TES
    args.logger.info('Train TES')
    args.tes_model_path = os.path.join('saved_models', 'tes_222_'+args.class_split_file[-10:-5]+'.pt')
    train_tes(model_clip,model_tes,train_loader, args,device)


    
        
    
    
  
