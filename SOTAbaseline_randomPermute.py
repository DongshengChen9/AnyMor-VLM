"""Random-permutation baseline for the AnyMor comparative evaluation."""

import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from util.caption_utils import CaptionResolver
from util.general_utils import init_experiment
from util.cluster_and_log_utils import log_accs_from_preds_v4
from config import exp_root, get_dir
from model import DINOHead, ContrastiveLearningViewGenerator, JointProjector, TES

from clip import clip
import torch.nn.functional as F
import json
from config import caption_root
    
def permutation_baseline(y_true, y_pred, mask_seen, mask_new, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        y_pred_perm = rng.permutation(y_pred)  
        stats.append(log_accs_from_preds_v4(y_true=y_true, y_pred=y_pred_perm, mask_seen=mask_seen, mask_new=mask_new,
                                                    T=0, eval_funcs=args.eval_funcs, save_name='test_randomPermute',
                                                    args=args))
    return np.array(stats)

def model_inference(backbone, projector, model_tes, caption_resolver,ln_t, jointer, data_loader, ckpt_path, args, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt['backbone'])
    projector.load_state_dict(ckpt['proj'])
    ln_t.load_state_dict(ckpt['ln_t'])
    jointer.load_state_dict(ckpt['jointer'])
    model_tes.projector.load_state_dict(ckpt['model_tes_projector'])

    backbone.to(device).eval()
    projector.to(device).eval()
    ln_t.to(device).eval()
    jointer.to(device).eval()
    model_tes.to(device).eval()


    preds_img, preds_text, preds_mix, targets = [], [], [], []
    mask_seen = np.array([])
    mask_new = np.array([])
   
    for batch_idx, (images, label, uq_idxs) in enumerate(tqdm(data_loader)):
        images = images.to(device, non_blocking=True)
        if isinstance(uq_idxs, torch.Tensor):
            uq_idxs_cpu = uq_idxs.detach().cpu().numpy()
        else:
            uq_idxs_cpu = np.asarray(uq_idxs)
        captions_batch = caption_resolver.get_batch(uq_idxs_cpu)
        caption_tokens = clip.tokenize(captions_batch,truncate=True,).to(device)

        with torch.no_grad():
            image_features = backbone.encode_image(images).float()
            text_features = model_tes(images)
            text_features_ln = ln_t(text_features)
            caption_features = backbone.encode_text(caption_tokens).float()
            joint_features = jointer([image_features, text_features_ln, caption_features])
            #embed_text, logits_text = projector(text_features_ln)
            #embed_img, logits_img = projector(image_features)
            embed_mix, logits_mix = projector(joint_features)


            #preds_img.append(logits_img.argmax(1).cpu().numpy())
            #preds_text.append(logits_text.argmax(1).cpu().numpy())
            preds_mix.append(logits_mix.argmax(1).cpu().numpy())

            targets.append(label.cpu().numpy())
            mask_new = np.append(mask_new, np.array([True if x.item() not in args.train_classes else False for x in label]))
            mask_seen = np.append(mask_seen, np.array([True if x.item() in args.train_labelled_classes else False for x in label]))
        
    #preds_img = np.concatenate(preds_img)
    #preds_text = np.concatenate(preds_text)
    preds_mix = np.concatenate(preds_mix)
    
    targets = np.concatenate(targets)

    all_acc_mix, old_acc_mix, self_acc_mix, new_acc_mix, all_ari_mix, old_ari_mix, self_ari_mix, new_ari_mix, all_nmi_mix, old_nmi_mix, self_nmi_mix, new_nmi_mix  = log_accs_from_preds_v4(y_true=targets, y_pred=preds_mix, mask_seen=mask_seen, mask_new=mask_new,
                                                    T=0, eval_funcs=args.eval_funcs, save_name='test_randomPermute',
                                                    args=args)
    
    #args.logger.info('Test Accuracies(img): All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_acc_img, old_acc_img, semi_acc_img, new_acc_img))
    #args.logger.info('Test Accuracies(text): All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_acc_text, old_acc_text, semi_acc_text, new_acc_text))
    args.logger.info('AnyMor-VLM ACC(mix): All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_acc_mix, old_acc_mix, self_acc_mix, new_acc_mix))
    args.logger.info('AnyMor-VLM ARI(mix): All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_ari_mix, old_ari_mix, self_ari_mix, new_ari_mix))
    args.logger.info('AnyMor-VLM NMI(mix): All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_nmi_mix, old_nmi_mix, self_nmi_mix, new_nmi_mix))

    rnd_results = permutation_baseline(targets, preds_mix, mask_seen, mask_new, n=1000, seed=0)
    means = np.mean(rnd_results, axis=0) 
    stds = np.std(rnd_results, axis=0)    
    args.logger.info('Random Permute ACC: All {:.4f}±{:.4f} | Old {:.4f}±{:.4f} | Self {:.4f}±{:.4f} | New {:.4f}±{:.4f}'.format(means[0], stds[0], means[1], stds[1], means[2], stds[2], means[3], stds[3]))
    args.logger.info('Random Permute ARI: All {:.4f}±{:.4f} | Old {:.4f}±{:.4f} | Self {:.4f}±{:.4f} | New {:.4f}±{:.4f}'.format(means[4], stds[4], means[5], stds[5], means[6], stds[6], means[7], stds[7]))
    args.logger.info('Random Permute NMI: All {:.4f}±{:.4f} | Old {:.4f}±{:.4f} | Self {:.4f}±{:.4f} | New {:.4f}±{:.4f}'.format(means[8], stds[8], means[9], stds[9], means[10], stds[10], means[11], stds[11]))


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
    #print(args.train_classes)
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

    parser.add_argument("--captions_path", type=str, default=os.path.join(caption_root, "captions_6cls.npz"), help="Path to the caption NPZ from the data repository.")
    parser.add_argument("--caption_lookup_mode",type=str,choices=["id", "index", "auto"],default="id",help=("How uq_idxs should be interpreted: ""'id', positional 'index', or 'auto'."))
    
    parser.add_argument('--num_classes4test', type=int, default=6, help='number of classes for testing')
    parser.add_argument('--class_split_file', type=str, default='data/class_split_info_run00.json', help='file path to split classes')
    parser.add_argument('--random_class_split', action='store_true', default=False, help='if true, randomly split Labeled-Unlabeled-Novel equally. if false, load --class_split_file')
    parser.add_argument('--test_model', type=str, default='Test_model/anymor_0.15w_222_0.6874_0.4481.pt', help='step2 model path')
    
    
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
    
    
    init_experiment(args, runner_name=['RandomPermute'], exp_id=args.exp_id)
    
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

    #print(f"train_dataset: {len(train_dataset)}")
    #print(f"Unlabelled_train_dataset: {len(unlabelled_train_examples_test)}")
    #print(f"train_examples_test: {len(train_examples_test)}")
    #print(f"test_dataset: {len(test_dataset)}")
    
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

    model_tes = TES(device=device, num_words=args.num_words, 
                              word_dim=args.word_dim, words_drop_ratio=args.words_drop_ratio)
    
    if args.TES_model_dir is not None:
        args.logger.info(f'Loading weights from {args.TES_model_dir}')
        model_tes.projector.load_state_dict(torch.load(args.TES_model_dir, map_location="cpu")['model_tes_projector'])
    else:    
        model_tes.projector.load_state_dict(torch.load(args.tes_model_path, map_location="cpu")['model_tes_projector'])
    
    model_clip, preprocess = clip.load("ViT-B/16",device)

    for m in model_tes.parameters():
        m.requires_grad = False
        
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
    
    caption_resolver = CaptionResolver(npz_path=args.captions_path, lookup_mode=args.caption_lookup_mode)
    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=768, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers).to(device)
    ln_t = nn.Linear(in_features=512,out_features=768).to(device)  
    jointer = JointProjector(dims=[768, 768, 512], out_dim=768).to(device)
    args.logger.info(f'Random permute predictions......')
    model_inference(model_clip, projector, model_tes, caption_resolver, ln_t, jointer, test_loader, args.test_model, args, device)
    
    
        
    
    
  
