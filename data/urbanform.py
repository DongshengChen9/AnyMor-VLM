import os
import json
import torchvision
import numpy as np
from copy import deepcopy

from data.data_utils import subsample_instances


from copy import deepcopy
from config import urbanform_root, allform_root


class UrbanFormDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)
        self.uq_idxs = np.arange(len(self))

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        uq_idx = self.uq_idxs[idx]
        return img, label, uq_idx


def subsample_dataset(dataset, idxs):
    """Subsample dataset by index list."""
    mask = np.zeros(len(dataset)).astype(bool)
    mask[idxs] = True

    dataset.samples = np.array(dataset.samples)[mask].tolist()
    dataset.targets = np.array(dataset.targets)[mask].tolist()
    dataset.uq_idxs = dataset.uq_idxs[mask]
    return dataset

def subsample_classes(dataset, include_classes=range(250)):

    cls_idxs = [x for x, l in enumerate(dataset.targets) if l in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_instances_per_class=5):

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        # Have a balanced test set
        v_ = np.random.choice(cls_idxs, replace=False, size=(val_instances_per_class,))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

def dataset_summary(name, dataset):
        if dataset is None:
            print(f"{name}: None")
            return
        targets = np.array(dataset.targets)
        unique, counts = np.unique(targets, return_counts=True)
        print(f"\n{name} summary:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Classes: {unique.tolist()}")
        print(f"  Samples per class:")
        for u, c in zip(unique, counts):
            print(f"    Class {u}: {c}")

def get_all_datasets(
        train_transform,
        test_transform,
        class_split_file,
        prop_train_labels=0.8,
        seed=42, 
        split_train_val=False
    ):
    with open(class_split_file, 'r', encoding='utf-8') as file:
        class_split_info = json.load(file)
    train_classes = class_split_info['train_classes']
    unlabeled_classes = class_split_info['unlabeled_classes']
    train_labelled_classes = class_split_info['train_labelled_classes']
    train_unlabelled_classes = class_split_info['train_unlabelled_classes']
    train_ratio = 0.8

    np.random.seed(seed)
    data_root = allform_root
    json_path = os.path.join(data_root, "split_info.json")
    full_dataset = UrbanFormDataset(root=data_root, transform=train_transform)
    #print(full_dataset.class_to_idx)

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            split_info = json.load(f)
        train_indices = split_info["train_indices"]
        test_indices = split_info["test_indices"]

    else:
        all_targets = np.array(full_dataset.targets)
        train_class_mask = np.isin(all_targets, train_classes)
        candidate_train_indices = np.where(train_class_mask)[0].tolist()
        candidate_test_indices = np.where(~train_class_mask)[0].tolist()
    
        np.random.shuffle(candidate_train_indices)
        np.random.shuffle(candidate_test_indices)
        #np.random.shuffle(all_targets)  # optional, just for consistency

        split_point = int(len(candidate_train_indices) * train_ratio)
        train_indices = candidate_train_indices[:split_point]
        test_indices_part1 = candidate_train_indices[split_point:]

        split_point2 = int(len(candidate_test_indices) * train_ratio)
        test_indices_part2 = candidate_test_indices[split_point2:]
        test_indices = test_indices_part1 + test_indices_part2

        train_indices = [int(i) for i in train_indices]
        test_indices = [int(i) for i in test_indices]

        split_info = {
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_ratio": train_ratio,
            "seed": seed
        }
        with open(json_path, "w") as f:
            json.dump(split_info, f, indent=2)

        #train_dataset = deepcopy(full_dataset)


    train_dataset = deepcopy(full_dataset)  
    test_dataset = deepcopy(full_dataset)

    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
    train_dataset.targets = [full_dataset.targets[i] for i in train_indices]
    #train_dataset.uq_idxs = np.array(train_indices)
    train_dataset.uq_idxs = full_dataset.uq_idxs[train_indices]
    train_dataset.transform = train_transform

    test_dataset.samples = [full_dataset.samples[i] for i in test_indices]
    test_dataset.targets = [full_dataset.targets[i] for i in test_indices]
    #test_dataset.uq_idxs = np.array(test_indices)
    test_dataset.uq_idxs = full_dataset.uq_idxs[test_indices]
    test_dataset.transform = test_transform

    train_dataset.targets = [int(t) for t in train_dataset.targets]
    test_dataset.targets = [int(t) for t in test_dataset.targets]
    

    #dataset_summary("train_dataset", train_dataset)
    train_dataset = subsample_classes(deepcopy(train_dataset), include_classes=train_classes)
    #dataset_summary("train_dataset", train_dataset)
    train_dataset_labelled = subsample_classes(deepcopy(train_dataset), include_classes=train_labelled_classes)
    #train_dataset_labelled = deepcopy(train_dataset)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    if split_train_val:
        train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled,
                                                     val_instances_per_class=5)
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        val_dataset_labelled_split.transform = test_transform

    else:
        train_dataset_labelled_split, val_dataset_labelled_split = None, None

    unlabelled_indices = set(train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    uq_to_local = {uq: i for i, uq in enumerate(train_dataset.uq_idxs)}
    valid_unlabelled_local_idxs = np.array([uq_to_local[uq] for uq in unlabelled_indices if uq in uq_to_local])
    #print(unlabelled_indices)
    #print(train_dataset_labelled.uq_idxs)
    
    
    train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset), valid_unlabelled_local_idxs)
    #print("labelled set unique labels:", np.unique(train_dataset_labelled.targets))
    #print("Unlabelled set unique labels:", np.unique(train_dataset_unlabelled.targets))
    #print("Test set unique labels:", np.unique(test_dataset.targets))
    #print(train_dataset_unlabelled.uq_idxs)

    #unlabelled_classes = list(set(train_dataset.targets) - set(train_labelled_classes))
    target_xform_dict = {}
    for i, k in enumerate(list(np.unique(train_dataset_unlabelled.targets))):
        target_xform_dict[k] = i

    #test_dataset.target_transform = lambda x: target_xform_dict[x]
    train_dataset_unlabelled.target_transform = lambda x: target_xform_dict[x]
    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    full_dataset.transform = test_transform
    #full_dataset.target_transform = lambda x: target_xform_dict[x]

    #dataset_summary("full_dataset", full_dataset)
    #dataset_summary("train_dataset", train_dataset)
    #dataset_summary("train_dataset_labelled", train_dataset_labelled)
    #dataset_summary("train_dataset_unlabelled", train_dataset_unlabelled)
    #dataset_summary("test_dataset", test_dataset)


    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
        'full': full_dataset
    }

    return all_datasets

def get_urbanform_datasets(
        train_transform,
        test_transform,
        class_split_file,
        prop_train_labels=0.8,
        seed=42, 
        split_train_val=False
    ):
    with open(class_split_file, 'r', encoding='utf-8') as file:
        class_split_info = json.load(file)
    train_classes = class_split_info['train_classes']
    unlabeled_classes = class_split_info['unlabeled_classes']
    train_labelled_classes = class_split_info['train_labelled_classes']
    train_unlabelled_classes = class_split_info['train_unlabelled_classes']
    print("run count: ", class_split_file[-10:-5])
    #print("train_classes: ", train_classes)
    #print("train_labelled_classes: ", train_labelled_classes)

    train_ratio = 0.8

    np.random.seed(seed)
    data_root = urbanform_root
    json_path = os.path.join(data_root, "split_info_fix_train_test.json")
    full_dataset = UrbanFormDataset(root=data_root, transform=train_transform)

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            split_info = json.load(f)
        train_indices = split_info["train_indices"]
        test_indices = split_info["test_indices"]

    else:
        all_targets = np.array(full_dataset.targets)
        train_class_mask = np.isin(all_targets, train_classes)
        candidate_train_indices = np.where(train_class_mask)[0].tolist()
        candidate_test_indices = np.where(~train_class_mask)[0].tolist()

        np.random.shuffle(candidate_train_indices)
        np.random.shuffle(candidate_test_indices)
        #np.random.shuffle(all_targets)  # optional, just for consistency

        split_point = int(len(candidate_train_indices) * train_ratio)
        train_indices = candidate_train_indices[:split_point]
        test_indices_part1 = candidate_train_indices[split_point:]

        split_point2 = int(len(candidate_test_indices) * train_ratio)
        test_indices_part2 = candidate_test_indices[split_point2:]
        test_indices = test_indices_part1 + test_indices_part2

        train_indices = [int(i) for i in train_indices]
        test_indices = [int(i) for i in test_indices]

        split_info = {
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_ratio": train_ratio,
            "seed": seed,
            "class_split_file": class_split_file,
            "train_classes": train_classes,
            "train_labelled_classes": train_labelled_classes,
            "train_unlabelled_classes": train_unlabelled_classes,
            "unlabeled_classes": unlabeled_classes
        }
        with open(json_path, "w") as f:
            json.dump(split_info, f, indent=2)

        #train_dataset = deepcopy(full_dataset)

    train_dataset = deepcopy(full_dataset)  
    test_dataset = deepcopy(full_dataset)

    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
    train_dataset.targets = [full_dataset.targets[i] for i in train_indices]
    #train_dataset.uq_idxs = np.array(train_indices)
    train_dataset.uq_idxs = full_dataset.uq_idxs[train_indices]
    train_dataset.transform = train_transform

    test_dataset.samples = [full_dataset.samples[i] for i in test_indices]
    test_dataset.targets = [full_dataset.targets[i] for i in test_indices]
    #test_dataset.uq_idxs = np.array(test_indices)
    test_dataset.uq_idxs = full_dataset.uq_idxs[test_indices]
    test_dataset.transform = test_transform

    train_dataset.targets = [int(t) for t in train_dataset.targets]
    test_dataset.targets = [int(t) for t in test_dataset.targets]
    

    #dataset_summary("train_dataset", train_dataset)
    train_dataset = subsample_classes(deepcopy(train_dataset), include_classes=train_classes)
    #dataset_summary("train_dataset", train_dataset)
    train_dataset_labelled = subsample_classes(deepcopy(train_dataset), include_classes=train_labelled_classes)
    #train_dataset_labelled = deepcopy(train_dataset)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    if split_train_val:
        train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled,
                                                     val_instances_per_class=5)
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        val_dataset_labelled_split.transform = test_transform

    else:
        train_dataset_labelled_split, val_dataset_labelled_split = None, None

    unlabelled_indices = set(train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    uq_to_local = {uq: i for i, uq in enumerate(train_dataset.uq_idxs)}
    valid_unlabelled_local_idxs = np.array([uq_to_local[uq] for uq in unlabelled_indices if uq in uq_to_local])
    #print(unlabelled_indices)
    #print(train_dataset_labelled.uq_idxs)
    
    
    train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset), valid_unlabelled_local_idxs)
    #print("labelled set unique labels:", np.unique(train_dataset_labelled.targets))
    #print("Unlabelled set unique labels:", np.unique(train_dataset_unlabelled.targets))
    #print("Test set unique labels:", np.unique(test_dataset.targets))
    #print(train_dataset_unlabelled.uq_idxs)

    #unlabelled_classes = list(set(train_dataset.targets) - set(train_labelled_classes))
    target_xform_dict = {}
    for i, k in enumerate(list(np.unique(train_dataset_unlabelled.targets))):
        target_xform_dict[k] = i

    #test_dataset.target_transform = lambda x: target_xform_dict[x]
    train_dataset_unlabelled.target_transform = lambda x: target_xform_dict[x]
    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    full_dataset.transform = test_transform
    #full_dataset.target_transform = lambda x: target_xform_dict[x]

    #dataset_summary("full_dataset", full_dataset)
    #dataset_summary("train_dataset", train_dataset)
    #dataset_summary("train_dataset_labelled", train_dataset_labelled)
    #dataset_summary("train_dataset_unlabelled", train_dataset_unlabelled)
    #dataset_summary("test_dataset", test_dataset)


    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
        'full': full_dataset
    }

    return all_datasets

def get_caseform_datasets(
        train_transform,
        test_transform,
        train_classes,
        prop_train_labels=0.8,
        seed=42, 
        split_train_val=False
    ):
    print("train_classes: ", train_classes)
    train_labelled_classes = range(2)
    train_unlabelled_classes = range(2,5)
    train_ratio = 0.8

    np.random.seed(seed)
    data_root = caseform_root
    json_path = os.path.join(data_root, "split_info.json")
    full_dataset = UrbanFormDataset(root=data_root, transform=train_transform)

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            split_info = json.load(f)
        train_indices = split_info["train_indices"]
        test_indices = split_info["test_indices"]

    else:
        all_targets = np.array(full_dataset.targets)
        train_class_mask = np.isin(all_targets, train_classes)
        candidate_train_indices = np.where(train_class_mask)[0].tolist()
        candidate_test_indices = np.where(~train_class_mask)[0].tolist()
        # 打乱仅属于指定类别的样本
        np.random.shuffle(candidate_train_indices)
        np.random.shuffle(candidate_test_indices)
        #np.random.shuffle(all_targets)  # optional, just for consistency

        split_point = int(len(candidate_train_indices) * train_ratio)
        train_indices = candidate_train_indices[:split_point]
        test_indices_part1 = candidate_train_indices[split_point:]

        split_point2 = int(len(candidate_test_indices) * train_ratio)
        test_indices_part2 = candidate_test_indices[split_point2:]
        test_indices = test_indices_part1 + test_indices_part2

        train_indices = [int(i) for i in train_indices]
        test_indices = [int(i) for i in test_indices]

        split_info = {
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_ratio": train_ratio,
            "seed": seed,
            "train_labelled_classes": [int(c) for c in train_labelled_classes],
            "train_unlabelled_classes": [int(c) for c in train_unlabelled_classes],
        }
        with open(json_path, "w") as f:
            json.dump(split_info, f, indent=2)

        train_dataset = deepcopy(full_dataset)

    train_dataset = deepcopy(full_dataset)  
    test_dataset = deepcopy(full_dataset)

    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
    train_dataset.targets = [full_dataset.targets[i] for i in train_indices]
    train_dataset.uq_idxs = np.array(train_indices)
    train_dataset.transform = train_transform

    test_dataset.samples = [full_dataset.samples[i] for i in test_indices]
    test_dataset.targets = [full_dataset.targets[i] for i in test_indices]
    test_dataset.uq_idxs = np.array(test_indices)
    test_dataset.transform = test_transform

    train_dataset.targets = [int(t) for t in train_dataset.targets]
    test_dataset.targets = [int(t) for t in test_dataset.targets]

    train_dataset_labelled = subsample_classes(deepcopy(train_dataset), include_classes=train_labelled_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    if split_train_val:
        train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled,
                                                     val_instances_per_class=5)
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        val_dataset_labelled_split.transform = test_transform

    else:
        train_dataset_labelled_split, val_dataset_labelled_split = None, None

    unlabelled_indices = set(train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    uq_to_local = {uq: i for i, uq in enumerate(train_dataset.uq_idxs)}
    valid_unlabelled_local_idxs = np.array([uq_to_local[uq] for uq in unlabelled_indices if uq in uq_to_local])
    #print(unlabelled_indices)
    #print(train_dataset_labelled.uq_idxs)
    
    
    train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset), valid_unlabelled_local_idxs)
    #print("labelled set unique labels:", np.unique(train_dataset_labelled.targets))
    #print("Unlabelled set unique labels:", np.unique(train_dataset_unlabelled.targets))
    #print("Test set unique labels:", np.unique(test_dataset.targets))
    #print(train_dataset_unlabelled.uq_idxs)

    #unlabelled_classes = list(set(train_dataset.targets) - set(train_labelled_classes))
    target_xform_dict = {}
    for i, k in enumerate(list(np.unique(full_dataset.targets))):
        target_xform_dict[k] = i

    test_dataset.target_transform = lambda x: target_xform_dict[x]
    train_dataset_unlabelled.target_transform = lambda x: target_xform_dict[x]
    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    full_dataset.transform = test_transform
    full_dataset.target_transform = lambda x: target_xform_dict[x]

    dataset_summary("full_dataset", full_dataset)
    dataset_summary("train_dataset_labelled", train_dataset_labelled)
    dataset_summary("train_dataset_unlabelled", train_dataset_unlabelled)
    dataset_summary("test_dataset", test_dataset)


    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
        'full': full_dataset
    }

    return all_datasets

def get_extraform_datasets(
        train_transform,
        test_transform,
        train_classes,
        prop_train_labels=0.8,
        seed=42, 
        split_train_val=False
    ):
    print("train_classes: ", train_classes)
    train_labelled_classes = range(2)
    train_unlabelled_classes = range(2,5)
    train_ratio = 0.8

    np.random.seed(seed)
    data_root = extraform_root
    json_path = os.path.join(data_root, "split_info.json")
    full_dataset = UrbanFormDataset(root=data_root, transform=train_transform)

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            split_info = json.load(f)
        train_indices = split_info["train_indices"]
        test_indices = split_info["test_indices"]

    else:
        all_targets = np.array(full_dataset.targets)
        train_class_mask = np.isin(all_targets, train_classes)
        candidate_train_indices = np.where(train_class_mask)[0].tolist()
        candidate_test_indices = np.where(~train_class_mask)[0].tolist()
        # 打乱仅属于指定类别的样本
        np.random.shuffle(candidate_train_indices)
        np.random.shuffle(candidate_test_indices)
        #np.random.shuffle(all_targets)  # optional, just for consistency

        split_point = int(len(candidate_train_indices) * train_ratio)
        train_indices = candidate_train_indices[:split_point]
        test_indices_part1 = candidate_train_indices[split_point:]

        split_point2 = int(len(candidate_test_indices) * train_ratio)
        test_indices_part2 = candidate_test_indices[split_point2:]
        test_indices = test_indices_part1 + test_indices_part2

        train_indices = [int(i) for i in train_indices]
        test_indices = [int(i) for i in test_indices]

        split_info = {
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_ratio": train_ratio,
            "seed": seed,
            "train_labelled_classes": [int(c) for c in train_labelled_classes],
            "train_unlabelled_classes": [int(c) for c in train_unlabelled_classes],
        }
        with open(json_path, "w") as f:
            json.dump(split_info, f, indent=2)

        train_dataset = deepcopy(full_dataset)

    train_dataset = deepcopy(full_dataset)  
    test_dataset = deepcopy(full_dataset)

    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
    train_dataset.targets = [full_dataset.targets[i] for i in train_indices]
    train_dataset.uq_idxs = np.array(train_indices)
    train_dataset.transform = train_transform

    test_dataset.samples = [full_dataset.samples[i] for i in test_indices]
    test_dataset.targets = [full_dataset.targets[i] for i in test_indices]
    test_dataset.uq_idxs = np.array(test_indices)
    test_dataset.transform = test_transform

    train_dataset.targets = [int(t) for t in train_dataset.targets]
    test_dataset.targets = [int(t) for t in test_dataset.targets]

    train_dataset_labelled = subsample_classes(deepcopy(train_dataset), include_classes=train_labelled_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    if split_train_val:
        train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled,
                                                     val_instances_per_class=5)
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        val_dataset_labelled_split.transform = test_transform

    else:
        train_dataset_labelled_split, val_dataset_labelled_split = None, None

    unlabelled_indices = set(train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    uq_to_local = {uq: i for i, uq in enumerate(train_dataset.uq_idxs)}
    valid_unlabelled_local_idxs = np.array([uq_to_local[uq] for uq in unlabelled_indices if uq in uq_to_local])
    #print(unlabelled_indices)
    #print(train_dataset_labelled.uq_idxs)
    
    
    train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset), valid_unlabelled_local_idxs)
    print("labelled set unique labels:", np.unique(train_dataset_labelled.targets))
    print("Unlabelled set unique labels:", np.unique(train_dataset_unlabelled.targets))
    print("Test set unique labels:", np.unique(test_dataset.targets))
    #print(train_dataset_unlabelled.uq_idxs)

    #unlabelled_classes = list(set(train_dataset.targets) - set(train_labelled_classes))
    target_xform_dict = {}
    for i, k in enumerate(list(np.unique(full_dataset.targets))):
        target_xform_dict[k] = i

    test_dataset.target_transform = lambda x: target_xform_dict[x]
    train_dataset_unlabelled.target_transform = lambda x: target_xform_dict[x]
    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    full_dataset.transform = test_transform
    full_dataset.target_transform = lambda x: target_xform_dict[x]

    dataset_summary("full_dataset", full_dataset)
    dataset_summary("train_dataset_labelled", train_dataset_labelled)
    dataset_summary("train_dataset_unlabelled", train_dataset_unlabelled)
    dataset_summary("test_dataset", test_dataset)


    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
        'full': full_dataset
    }

    return all_datasets
# if __name__ == '__main__':
#     np.random.seed(0)
#     train_classes = [0,1,2,3]

#     x = get_urbanform_datasets(None, None, train_classes=train_classes,
#                                prop_train_labels=0.5)

#     assert set(x['train_unlabelled'].targets) == set(range(3))

#     print('Printing lens...')
#     for k, v in x.items():
#         if v is not None:
#             print(f'{k}: {len(v)}')

#     print('Printing labelled and unlabelled overlap...')
#     print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
#     print('Printing total instances in train...')
#     print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))
#     print('Printing number of labelled classes...')
#     print(len(set(x['train_labelled'].targets)))
#     print('Printing total number of classes...')
#     print(len(set(x['train_unlabelled'].targets)))

#     print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
#     print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
#     print(f'Len labelled set: {len(x["train_labelled"])}')
#     print(f'Len unlabelled set: {len(x["train_unlabelled"])}')
