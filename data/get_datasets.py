from data.data_utils import MergedDataset

from data.urbanform import get_urbanform_datasets, get_all_datasets



from copy import deepcopy
import json
import os


get_dataset_funcs = {
    'urbanform':get_urbanform_datasets,
    'allform':get_all_datasets,
}

class TargetTransform:
    def __init__(self, mapping):
        self.mapping = mapping
    def __call__(self, x):
        if x in self.mapping:
            return self.mapping[x]
        elif str(x) in self.mapping:
            return self.mapping[str(x)]
        elif int(x) in self.mapping:
            return self.mapping[int(x)]
        else:
            raise KeyError(f"Label {x} not found in mapping {self.mapping.keys()}")
            
class SafeTargetTransform:
    def __init__(self, mapping, unknown_id=None):
        self.mapping = mapping
        self.unknown_id = unknown_id

    def __call__(self, y):
        y = int(y)
        if y in self.mapping:
            return self.mapping[y]
        if self.unknown_id is None:
            raise KeyError(f"Label {y} not in mapping.")
        return self.unknown_id

def build_mapping_from_dataset(ds):
    labels = sorted(set(int(t) for t in ds.targets))
    orig_to_new = {orig: i for i, orig in enumerate(labels)}
    new_to_orig = {i: orig for orig, i in orig_to_new.items()}
    return orig_to_new, new_to_orig

def remap_imagefolder_inplace(ds, mapping):
    ds.samples = [(p, int(mapping[int(y)])) for (p, y) in ds.samples]
    ds.targets = [y for _, y in ds.samples]
    ds.target_transform = None
    return ds

def save_mapping_json(orig_to_new, new_to_orig, save_path, meta=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload = {
        "orig_to_new": {str(k): int(v) for k, v in orig_to_new.items()},
        "new_to_orig": {str(k): int(v) for k, v in new_to_orig.items()},
        "meta": meta or {}
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        
def get_datasets(dataset_name, train_transform, test_transform, args):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError
    print(dataset_name)
    # Get datasets
    if dataset_name == 'urbanform' or dataset_name == 'allform':
        get_dataset_f = get_dataset_funcs[dataset_name]
        datasets = get_dataset_f(train_transform = train_transform, test_transform = test_transform,
                                class_split_file = args.class_split_file,
                                prop_train_labels = args.prop_train_labels,
                                split_train_val = False)
               
    else:
        get_dataset_f = get_dataset_funcs[dataset_name]
        datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                                train_classes=args.train_classes,
                                prop_train_labels=args.prop_train_labels,
                                split_train_val=False)
                                
    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(set(list(args.train_classes) + list(args.unlabeled_classes)))):
        target_transform_dict[cls] = i
    #target_transform = lambda x: target_transform_dict[x]
    target_transform = TargetTransform(target_transform_dict)

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = deepcopy(datasets['test'])
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_test.transform = test_transform
    
    labelled_train_examples_test  = deepcopy(datasets['train_labelled'])
    labelled_train_examples_test.transform = test_transform
    
    train_examples_test = MergedDataset(labelled_dataset=labelled_train_examples_test,
                                  unlabelled_dataset=unlabelled_train_examples_test)

    full_dataset  = deepcopy(datasets['full'])
    full_dataset.transform = test_transform
    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets,train_examples_test, full_dataset


def get_class_splits(args):

    if args.dataset_name == 'allform':

        args.image_size = 224

    elif args.dataset_name == 'urbanform':

        args.image_size = 224
        
    else:

        raise NotImplementedError

    return args
