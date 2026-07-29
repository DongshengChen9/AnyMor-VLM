import numpy as np
import os
from config import get_dir

def gen(dataset_name, save_class_names_path):
    if dataset_name == 'allform':

        import pandas as pd

        urbanform_classes_path = os.path.join(get_dir, "dataset_class_name/allform_types.txt")
        df = pd.read_csv(urbanform_classes_path)
        classnames = df["MorphologicalName"].tolist()
        classdescrip = df["Description"].tolist()
        assert len(classnames) == 6    
        np.save('./dataset_class_name/allform_descriptions.npy',np.array(classdescrip))

    elif dataset_name == 'urbanform':

        import pandas as pd

        urbanform_classes_path = os.path.join(get_dir, "dataset_class_name/urbanform_types.txt")
        df = pd.read_csv(urbanform_classes_path)
        classnames = df["MorphologicalName"].tolist()
        classdescrip = df["Description"].tolist()
        assert len(classnames) == 6    
        np.save('./dataset_class_name/urbanform_descriptions.npy',np.array(classdescrip))
   
    else:

        raise NotImplementedError

    
    np.save(save_class_names_path,np.array(classnames))
    
    return

if __name__ == "__main__":
    gen("cub", None)