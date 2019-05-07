import os
import numpy as np
import pandas as pd

def f2cat(filename: str) -> str:
    return filename.split('.')[0]

def list_all_categories(input_dir, folder_name):
    files = os.listdir(os.path.join(input_dir, folder_name))
    return sorted([f2cat(f) for f in files], key=str.lower)

def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])