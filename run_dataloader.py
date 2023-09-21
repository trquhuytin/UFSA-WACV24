import numpy as np
import os
from dataloder import DataGenerator



if __name__ == '__main__' :
    
    dataset= '50salads'
    data_root= '/home/ahmed/Ahmed_data/UVAST/UVAST/data'
    split = 1
    dataset=DataGenerator(data_root=data_root,
                 split=1,
                 dataset=dataset,
                 mode='train',
                 transform=None,
                 usebatch=False,
                 args=None,
                 len_seg_max=100,
                 features_path=None,
                 feature_type=".npy",
                 feature_mode="feature")
