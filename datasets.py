"""
using args select dataset

"""

from dataset.HDRTV_set import HDRTV_set



def Datasets(args=None,train=True,cfg=None):
    
    dataset = HDRTV_set.Dataset(args=args, dataset_train=train,cfg=cfg)
    
    return dataset
