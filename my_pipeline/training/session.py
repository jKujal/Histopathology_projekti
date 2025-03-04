import os
import sys
import random
import time
import torch
import numpy as np

def init_session(args):

    '''

    :param args:
    :return:
    '''


    #Create random seeds for....
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.multi_gpu:
        torch.cuda.manual_seed(args.seed)
    elif args.multi_gpu:
        torch.cuda.manual_seed_all(args.seed)

