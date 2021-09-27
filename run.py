import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import os
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path to GT")
parser.add_argument("--path2save", type=str, help="Path to save")
args = parser.parse_args()

def RLE():
    strings = []
    ids = []
    path = args.path
    path2save = args.path2save
    for pt in tqdm(os.listdir(path)):
        image_id = pt.split('.')[0]
        ids.append(image_id)
        image = cv2.imread(os.path.join(path, pt))[:,:,::-1] #convert BGR => RGB
        string = mask2rle(image)
        strings.append(string)
    df = pd.DataFrame(columns=['id', 'results'])
    df['id'] = ids
    df['results'] = strings
    df.to_csv(path2save, index=False)
if __name__=="__main__":
    RLE()





