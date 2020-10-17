import os   
import glob
import json
import h5py

import numpy as np   
import tqdm as tqdm
import PIL.Image as Image  

import matplotlib.pyplot as plt  
from matplotlib import cm as CM

import scipy
import scipy.io as io
from scipy import spatial
from scipy.ndimage.filters import gaussian_filter

import torch


# function to create density maps for images
def gaussian_filter_density(gt, count):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('Generating Density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('Done.... {index}/{length}'.format(index=count, length=img_path_len))
    print('')
    return density

root = '...input/shanghaitech_with_people_density_map/ShanghaiTech'

# choose part_A or part_B
part = 'part_B'
part_train = os.path.join(root, part, 'train_data', 'images')
part_test = os.path.join(root, part, 'test_data', 'images')
path_sets = [part_train, part_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_path_len = len(img_paths)
count = 1

for img_path in img_paths:
    print(img_path)
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    k = gaussian_filter_density(k, count)
    count += 1
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground-truth'), 'w') as hf:
            hf['density'] = k

gt_file = h5py.File(img_paths[0].replace('.jpg','.h5').replace('images','ground-truth'),'r')
groundtruth = np.asarray(gt_file['density'])
print('[Check] IMG_1 : {sum}'.format(sum=np.sum(groundtruth)))
