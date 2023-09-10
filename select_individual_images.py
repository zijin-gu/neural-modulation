import numpy as np
import h5py
from scipy.io import loadmat
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from encoding_tools import load_encoding
import torch
import argparse

parser = argparse.ArgumentParser(prog='get natural images', 
	description='input subject max or rand,save the images',
	usage='python GETIMGS_INI.py --type i --roi j')

parser.add_argument('--type', type=str)
parser.add_argument('--maxroi', type=str)
args = parser.parse_args()

ROIs_subj = []
ROIs1 = ['OFA', 'FFA1', 'FFA2', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
ROIs2 = ['OFA', 'FFA1', 'FFA2', 'aTLfaces', 'EBA', 'FBA2', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
ROIs3 = ['OFA', 'FFA1', 'FFA2', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
ROIs4 = ['OFA', 'FFA1', 'FFA2', 'mTLfaces', 'EBA', 'FBA1', 'FBA2', 'mTLbodies', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
ROIs5 = ['OFA', 'FFA1', 'FFA2', 'mTLfaces', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'mTLbodies', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
ROIs6 = ['OFA', 'FFA1', 'FFA2', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'mTLbodies', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
ROIs7 = ['OFA', 'FFA1', 'FFA2', 'mTLfaces', 'aTLfaces', 'EBA', 'FBA2', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
ROIs8 = ['OFA', 'FFA1', 'FFA2', 'mTLfaces', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'mTLbodies', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
ROIs_subj.append(ROIs1)
ROIs_subj.append(ROIs2)
ROIs_subj.append(ROIs3)
ROIs_subj.append(ROIs4)
ROIs_subj.append(ROIs5)
ROIs_subj.append(ROIs6)
ROIs_subj.append(ROIs7)
ROIs_subj.append(ROIs8)

device = torch.device("cuda")

nsd_root = "/home/zg243/nsd/"
stim_root = nsd_root + "stimuli/"
exp_design_file = nsd_root + "experiments/nsd_expdesign.mat"
exp_design = loadmat(exp_design_file)
ordering = exp_design['masterordering'].flatten() - 1 # zero-indexed ordering of indices (matlab-like to python-like)

fwrf, fmaps = [], []

for subject in range(1,9):
    tmp0, tmp1 = load_encoding(subject=subject, model_name='dnn_fwrf', device=device)
    fwrf.append(tmp0)
    fmaps.append(tmp1)
    
allsubj_pred = {}
allsubj_img = {}
trials = np.array([30000, 30000, 24000, 22500, 30000, 24000, 30000, 22500])

for subject in range(1,9):
    print('subject %d'%subject)
    
    image_data_set = h5py.File(stim_root + "S%d_stimuli_227.h5py"%subject, 'r')
    image_data = np.copy(image_data_set['stimuli']).astype(np.float32) / 255.
    image_data_set.close()

    data_size = trials[subject-1]
    ordering_data = ordering[:data_size]
    shared_mask   = ordering_data<1000  # the first 1000 indices are the shared indices

    stim_data = image_data[ordering_data]  # reduce to only the samples available thus far

    trn_stim_data = stim_data[~shared_mask]
    
    del image_data, stim_data
    maxroi_idx = ROIs_subj[subject-1].index(args.maxroi)
    pred = np.zeros([len(trn_stim_data), len(ROIs_subj[subject-1])])
    j = 0
    while (j+1)*100 < len(trn_stim_data):
        pred[j*100:(j+1)*100] = fwrf[subject-1](fmaps[subject-1](torch.from_numpy(trn_stim_data[j*100:(j+1)*100]).to(device))).cpu().detach().numpy()
        j += 1

    pred[j*100:] = fwrf[subject-1](fmaps[subject-1](torch.from_numpy(trn_stim_data[j*100:]).to(device))).cpu().detach().numpy()
    
    if args.type == 'max':
        h2l_idx = np.argsort(pred[:,maxroi_idx])[::-1] # for max natural
    if args.type == 'rand':
        h2l_idx = np.argsort(abs(pred[:,maxroi_idx]))
    m = 0
    plt.figure(figsize=(20,8))
    filtered_pred = []
    filtered_img = []
    for i in range(200):
        if pred[h2l_idx[i], maxroi_idx] <= 0:
            continue
        if i==0:
            filtered_pred.append(pred[h2l_idx[i],maxroi_idx])
            filtered_img.append(trn_stim_data[h2l_idx[i]])
        else:
            if round(pred[h2l_idx[i], maxroi_idx],3) != round(pred[h2l_idx[i-1], maxroi_idx],3):
                filtered_pred.append(pred[h2l_idx[i],maxroi_idx])
                filtered_img.append(trn_stim_data[h2l_idx[i]])
            
    del pred
    allsubj_pred['S%d'%subject] = filtered_pred
    allsubj_img['S%d'%subject] = filtered_img
    
np.save('natural_pred_allsubj.npy', allsubj_pred)
np.save('natural_img_allsubj.npy', allsubj_img)
