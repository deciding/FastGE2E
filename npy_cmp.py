import numpy as np
from glob import glob
import argparse
from tqdm import tqdm

cmp_flag=False
if cmp_flag:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True, help="input dir")
    parser.add_argument("--out_dir", type=str, required=True, help="out dir")
    args = parser.parse_args()

    in_dir=args.in_dir
    out_dir=args.out_dir

    in_npy=glob('%s/**/*.npy' % in_dir)
    out_npy=glob('%s/**/*.npy' % out_dir)

    for infile, outfile in zip(in_npy, out_npy):
        inwav=np.load(infile)
        outwav=np.load(outfile)
        #import pdb;pdb.set_trace()
        mse=np.mean((inwav-outwav) ** 2)
        print(mse, '           ', infile)
    exit()

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, required=True, help="input dir")
args = parser.parse_args()

in_dir=args.in_dir

spk_dirs=glob('%s/*' % in_dir)

all_spkids=[]
for spk_dir in spk_dirs:
    cur_spkids=[]
    wav_spkid_npys=glob('%s/*.npy' % spk_dir)
    for wav_spkid_npy in wav_spkid_npys:
        wav_spkid=np.load(wav_spkid_npy)
        cur_spkids.append(wav_spkid)
    all_spkids.append(cur_spkids)

from numpy import dot
from numpy.linalg import norm

def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

centroids=[np.mean(cur_spkids, axis=0) for cur_spkids in all_spkids]
predicts=[]
labels=[]
for ind, cur_spkids in tqdm(enumerate(all_spkids)):
    for spkid in tqdm(cur_spkids):
        for ind_cent, cent in enumerate(centroids):
            score=cos_sim(spkid, cent)
            if ind==ind_cent:
                label=1
            else:
                label=0
            predicts.append(score)
            labels.append(label)

#import numpy
#import argparse
#import pdb

from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

# ==================== === ====================

#parser = argparse.ArgumentParser(description = "VoxSRC");
#
#parser.add_argument('--ground_truth', type=str, default='data/veri_test.txt', help='Ground truth file');
#parser.add_argument('--prediction', type=str, default='data/veri_test_output.txt', help='Prediction file');
#parser.add_argument('--positive', type=int, default=1, help='1 if higher is positive; 0 is lower is positive');
#
#opt = parser.parse_args();
pos=1

# ==================== === ====================

def calculate_eer(y, y_score, pos):
# y denotes groundtruth scores,
# y_score denotes the prediction scores.

	fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=pos)
        # fpr = fp/n
        # tpr = tp/p
        # fnr = fn/p = 1-tpr
        # tnr = tn/n = 1-fpr
        # since fpr == fnr in this case, the total error rate is fpr
	eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
	thresh = interp1d(fpr, thresholds)(eer)

	return eer, thresh

# ==================== === ====================

eer, thresh = calculate_eer(labels, predicts, pos)

print('EER : %.3f%%'%(eer*100))
