import cv2
import pandas as pd
import numpy as np
from scipy import stats
import lpips
import torch

#### change vgg to alex to use the alex net ######
network = 'alex'


loss_fn = lpips.LPIPS(net=network)

def lpips_compare(gt_file_name, re_col_file_name):
    gt_image = cv2.imread(f'./HECDImages/{gt_file_name}')
    re_col_image = cv2.imread(f'./HECDImages/{re_col_file_name}')
    if np.shape(gt_image)[0] < np.shape(gt_image)[1]:
                gt_image = cv2.rotate(gt_image, cv2.cv2.ROTATE_90_CLOCKWISE)
                re_col_image = cv2.rotate(re_col_image, cv2.cv2.ROTATE_90_CLOCKWISE)
    print(np.shape(gt_image))
    d = 0
    for i in range(0,7):
        for j in range(0,5):


            img0 = gt_image[i*64:i*64+64, j*64:j*64+64,:]

            img0 = np.reshape(img0, (1,3,64,64))
            img0 = (img0/127.5)-1


            img0 = torch.tensor(img0).float()


            img1 = re_col_image[i*64:i*64+64, j*64:j*64+64,:]
            img1 = np.reshape(img1, (1,3,64,64))
            img1 = (img1/127.5)-1
            img1 = torch.tensor(img1).float()


            d += loss_fn(img0, img1).cpu().detach().numpy()

    return d




df = pd.read_csv('./bokeh_app/HumanAggregatedResults.csv')
gt_files = df['GTFileName'].unique()

human_results = {}
lpips_results = {}
for gt_file_name in gt_files:
    print(gt_file_name)
    re_col_files = df[df['GTFileName']==gt_file_name]['ReColFileName'].unique()
    re_col_dict = {}
    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= df[df['ReColFileName']==re_col_file_name]['zScore'].iloc[0]
    human_results[gt_file_name] = re_col_dict
    re_col_dict = {}

    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= lpips_compare(gt_file_name, re_col_file_name)
    lpips_results[gt_file_name] = re_col_dict

################################# Spearman  ######################################
human_results_array = np.zeros(66)
lpips_results_array = np.zeros(66)
row = []
for gt_file_name in gt_files:
    i = 0
    re_col_files = df[df['GTFileName']==gt_file_name]['ReColFileName'].unique()

    for re_col_file_name in re_col_files:
        human_results_array[i] = human_results[gt_file_name][re_col_file_name]
        lpips_results_array[i] = lpips_results[gt_file_name][re_col_file_name]
        i+=1
    row.append([gt_file_name, np.round(stats.spearmanr(human_results_array, lpips_results_array)[0], 3), np.round(stats.spearmanr(human_results_array, lpips_results_array)[1],3)])

human_results_array = np.zeros(66*20)
lpips_results_array = np.zeros(66*20)
i=0
for gt_file_name in gt_files:

    re_col_files = df[df['GTFileName']==gt_file_name]['ReColFileName'].unique()

    for re_col_file_name in re_col_files:
        human_results_array[i] = human_results[gt_file_name][re_col_file_name]
        lpips_results_array[i] = lpips_results[gt_file_name][re_col_file_name]
        i+=1

row.append(['All', np.round(stats.spearmanr(human_results_array, lpips_results_array)[0], 3), np.round(stats.spearmanr(human_results_array, lpips_results_array)[1],3)])


table = pd.DataFrame(row,columns=['GT FileName', f'lpips-{network}-r', f'lpips-{network}-p'])

table.to_csv(f"SpearmanCorrelationLpips{network}.csv")


#######################  Kendal  #######################################

human_results_array = np.zeros(66)
lpips_results_array = np.zeros(66)
row = []
for gt_file_name in gt_files:
    i = 0
    re_col_files = df[df['GTFileName']==gt_file_name]['ReColFileName'].unique()

    for re_col_file_name in re_col_files:
        human_results_array[i] = human_results[gt_file_name][re_col_file_name]
        lpips_results_array[i] = lpips_results[gt_file_name][re_col_file_name]
        i+=1
    row.append([gt_file_name, np.round(stats.kendalltau(human_results_array, lpips_results_array)[0], 3), np.round(stats.kendalltau(human_results_array, lpips_results_array)[1],3)])

human_results_array = np.zeros(66*20)
lpips_results_array = np.zeros(66*20)
i=0
for gt_file_name in gt_files:

    re_col_files = df[df['GTFileName']==gt_file_name]['ReColFileName'].unique()

    for re_col_file_name in re_col_files:
        human_results_array[i] = human_results[gt_file_name][re_col_file_name]
        lpips_results_array[i] = lpips_results[gt_file_name][re_col_file_name]
        i+=1

row.append(['All', np.round(stats.kendalltau(human_results_array, lpips_results_array)[0], 3), np.round(stats.kendalltau(human_results_array, lpips_results_array)[1],3)])


table = pd.DataFrame(row,columns=['GT FileName', 'lpips-vgg-r', 'lpips-vgg-p'])

table.to_csv(f"kendallCorrelationLpips{network}.csv")

