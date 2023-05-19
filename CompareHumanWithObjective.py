import cv2
import pandas as pd
import numpy as np
from scipy import stats
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error as mae
#from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
import math
from sewar import msssim, rmse, mse
from scipy import ndimage
from scipy.spatial import distance


def r2p(x):
    return np.abs(x), np.angle(x)

def colourfulness_compare(gt_file_name, re_col_file_name):
    gt_file = cv2.imread(f'./HECDImages/{gt_file_name}')
    gt_file = cv2.cvtColor(gt_file, cv2.COLOR_BGR2LAB)
    g_std_a =  np.std(gt_file[1])
    g_std_b = np.std(gt_file[2])
    g_std_ab = math.sqrt(g_std_a*g_std_a+g_std_b*g_std_b)

    g_mean_a = np.mean(gt_file[1])
    g_mean_b = np.mean(gt_file[2])
    g_mean_ab = math.sqrt(g_mean_a*g_mean_a+g_mean_b*g_mean_b)
    g_colf = g_std_ab+0.37*g_mean_ab

    re_col_file = cv2.imread(f'./HECDImages/{re_col_file_name}')
    re_col_file = cv2.cvtColor(re_col_file, cv2.COLOR_BGR2LAB)
    r_std_a =  np.std(re_col_file[1])
    r_std_b = np.std(re_col_file[2])
    r_std_ab = math.sqrt(r_std_a*r_std_a+r_std_b*r_std_b)

    r_mean_a = np.mean(re_col_file[1])
    r_mean_b = np.mean(re_col_file[2])
    r_mean_ab = math.sqrt(r_mean_a*r_mean_a+r_mean_b*r_mean_b)
    r_colf = r_std_ab+0.37*r_mean_ab

    return g_colf-r_colf

def colourfulness(re_col_file_name):

    re_col_file = cv2.imread(f'./HECDImages/{re_col_file_name}')
    re_col_file = cv2.cvtColor(re_col_file, cv2.COLOR_BGR2LAB)
    std_a =  np.std(re_col_file[1])
    std_b = np.std(re_col_file[2])
    std_ab = math.sqrt(std_a*std_a+std_b*std_b)

    mean_a = np.mean(re_col_file[1])
    mean_b = np.mean(re_col_file[2])
    mean_ab = math.sqrt(mean_a*mean_a+mean_b*mean_b)
    return std_ab+0.37*mean_ab

def LAB2LHC(image):
    new_image = np.copy(image)
    comp = np.array(image[:, :, 1:3], np.float64)  # change ab to floating point
    comp = (comp[:, :, 0]-128) + (1j * (comp[:, :, 1]-128))  # convert to complex number (cartesian) format
    c, h = r2p(comp)  # convert to polar form c is magnitude, h is hue
    h = np.array(h/math.pi*128+128, np.dtype(np.uint8))  # change the range of h from (-pi, +pi) to (0,255)
    c = np.array(c, np.uint8)     # just change to uint8
    new_image[:, :, 1] = h       # merge the three channels
    new_image[:,:,2] = c

    return np.copy(new_image)




def ssim_compare_hc(gt_file_name, re_col_file_name):
    gt_file = cv2.imread(f'./HECDImages/{gt_file_name}')
    gt_file = cv2.cvtColor(gt_file, cv2.COLOR_BGR2LAB)
    gt_file1 = LAB2LHC(gt_file)
    re_col_file = cv2.imread(f'./HECDImages/{re_col_file_name}')
    re_col_file = cv2.cvtColor(re_col_file, cv2.COLOR_BGR2LAB)
    re_col_file1 =LAB2LHC(re_col_file)
    return ssim(gt_file1[:,:,1], re_col_file1[:,:,1])*ssim(gt_file1[:,:,2], re_col_file1[:,:,2])

def ssim_compare_ab(gt_file_name, re_col_file_name):
    gt_file = cv2.imread(f'./HECDImages/{gt_file_name}')
    gt_file = cv2.cvtColor(gt_file, cv2.COLOR_BGR2LAB)

    re_col_file = cv2.imread(f'./HECDImages/{re_col_file_name}')
    re_col_file = cv2.cvtColor(re_col_file, cv2.COLOR_BGR2LAB)
    return ssim(gt_file[:,:,1], re_col_file[:,:,1])*ssim(gt_file[:,:,2], re_col_file[:,:,2])

def ssim_compare_bgr(gt_file_name, re_col_file_name):
    gt_file = cv2.imread(f'./HECDImages/{gt_file_name}')
    re_col_file = cv2.imread(f'./HECDImages/{re_col_file_name}')
    return ssim(gt_file[:,:,0], re_col_file[:,:,0])*ssim(gt_file[:,:,1], re_col_file[:,:,1])*ssim(gt_file[:,:,2], re_col_file[:,:,2])

def msssim_compare_hc(gt_file_name, re_col_file_name):
    gt_file = cv2.imread(f'./HECDImages/{gt_file_name}')
    gt_file = cv2.cvtColor(gt_file, cv2.COLOR_BGR2LAB)
    gt_file1 = LAB2LHC(gt_file)

    re_col_file = cv2.imread(f'./HECDImages/{re_col_file_name}')
    re_col_file = cv2.cvtColor(re_col_file, cv2.COLOR_BGR2LAB)
    re_col_file1 =LAB2LHC(re_col_file)
    return abs(msssim(gt_file1[:,:,1], re_col_file1[:,:,1])*msssim(gt_file1[:,:,2], re_col_file1[:,:,2]))

def msssim_compare_ab(gt_file_name, re_col_file_name):
    gt_file = cv2.imread(f'./HECDImages/{gt_file_name}')
    gt_file = cv2.cvtColor(gt_file, cv2.COLOR_BGR2LAB)
    re_col_file = cv2.imread(f'./HECDImages/{re_col_file_name}')
    re_col_file = cv2.cvtColor(re_col_file, cv2.COLOR_BGR2LAB)
    return abs(msssim(gt_file[:,:,1], re_col_file[:,:,1])*msssim(gt_file[:,:,2], re_col_file[:,:,2]))

def mse_compare_hc(gt_file_name, re_col_file_name):
    gt_file = cv2.imread(f'./HECDImages/{gt_file_name}')
    gt_file = cv2.cvtColor(gt_file, cv2.COLOR_BGR2LAB)

    re_col_file = cv2.imread(f'./HECDImages/{re_col_file_name}')
    re_col_file = cv2.cvtColor(re_col_file, cv2.COLOR_BGR2LAB)
    re_col_file =LAB2LHC(re_col_file)
    return mse(gt_file[:,:,1], re_col_file[:,:,1])+mse(gt_file[:,:,2], re_col_file[:,:,2])

def msssim_compare_bgr(gt_file_name, re_col_file_name):
    gt_file = cv2.imread(f'./HECDImages/{gt_file_name}')
    re_col_file = cv2.imread(f'./HECDImages/{re_col_file_name}')
    return abs(msssim(gt_file[:,:,0], re_col_file[:,:,0])*msssim(gt_file[:,:,1], re_col_file[:,:,1])*msssim(gt_file[:,:,2], re_col_file[:,:,2]))

def mse_compare_ab(gt_file_name, re_col_file_name):
    gt_file = cv2.imread(f'./HECDImages/{gt_file_name}')
    gt_file = cv2.cvtColor(gt_file, cv2.COLOR_BGR2LAB)

    re_col_file = cv2.imread(f'./HECDImages/{re_col_file_name}')
    re_col_file = cv2.cvtColor(re_col_file, cv2.COLOR_BGR2LAB)
    return mse(gt_file[:,:,1], re_col_file[:,:,1])+mse(gt_file[:,:,2], re_col_file[:,:,2])

def rmse_compare_hc(gt_file_name, re_col_file_name):
    gt_file = cv2.imread(f'./HECDImages/{gt_file_name}')
    gt_file = cv2.cvtColor(gt_file, cv2.COLOR_BGR2LAB)

    re_col_file = cv2.imread(f'./HECDImages/{re_col_file_name}')
    re_col_file = cv2.cvtColor(re_col_file, cv2.COLOR_BGR2LAB)
    re_col_file =LAB2LHC(re_col_file)
    return rmse(gt_file[:,:,1], re_col_file[:,:,1])+rmse(gt_file[:,:,2], re_col_file[:,:,2])

def rmse_compare_ab(gt_file_name, re_col_file_name):
    gt_file = cv2.imread(f'./HECDImages/{gt_file_name}')
    gt_file = cv2.cvtColor(gt_file, cv2.COLOR_BGR2LAB)

    re_col_file = cv2.imread(f'./HECDImages/{re_col_file_name}')
    re_col_file = cv2.cvtColor(re_col_file, cv2.COLOR_BGR2LAB)
    return rmse(gt_file[:,:,1], re_col_file[:,:,1])+rmse(gt_file[:,:,2], re_col_file[:,:,2])

def mae_compare_hc(gt_file_name, re_col_file_name):
    gt_file = cv2.imread(f'./HECDImages/{gt_file_name}')
    gt_file = cv2.cvtColor(gt_file, cv2.COLOR_BGR2LAB)

    re_col_file = cv2.imread(f'./HECDImages/{re_col_file_name}')
    re_col_file = cv2.cvtColor(re_col_file, cv2.COLOR_BGR2LAB)
    re_col_file =LAB2LHC(re_col_file)
    return mae(gt_file[:,:,1], re_col_file[:,:,1])+mae(gt_file[:,:,2], re_col_file[:,:,2])

def mae_compare_ab(gt_file_name, re_col_file_name):
    gt_file = cv2.imread(f'./HECDImages/{gt_file_name}')
    gt_file = cv2.cvtColor(gt_file, cv2.COLOR_BGR2LAB)

    re_col_file = cv2.imread(f'./HECDImages/{re_col_file_name}')
    re_col_file = cv2.cvtColor(re_col_file, cv2.COLOR_BGR2LAB)
    return mae(gt_file[:,:,1], re_col_file[:,:,1])+mae(gt_file[:,:,2], re_col_file[:,:,2])

def psnr_ab(gt_file_name, re_col_file_name):
    gt_file = cv2.imread(f'./HECDImages/{gt_file_name}')
    gt_file = cv2.cvtColor(gt_file, cv2.COLOR_BGR2LAB)
    gt_file = gt_file[:,:,1:2]
    re_col_file = cv2.imread(f'./HECDImages/{re_col_file_name}')
    re_col_file = cv2.cvtColor(re_col_file, cv2.COLOR_BGR2LAB)
    re_col_file = re_col_file[:,:,1:2]

    return psnr(gt_file, re_col_file)


def psnr_hc(gt_file_name, re_col_file_name):
    gt_file = cv2.imread(f'./HECDImages/{gt_file_name}')
    gt_file = cv2.cvtColor(gt_file, cv2.COLOR_BGR2LAB)
    gt_file = LAB2LHC(gt_file)
    gt_file = gt_file[:,:,1:2]

    re_col_file = cv2.imread(f'./HECDImages/{re_col_file_name}')
    re_col_file = cv2.cvtColor(re_col_file, cv2.COLOR_BGR2LAB)
    re_col_file = LAB2LHC(re_col_file)
    re_col_file = re_col_file[:,:,1:2]


    return psnr(gt_file, re_col_file)

df = pd.read_csv('./bokeh_app/HumanAggregatedResults.csv')

#GTFileName	ReColFileName	Mod Number	WB Corrected	ReColorMod	zScore	GT_zScore
gt_files = df['GTFileName'].unique()
human_results = {}
ssim_results_ab = {}
msssim_results_ab = {}
ssim_results_hc = {}
msssim_results_hc = {}
ssim_results_bgr = {}
msssim_results_bgr = {}

mse_results_ab = {}
mse_results_hc = {}

rmse_results_ab = {}
rmse_results_hc = {}
mae_results_ab = {}
mae_results_hc = {}


colourfulness_results = {}
colourfulness_dif_results = {}
psnr_ab_results = {}
psnr_hc_results = {}
mah_results_ab = {}
for gt_file_name in gt_files:
    print(gt_file_name)
    re_col_files = df[df['GTFileName']==gt_file_name]['ReColFileName'].unique()
    re_col_dict = {}
    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= df[df['ReColFileName']==re_col_file_name]['zScore'].iloc[0]
    human_results[gt_file_name] = re_col_dict
    print('SSIM-ab*******************************************************')
    re_col_dict = {}
    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= ssim_compare_ab(gt_file_name, re_col_file_name)
        #print(f'{re_col_file_name} = {re_col_dict[re_col_file_name]}')
    ssim_results_ab[gt_file_name] = re_col_dict
    print('SSIM-hc********************************************************')
    re_col_dict = {}
    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= ssim_compare_hc(gt_file_name, re_col_file_name)
        #print(f'{re_col_file_name} = {re_col_dict[re_col_file_name]}')
    ssim_results_hc[gt_file_name] = re_col_dict

    print('SSIM-bgr********************************************************')
    re_col_dict = {}
    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= ssim_compare_bgr(gt_file_name, re_col_file_name)
        #print(f'{re_col_file_name} = {re_col_dict[re_col_file_name]}')
    ssim_results_bgr[gt_file_name] = re_col_dict


    print('MS-SSIM-ab')
    re_col_dict = {}
    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= msssim_compare_ab(gt_file_name, re_col_file_name)
        #print(f'{re_col_file_name} = {re_col_dict[re_col_file_name]}')
    msssim_results_ab[gt_file_name] = re_col_dict
    print('MS-SSIM-hc')
    re_col_dict = {}
    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= msssim_compare_hc(gt_file_name, re_col_file_name)
        #print(f'{re_col_file_name} = {re_col_dict[re_col_file_name]}')
    msssim_results_hc[gt_file_name] = re_col_dict

    print('MS-SSIM-bgr')
    re_col_dict = {}
    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= msssim_compare_bgr(gt_file_name, re_col_file_name)
        #print(f'{re_col_file_name} = {re_col_dict[re_col_file_name]}')
    msssim_results_bgr[gt_file_name] = re_col_dict
    print('MSE-ab')
    re_col_dict = {}
    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= mse_compare_ab(gt_file_name, re_col_file_name)
    mse_results_ab[gt_file_name] = re_col_dict
    print('MSE-hc')
    re_col_dict = {}
    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= mse_compare_hc(gt_file_name, re_col_file_name)
    mse_results_hc[gt_file_name] = re_col_dict
    print('RMSE-ab')
    re_col_dict = {}
    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= rmse_compare_ab(gt_file_name, re_col_file_name)
    rmse_results_ab[gt_file_name] = re_col_dict
    print('RMSE-hc')
    re_col_dict = {}
    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= rmse_compare_hc(gt_file_name, re_col_file_name)
    rmse_results_hc[gt_file_name] = re_col_dict
    print('MAE-ab')
    re_col_dict = {}
    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= mae_compare_ab(gt_file_name, re_col_file_name)
    mae_results_ab[gt_file_name] = re_col_dict
    print('MAE-hc')
    re_col_dict = {}
    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= mae_compare_hc(gt_file_name, re_col_file_name)
    mae_results_hc[gt_file_name] = re_col_dict

    print('Colourfullness')
    re_col_dict = {}
    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= colourfulness(re_col_file_name)
    colourfulness_results[gt_file_name] = re_col_dict
    print('Colourfullness Difference')
    re_col_dict = {}
    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= colourfulness_compare(gt_file_name,re_col_file_name)
    colourfulness_dif_results[gt_file_name] = re_col_dict
    print('PSNR-ab')
    re_col_dict = {}
    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= psnr_ab(gt_file_name,re_col_file_name)
    psnr_ab_results[gt_file_name] = re_col_dict

    print('PSNR-hc')
    re_col_dict = {}
    for re_col_file_name in re_col_files:
        re_col_dict[re_col_file_name]= psnr_hc(gt_file_name,re_col_file_name)
    psnr_hc_results[gt_file_name] = re_col_dict

human_results_array = np.zeros(66)
ssim_results_ab_array = np.zeros(66)
ssim_results_hc_array = np.zeros(66)
ssim_results_bgr_array = np.zeros(66)
msssim_results_ab_array = np.zeros(66)
msssim_results_hc_array = np.zeros(66)
msssim_results_bgr_array = np.zeros(66)
mse_results_ab_array = np.zeros(66)
mse_results_hc_array = np.zeros(66)
rmse_results_ab_array = np.zeros(66)
rmse_results_hc_array = np.zeros(66)
mae_results_ab_array = np.zeros(66)
mae_results_hc_array = np.zeros(66)
colourfulness_results_array = np.zeros(66)
colourfulness_dif_results_array = np.zeros(66)
psnr_results_ab_array = np.zeros(66)
psnr_results_hc_array = np.zeros(66)
mah_results_ab_array = np.zeros(66)
row = []
for gt_file_name in gt_files:
    i = 0
    re_col_files = df[df['GTFileName']==gt_file_name]['ReColFileName'].unique()

    for re_col_file_name in re_col_files:
        human_results_array[i] = human_results[gt_file_name][re_col_file_name]
        ssim_results_ab_array[i] = ssim_results_ab[gt_file_name][re_col_file_name]
        ssim_results_hc_array[i] = ssim_results_hc[gt_file_name][re_col_file_name]
        ssim_results_bgr_array[i] = ssim_results_hc[gt_file_name][re_col_file_name]
        msssim_results_ab_array[i] = msssim_results_ab[gt_file_name][re_col_file_name]
        msssim_results_hc_array[i] = msssim_results_hc[gt_file_name][re_col_file_name]
        msssim_results_bgr_array[i] = msssim_results_bgr[gt_file_name][re_col_file_name]
        mse_results_ab_array[i] = mse_results_ab[gt_file_name][re_col_file_name]
        mse_results_hc_array[i] = mse_results_hc[gt_file_name][re_col_file_name]
        rmse_results_ab_array[i] = rmse_results_ab[gt_file_name][re_col_file_name]
        rmse_results_hc_array[i] = rmse_results_hc[gt_file_name][re_col_file_name]
        mae_results_ab_array[i] = mae_results_ab[gt_file_name][re_col_file_name]
        mae_results_hc_array[i] = mae_results_hc[gt_file_name][re_col_file_name]
        colourfulness_results_array[i] = colourfulness_results[gt_file_name][re_col_file_name]
        colourfulness_dif_results_array[i] = colourfulness_dif_results[gt_file_name][re_col_file_name]
        psnr_results_ab_array[i] = psnr_ab_results[gt_file_name][re_col_file_name]
        psnr_results_hc_array[i] = psnr_hc_results[gt_file_name][re_col_file_name]
        i+=1
    row.append([gt_file_name, np.round(stats.kendalltau(human_results_array, ssim_results_ab_array)[0], 3), np.round(stats.kendalltau(human_results_array, ssim_results_ab_array)[1],3)
                , np.round(stats.kendalltau(human_results_array, ssim_results_hc_array)[0], 3), np.round(stats.kendalltau(human_results_array, ssim_results_hc_array)[1],3),
                np.round(stats.kendalltau(human_results_array, ssim_results_bgr_array)[0], 3), np.round(stats.kendalltau(human_results_array, ssim_results_bgr_array)[1],3),
                np.round(stats.kendalltau(human_results_array, msssim_results_ab_array)[0], 3), np.round(stats.kendalltau(human_results_array, msssim_results_ab_array)[1],3)
                , np.round(stats.kendalltau(human_results_array, msssim_results_hc_array)[0], 3), np.round(stats.kendalltau(human_results_array, msssim_results_hc_array)[1],3),
                 np.round(stats.kendalltau(human_results_array, msssim_results_bgr_array)[0], 3), np.round(stats.kendalltau(human_results_array, msssim_results_bgr_array)[1],3),
                 np.round(stats.kendalltau(human_results_array, mse_results_ab_array)[0], 3), np.round(stats.kendalltau(human_results_array, mse_results_ab_array)[1],3),
                np.round(stats.kendalltau(human_results_array, mse_results_hc_array)[0],3), np.round(stats.kendalltau(human_results_array, mse_results_hc_array)[1],3) ,
                np.round(stats.kendalltau(human_results_array, rmse_results_ab_array)[0], 3), np.round(stats.kendalltau(human_results_array, rmse_results_ab_array)[1],3),
                np.round(stats.kendalltau(human_results_array, rmse_results_hc_array)[0],3), np.round(stats.kendalltau(human_results_array, rmse_results_hc_array)[1],3) ,
                np.round(stats.kendalltau(human_results_array, mae_results_ab_array)[0], 3), np.round(stats.kendalltau(human_results_array, mae_results_ab_array)[1],3),
                np.round(stats.kendalltau(human_results_array, mae_results_hc_array)[0],3), np.round(stats.kendalltau(human_results_array, mae_results_hc_array)[1],3) ,
                np.round(stats.kendalltau(human_results_array, colourfulness_results_array)[0],3), np.round(stats.kendalltau(human_results_array, colourfulness_results_array)[1],3),
                np.round(stats.kendalltau(human_results_array, colourfulness_dif_results_array)[0],3), np.round(stats.kendalltau(human_results_array, colourfulness_dif_results_array)[1],3),
                np.round(stats.kendalltau(human_results_array, psnr_results_ab_array)[0],3), np.round(stats.kendalltau(human_results_array, psnr_results_ab_array)[1],3),
                np.round(stats.kendalltau(human_results_array, psnr_results_hc_array)[0],3), np.round(stats.kendalltau(human_results_array, psnr_results_hc_array)[1],3)]
               )



human_results_array = np.zeros(66*20)
ssim_results_ab_array = np.zeros(66*20)
ssim_results_hc_array = np.zeros(66*20)
ssim_results_bgr_array = np.zeros(66*20)
msssim_results_ab_array = np.zeros(66*20)
msssim_results_hc_array = np.zeros(66*20)
msssim_results_bgr_array = np.zeros(66*20)
mse_results_ab_array = np.zeros(66*20)
mse_results_hc_array = np.zeros(66*20)
rmse_results_ab_array = np.zeros(66*20)
rmse_results_hc_array = np.zeros(66*20)
mae_results_ab_array = np.zeros(66*20)
mae_results_hc_array = np.zeros(66*20)

colourfulness_results_array = np.zeros(66*20)
colourfulness_dif_results_array = np.zeros(66*20)
psnr_results_ab_array = np.zeros(66*20)
psnr_results_hc_array = np.zeros(66*20)

i=0
for gt_file_name in gt_files:

    re_col_files = df[df['GTFileName']==gt_file_name]['ReColFileName'].unique()

    for re_col_file_name in re_col_files:
        human_results_array[i] = human_results[gt_file_name][re_col_file_name]
        ssim_results_ab_array[i] = ssim_results_ab[gt_file_name][re_col_file_name]
        ssim_results_hc_array[i] = ssim_results_hc[gt_file_name][re_col_file_name]
        ssim_results_bgr_array[i] = ssim_results_bgr[gt_file_name][re_col_file_name]
        msssim_results_ab_array[i] = msssim_results_ab[gt_file_name][re_col_file_name]
        msssim_results_hc_array[i] = msssim_results_hc[gt_file_name][re_col_file_name]
        msssim_results_bgr_array[i] = msssim_results_bgr[gt_file_name][re_col_file_name]
        mse_results_ab_array[i] = mse_results_ab[gt_file_name][re_col_file_name]
        mse_results_hc_array[i] = mse_results_hc[gt_file_name][re_col_file_name]
        rmse_results_ab_array[i] = rmse_results_ab[gt_file_name][re_col_file_name]
        rmse_results_hc_array[i] = rmse_results_hc[gt_file_name][re_col_file_name]
        mae_results_ab_array[i] = mae_results_ab[gt_file_name][re_col_file_name]
        mae_results_hc_array[i] = mae_results_hc[gt_file_name][re_col_file_name]
        colourfulness_results_array[i] = colourfulness_results[gt_file_name][re_col_file_name]
        colourfulness_dif_results_array[i] = colourfulness_dif_results[gt_file_name][re_col_file_name]
        psnr_results_ab_array[i] = psnr_ab_results[gt_file_name][re_col_file_name]
        psnr_results_hc_array[i] = psnr_hc_results[gt_file_name][re_col_file_name]
        i+=1

row.append(['All', np.round(stats.kendalltau(human_results_array, ssim_results_ab_array)[0], 3), np.round(stats.kendalltau(human_results_array, ssim_results_ab_array)[1],3)
                , np.round(stats.kendalltau(human_results_array, ssim_results_hc_array)[0], 3), np.round(stats.kendalltau(human_results_array, ssim_results_hc_array)[1],3),
            np.round(stats.kendalltau(human_results_array, ssim_results_bgr_array)[0], 3), np.round(stats.kendalltau(human_results_array, ssim_results_bgr_array)[1],3),
            np.round(stats.kendalltau(human_results_array, msssim_results_ab_array)[0], 3), np.round(stats.kendalltau(human_results_array, msssim_results_ab_array)[1],3)
                , np.round(stats.kendalltau(human_results_array, msssim_results_hc_array)[0], 3), np.round(stats.kendalltau(human_results_array, msssim_results_hc_array)[1],3),
            np.round(stats.kendalltau(human_results_array, msssim_results_bgr_array)[0], 3), np.round(stats.kendalltau(human_results_array, msssim_results_bgr_array)[1],3),
                 np.round(stats.kendalltau(human_results_array, mse_results_ab_array)[0], 3), np.round(stats.kendalltau(human_results_array, mse_results_ab_array)[1],3),
                np.round(stats.kendalltau(human_results_array, mse_results_hc_array)[0],3), np.round(stats.kendalltau(human_results_array, mse_results_hc_array)[1],3) ,
            np.round(stats.kendalltau(human_results_array, rmse_results_ab_array)[0], 3), np.round(stats.kendalltau(human_results_array, rmse_results_ab_array)[1],3),
                np.round(stats.kendalltau(human_results_array, rmse_results_hc_array)[0],3), np.round(stats.kendalltau(human_results_array, rmse_results_hc_array)[1],3) ,
            np.round(stats.kendalltau(human_results_array, mae_results_ab_array)[0], 3), np.round(stats.kendalltau(human_results_array, mae_results_ab_array)[1],3),
                np.round(stats.kendalltau(human_results_array, mae_results_hc_array)[0],3), np.round(stats.kendalltau(human_results_array, mae_results_hc_array)[1],3) ,
            np.round(stats.kendalltau(human_results_array, colourfulness_results_array)[0],3), np.round(stats.kendalltau(human_results_array, colourfulness_results_array)[1],3),
            np.round(stats.kendalltau(human_results_array, colourfulness_dif_results_array)[0],3), np.round(stats.kendalltau(human_results_array, colourfulness_dif_results_array)[1],3),
            np.round(stats.kendalltau(human_results_array, psnr_results_ab_array)[0],3), np.round(stats.kendalltau(human_results_array, psnr_results_ab_array)[1],3),
            np.round(stats.kendalltau(human_results_array, psnr_results_hc_array)[0],3), np.round(stats.kendalltau(human_results_array, psnr_results_hc_array)[1],3)]
               )

table = pd.DataFrame(row,columns=['GT FileName', 'SSIM-tau (a*b*)', 'SSIM-p (a*b*)', 'SSIM-tau (hc)', 'SSIM-p (hc)','SSIM-tau (bgr)', 'SSIM-p (bgr)',
                                  'MS-SSIM-tau (a*b*)', 'MS-SSIM-p (a*b*)', 'MS-SSIM-tau (hc)', 'MS-SSIM-p (hc)','MS-SSIM-tau (bgr)', 'MS-SSIM-p (bgr)',
                                  'MSE-tau (a*b*)', 'MSE-p (a*b*)', 'MSE-tau (hc)', 'MSE-p (hc)',
                                  'RMSE-tau (a*b*)', 'RMSE-p (a*b*)', 'RMSE-tau (hc)', 'RMSE-p (hc)',
                                  'MAE-tau (a*b*)', 'MAE-p (a*b*)', 'MAE-tau (hc)', 'MAE-p (hc)',
                                  'Colourfulness-tau', 'Colourfulness-p', 'Colourfulness-dif-tau',
                                  'Colourfulness-dif-p', 'psnr-ab-tau', 'psnr-ab-p', 'psnr-hc-tau', 'psnr-hc-p'])

table.to_csv("KendallCorrellation.csv")

######################################################################################################################
human_results_array = np.zeros(66)
ssim_results_ab_array = np.zeros(66)
ssim_results_hc_array = np.zeros(66)
ssim_results_bgr_array = np.zeros(66)
msssim_results_ab_array = np.zeros(66)
msssim_results_hc_array = np.zeros(66)
msssim_results_bgr_array = np.zeros(66)
mse_results_ab_array = np.zeros(66)
mse_results_hc_array = np.zeros(66)
rmse_results_ab_array = np.zeros(66)
rmse_results_hc_array = np.zeros(66)
mae_results_ab_array = np.zeros(66)
mae_results_hc_array = np.zeros(66)
colourfulness_results_array = np.zeros(66)
colourfulness_dif_results_array = np.zeros(66)
psnr_results_ab_array = np.zeros(66)
psnr_results_hc_array = np.zeros(66)
mah_results_ab_array = np.zeros(66)
row = []
for gt_file_name in gt_files:
    i = 0
    re_col_files = df[df['GTFileName']==gt_file_name]['ReColFileName'].unique()

    for re_col_file_name in re_col_files:
        human_results_array[i] = human_results[gt_file_name][re_col_file_name]
        #print(f'{re_col_file_name}:{human_results_array[i]}')
        ssim_results_ab_array[i] = ssim_results_ab[gt_file_name][re_col_file_name]
        ssim_results_hc_array[i] = ssim_results_hc[gt_file_name][re_col_file_name]
        ssim_results_bgr_array[i] = ssim_results_hc[gt_file_name][re_col_file_name]
        msssim_results_ab_array[i] = msssim_results_ab[gt_file_name][re_col_file_name]
        msssim_results_hc_array[i] = msssim_results_hc[gt_file_name][re_col_file_name]
        msssim_results_bgr_array[i] = msssim_results_bgr[gt_file_name][re_col_file_name]
        mse_results_ab_array[i] = mse_results_ab[gt_file_name][re_col_file_name]
        mse_results_hc_array[i] = mse_results_hc[gt_file_name][re_col_file_name]
        rmse_results_ab_array[i] = rmse_results_ab[gt_file_name][re_col_file_name]
        rmse_results_hc_array[i] = rmse_results_hc[gt_file_name][re_col_file_name]
        mae_results_ab_array[i] = mae_results_ab[gt_file_name][re_col_file_name]
        mae_results_hc_array[i] = mae_results_hc[gt_file_name][re_col_file_name]
        colourfulness_results_array[i] = colourfulness_results[gt_file_name][re_col_file_name]
        colourfulness_dif_results_array[i] = colourfulness_dif_results[gt_file_name][re_col_file_name]
        psnr_results_ab_array[i] = psnr_ab_results[gt_file_name][re_col_file_name]
        psnr_results_hc_array[i] = psnr_hc_results[gt_file_name][re_col_file_name]
        i+=1
    row.append([gt_file_name, np.round(stats.spearmanr(human_results_array, ssim_results_ab_array)[0], 3), np.round(stats.spearmanr(human_results_array, ssim_results_ab_array)[1],3)
                , np.round(stats.spearmanr(human_results_array, ssim_results_hc_array)[0], 3), np.round(stats.spearmanr(human_results_array, ssim_results_hc_array)[1],3),
                np.round(stats.spearmanr(human_results_array, ssim_results_bgr_array)[0], 3), np.round(stats.spearmanr(human_results_array, ssim_results_bgr_array)[1],3),
                np.round(stats.spearmanr(human_results_array, msssim_results_ab_array)[0], 3), np.round(stats.spearmanr(human_results_array, msssim_results_ab_array)[1],3)
                , np.round(stats.spearmanr(human_results_array, msssim_results_hc_array)[0], 3), np.round(stats.spearmanr(human_results_array, msssim_results_hc_array)[1],3),
                 np.round(stats.spearmanr(human_results_array, msssim_results_bgr_array)[0], 3), np.round(stats.spearmanr(human_results_array, msssim_results_bgr_array)[1],3),
                 np.round(stats.spearmanr(human_results_array, mse_results_ab_array)[0], 3), np.round(stats.spearmanr(human_results_array, mse_results_ab_array)[1],3),
                np.round(stats.spearmanr(human_results_array, mse_results_hc_array)[0],3), np.round(stats.spearmanr(human_results_array, mse_results_hc_array)[1],3) ,
                np.round(stats.spearmanr(human_results_array, rmse_results_ab_array)[0], 3), np.round(stats.spearmanr(human_results_array, rmse_results_ab_array)[1],3),
                np.round(stats.spearmanr(human_results_array, rmse_results_hc_array)[0],3), np.round(stats.spearmanr(human_results_array, rmse_results_hc_array)[1],3) ,
                np.round(stats.spearmanr(human_results_array, mae_results_ab_array)[0], 3), np.round(stats.spearmanr(human_results_array, mae_results_ab_array)[1],3),
                np.round(stats.spearmanr(human_results_array, mae_results_hc_array)[0],3), np.round(stats.spearmanr(human_results_array, mae_results_hc_array)[1],3) ,
                np.round(stats.spearmanr(human_results_array, colourfulness_results_array)[0],3), np.round(stats.spearmanr(human_results_array, colourfulness_results_array)[1],3),
                np.round(stats.spearmanr(human_results_array, colourfulness_dif_results_array)[0],3), np.round(stats.spearmanr(human_results_array, colourfulness_dif_results_array)[1],3),
                np.round(stats.spearmanr(human_results_array, psnr_results_ab_array)[0],3), np.round(stats.spearmanr(human_results_array, psnr_results_ab_array)[1],3),
                np.round(stats.spearmanr(human_results_array, psnr_results_hc_array)[0],3), np.round(stats.spearmanr(human_results_array, psnr_results_hc_array)[1],3)]
               )



human_results_array = np.zeros(66*20)
ssim_results_ab_array = np.zeros(66*20)
ssim_results_hc_array = np.zeros(66*20)
ssim_results_bgr_array = np.zeros(66*20)
msssim_results_ab_array = np.zeros(66*20)
msssim_results_hc_array = np.zeros(66*20)
msssim_results_bgr_array = np.zeros(66*20)
mse_results_ab_array = np.zeros(66*20)
mse_results_hc_array = np.zeros(66*20)
rmse_results_ab_array = np.zeros(66*20)
rmse_results_hc_array = np.zeros(66*20)
mae_results_ab_array = np.zeros(66*20)
mae_results_hc_array = np.zeros(66*20)

colourfulness_results_array = np.zeros(66*20)
colourfulness_dif_results_array = np.zeros(66*20)
psnr_results_ab_array = np.zeros(66*20)
psnr_results_hc_array = np.zeros(66*20)

i=0
for gt_file_name in gt_files:

    re_col_files = df[df['GTFileName']==gt_file_name]['ReColFileName'].unique()

    for re_col_file_name in re_col_files:
        human_results_array[i] = human_results[gt_file_name][re_col_file_name]
        ssim_results_ab_array[i] = ssim_results_ab[gt_file_name][re_col_file_name]
        ssim_results_hc_array[i] = ssim_results_hc[gt_file_name][re_col_file_name]
        ssim_results_bgr_array[i] = ssim_results_bgr[gt_file_name][re_col_file_name]
        msssim_results_ab_array[i] = msssim_results_ab[gt_file_name][re_col_file_name]
        msssim_results_hc_array[i] = msssim_results_hc[gt_file_name][re_col_file_name]
        msssim_results_bgr_array[i] = msssim_results_bgr[gt_file_name][re_col_file_name]
        mse_results_ab_array[i] = mse_results_ab[gt_file_name][re_col_file_name]
        mse_results_hc_array[i] = mse_results_hc[gt_file_name][re_col_file_name]
        rmse_results_ab_array[i] = rmse_results_ab[gt_file_name][re_col_file_name]
        rmse_results_hc_array[i] = rmse_results_hc[gt_file_name][re_col_file_name]
        mae_results_ab_array[i] = mae_results_ab[gt_file_name][re_col_file_name]
        mae_results_hc_array[i] = mae_results_hc[gt_file_name][re_col_file_name]
        colourfulness_results_array[i] = colourfulness_results[gt_file_name][re_col_file_name]
        colourfulness_dif_results_array[i] = colourfulness_dif_results[gt_file_name][re_col_file_name]
        psnr_results_ab_array[i] = psnr_ab_results[gt_file_name][re_col_file_name]
        psnr_results_hc_array[i] = psnr_hc_results[gt_file_name][re_col_file_name]
        i+=1

row.append(['All', np.round(stats.spearmanr(human_results_array, ssim_results_ab_array)[0], 3), np.round(stats.spearmanr(human_results_array, ssim_results_ab_array)[1],3)
                , np.round(stats.spearmanr(human_results_array, ssim_results_hc_array)[0], 3), np.round(stats.spearmanr(human_results_array, ssim_results_hc_array)[1],3),
            np.round(stats.spearmanr(human_results_array, ssim_results_bgr_array)[0], 3), np.round(stats.spearmanr(human_results_array, ssim_results_bgr_array)[1],3),
            np.round(stats.spearmanr(human_results_array, msssim_results_ab_array)[0], 3), np.round(stats.spearmanr(human_results_array, msssim_results_ab_array)[1],3)
                , np.round(stats.spearmanr(human_results_array, msssim_results_hc_array)[0], 3), np.round(stats.spearmanr(human_results_array, msssim_results_hc_array)[1],3),
            np.round(stats.spearmanr(human_results_array, msssim_results_bgr_array)[0], 3), np.round(stats.spearmanr(human_results_array, msssim_results_bgr_array)[1],3),
                 np.round(stats.spearmanr(human_results_array, mse_results_ab_array)[0], 3), np.round(stats.spearmanr(human_results_array, mse_results_ab_array)[1],3),
                np.round(stats.spearmanr(human_results_array, mse_results_hc_array)[0],3), np.round(stats.spearmanr(human_results_array, mse_results_hc_array)[1],3) ,
            np.round(stats.spearmanr(human_results_array, rmse_results_ab_array)[0], 3), np.round(stats.spearmanr(human_results_array, rmse_results_ab_array)[1],3),
                np.round(stats.spearmanr(human_results_array, rmse_results_hc_array)[0],3), np.round(stats.spearmanr(human_results_array, rmse_results_hc_array)[1],3) ,
            np.round(stats.spearmanr(human_results_array, mae_results_ab_array)[0], 3), np.round(stats.spearmanr(human_results_array, mae_results_ab_array)[1],3),
                np.round(stats.spearmanr(human_results_array, mae_results_hc_array)[0],3), np.round(stats.spearmanr(human_results_array, mae_results_hc_array)[1],3) ,
            np.round(stats.spearmanr(human_results_array, colourfulness_results_array)[0],3), np.round(stats.spearmanr(human_results_array, colourfulness_results_array)[1],3),
            np.round(stats.spearmanr(human_results_array, colourfulness_dif_results_array)[0],3), np.round(stats.spearmanr(human_results_array, colourfulness_dif_results_array)[1],3),
            np.round(stats.spearmanr(human_results_array, psnr_results_ab_array)[0],3), np.round(stats.spearmanr(human_results_array, psnr_results_ab_array)[1],3),
            np.round(stats.spearmanr(human_results_array, psnr_results_hc_array)[0],3), np.round(stats.spearmanr(human_results_array, psnr_results_hc_array)[1],3)]
               )

table = pd.DataFrame(row,columns=['GT FileName', 'SSIM-r (a*b*)', 'SSIM-p (a*b*)', 'SSIM-r (hc)', 'SSIM-p (hc)','SSIM-r (bgr)', 'SSIM-p (bgr)',
                                  'MS-SSIM-r (a*b*)', 'MS-SSIM-p (a*b*)', 'MS-SSIM-r (hc)', 'MS-SSIM-p (hc)','MS-SSIM-r (bgr)', 'MS-SSIM-p (bgr)',
                                  'MSE-r (a*b*)', 'MSE-p (a*b*)', 'MSE-r (hc)', 'MSE-p (hc)',
                                  'RMSE-r (a*b*)', 'RMSE-p (a*b*)', 'RMSE-r (hc)', 'RMSE-p (hc)',
                                  'MAE-r (a*b*)', 'MAE-p (a*b*)', 'MAE-r (hc)', 'MAE-p (hc)',
                                  'Colourfulness-r', 'Colourfulness-p', 'Colourfulness-dif-r',
                                  'Colourfulness-dif-p', 'psnr-ab-r', 'psnr-ab-p', 'psnr-hc-r', 'psnr-hc-p'])

table.to_csv("SpearmanCorrellation.csv")
