''' Various functions for extracting metrics, creating dataframes,
and getting tissue-wise + regional averages from images. '''

import nilearn as nl
from nilearn import image
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import metrics

def get_metric(y_true, y_pred, mask, param='cbf', metric='rmse', scaling=1., tiss_type='GM'):
    ''' y_true: numpy array of ground truth image
        y_pred: numpy array of network prediction/estimate
        mask: str of path to mask file'''
    
    seg_np = image.get_data(mask)
    params = {'cbf':0, 'att':1}
    tissues = {'CSF':1, "GM":2, 'WM':3}
    
    mask_np = np.repeat(seg_np[...,None], 2, axis=-1)

    params = {'cbf':0, 'att':1}
    
    mask_np[...,params[param]] = 0

    true = y_true[0][mask_np==tissues[tiss_type]] * scaling
    pred = y_pred[mask_np==tissues[tiss_type]] * scaling
    
    if metric == 'rmse':
        value = metrics.mean_squared_error(true, pred, squared=False)
    elif metric =='mae':
        value = metrics.mean_absolute_error(true, pred)
    elif metric == 'ssim':
        t = tf.convert_to_tensor(y_true[0][...,params[param]], dtype=tf.float32)
        p = tf.convert_to_tensor(y_pred[...,params[param]], dtype=tf.float32)

        value = tf.image.ssim(t,p,max_val=2.5, filter_size=11, filter_sigma=1.5).numpy()
        
    return value

def get_dframe(ground_truths, predictions, masks, metric = 'mae'):
    '''Get a dataframe of the results given some images. This is very specific to
    the design of this study.'''
    
    cbf_mae = []
    att_mae = []
    for model_num in range(4):
        temp_cbf = []
        temp_att = []
        for rem_num in range(5):
            pld_att_mae = [get_metric(gt, pred, mask, param='att', metric=metric, scaling=1.36, tiss_type='GM')\
                        for gt, pred, mask in zip(ground_truths, predictions[model_num][rem_num], masks)]
            temp_att.append(pld_att_mae)
    
            pld_cbf_mae = [get_metric(gt, pred, mask, param='cbf', metric=metric, scaling=92., tiss_type='GM')\
                        for gt, pred, mask in zip(ground_truths, predictions[model_num][rem_num], masks)]
            temp_cbf.append(pld_cbf_mae)
    
        att_mae.append(temp_att)
        cbf_mae.append(temp_cbf)
    
    ids = [mask.split('/')[-1][4:8] for mask in masks]
    dfs = [pd.DataFrame({'ID': ids*2, 'value':cbf+att, 'metric': [metric,]*(len(cbf)*2), 'param':['CBF',]*len(cbf)+['ATT',]*len(cbf), 'PLDs dropped (training)': [j,]*int(len(cbf)*2), 'Network': ['CNN',]*int(len(cbf)*2), 'PLDs Removed':[i,]*int(len(cbf)*2), 'Field Strength': ([1.5,]*10+[3,]*10)*2})\
       for cbf_model, att_model, j in zip(cbf_mae,att_mae, range(4)) for cbf, att, i in zip(cbf_model, att_model, range(5))]
    df = pd.concat(dfs)
    
    return df

def get_tissue_voxels(gt_org, pred_org, seg, param='cbf', scaling=1., avg=False):
        
    seg_np = image.get_data(seg)
    params = {'cbf':0, 'att':1}
    tissues = {'1': 'CSF', '2': "GM", '3': 'WM'}
        
    gt = gt_org[...,params[param]] * scaling
    pred = pred_org[...,params[param]] * scaling
        
    seg_df = []
    for i in range(1,4):
        if avg:
            gt_temp = gt[seg_np==i].mean()
            pred_temp = pred[seg_np==i].mean()
            diff_temp = get_metric(gt_org[np.newaxis,...], pred_org, seg, param=param, metric='mae', scaling=scaling, tiss_type=tissues[str(i)])

            length = 1
        else:
            gt_temp = gt[seg_np==i]
            pred_temp = pred[seg_np==i]
            diff_temp = diff[seg_np==i]
            length = len(gt_temp)
            
        df = pd.DataFrame({'ID': seg.split('/')[-1][4:8],
                              'gt': gt_temp, 
                              'pred': pred_temp,
                              'Tissue type': [tissues[str(i)],]*length,
                              'param': [params[param],]*length})

        seg_df.append(df)
        
    return pd.concat(seg_df)
    

def get_atlas_values(gt_org, pred_org, seg, param='cbf', scaling=1., avg=False):

    def get_metric(y_true, y_pred, mask, param='cbf', metric='mae', scaling=1., tiss_type='GM'):
        ''' y_true: numpy array of ground truth image
        y_pred: numpy array of network prediction/estimate
        mask: str of path to mask file'''
    
        seg_np = image.get_data(mask)
        params = {'cbf':0, 'att':1}
        tissues = {'Frontal': 3,
               "Parietal": 6,
               'Occipital': 5,
               'Subcortical': 1,
               'Temporal': 8}
    
        mask_np = np.repeat(seg_np[...,None], 2, axis=-1)

        params = {'cbf':0, 'att':1}
    
        mask_np[...,params[param]] = 0

        true = y_true[0][mask_np==tissues[tiss_type]] * scaling
        pred = y_pred[mask_np==tissues[tiss_type]] * scaling
    
        if metric =='mae':
            value = metrics.mean_absolute_error(true, pred)
        elif metric == 'ssim':
            t = tf.convert_to_tensor(y_true[0][...,params[param]], dtype=tf.float32)
            p = tf.convert_to_tensor(y_pred[...,params[param]], dtype=tf.float32)

            value = tf.image.ssim(t,p,max_val=2.5, filter_size=11, filter_sigma=1.5).numpy()
        
        return value

    seg_np = image.get_data(seg)
    seg_np[(seg==1) | (seg==4) | (seg==7) | (seg==9)] = 1              
                   
    params = {'cbf':0, 'att':1}
    tissues = {'3': 'Frontal',
               '6': "Parietal",
               '5': 'Occipital',
               '8': 'Temporal',
               '1': 'Subcortical'}
        
    gt = gt_org[...,params[param]] * scaling
    pred = pred_org[...,params[param]] * scaling
        
    seg_df = []
    for i in tissues.keys():
        if avg:
            gt_temp = gt[seg_np==int(i)].mean()
            pred_temp = pred[seg_np==int(i)].mean()
            diff_temp = get_metric(gt_org[np.newaxis,...], pred_org, seg, param=param, 
                                   metric='mae', scaling=scaling, tiss_type=tissues[str(i)])

            length = 1
        else:
            gt_temp = gt[seg_np==i]
            pred_temp = pred[seg_np==i]
            length = len(gt_temp)
            
        df = pd.DataFrame({'ID': seg.split('/')[-1][4:8],
                              'gt': gt_temp, 
                              'pred': pred_temp,
                              'diff': diff_temp,
                              'Tissue type': [tissues[str(i)],]*length,
                              'param': [params[param],]*length})

        seg_df.append(df)
        
    return pd.concat(seg_df)