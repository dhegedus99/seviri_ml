from glob import glob
import os
os.environ['XRIT_DECOMPRESS_PATH'] = '/home/users/gethomas/bin/xRITDecompress'
import sys
sys.path.append('/home/users/dhegedus/seviri_ml')
import seviri_hrit_proc as sevproc
import prediction_funcs as preds
sys.path.append('/home/users/dhegedus/seviri_ql')
import plotting as sev_plot
import numpy as np
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

import logging, sys
logging.disable(sys.maxsize)
#pixellimit=[1370, 2735, 2000, 3509]


def main(opts):
    outfiles_cs = sevproc.set_output_files(opts, cesium=True)
    outfiles_ql = sevproc.set_output_files(opts, cesium=False)
    
    vis006, vis008, ir016, ir039, ir062, ir073, ir082, ir108, ir120, ir134, satzen, solzen, ds = sevproc.load_seviri_hrit(opts.filename, opts.pixellimit)
    lsm = sevproc.load_lsm(opts.filename_lsm, opts.pixellimit)
    skt = sevproc.load_skt(opts.filename_skt, ds, opts.pixellimit)
    
    print('---CMA---')
    opts.outvar = 'cot'
    opts.outvarunc = 'cma_unc' 
    opts.outvarflag = 'cloudmask'
    cma = preds.predict_cma(vis006, vis008, ir016, ir039, ir062, ir073, 
                        ir082, ir108, ir120, ir134, lsm, skt, 
                        solzen=solzen, satzen=satzen, 
                        undo_true_refl=False, 
                        correct_vis_cal_nasa_to_impf=3)
    res1, res2, res3, res_area, area_ext = sevproc.resample_data(ds, cma, opts)
    print(np.nanmin(np.where(res1!=-999, res1, np.nan)),np.nanmax(res1),np.nanmin(np.where(res2!=-999, res2, np.nan)),np.nanmax(res2), res3)
    cldmask = cma[1]
    
    print('---CPH---')
    opts.outvar = 'cph' 
    opts.outvarunc = 'cph_unc'
    opts.outvarflag = 'phase'
    cph = preds.predict_cph(vis006, vis008, ir016, ir039, ir062, ir073,
                        ir082, ir108, ir120, ir134, lsm, skt,
                        solzen=solzen, satzen=satzen,
                        undo_true_refl=False, correct_vis_cal_nasa_to_impf=3,
                        cldmask=cldmask)
    res1, res2, res3, res_area, area_ext = sevproc.resample_data(ds, cph, opts)
    print(np.nanmin(np.where(res1!=-999, res1, np.nan)),np.nanmax(res1),np.nanmin(np.where(res2!=-999, res2, np.nan)),np.nanmax(res2),np.unique(res3))
    im = sevproc.save_plot_phs(outfiles_cs['PHS'], res3, opts)
    sevproc.save_plot_phs_ql(outfiles_ql['PHS'], res3, opts, im, area_ext)

    print('---CTP---')
    opts.outvar = 'ctp'
    opts.outvarunc = 'ctp_unc'                         
    ctp = preds.predict_ctp(vis006, vis008, ir016, ir039, ir062, ir073,
                        ir082, ir108, ir120, ir134, lsm, skt,
                        solzen=solzen, satzen=satzen,
                        undo_true_refl=False, correct_vis_cal_nasa_to_impf=3,
                        cldmask=cldmask)
    res1, res2, _, res_area, area_ext = sevproc.resample_data(ds, ctp, opts)
    opts.varname = 'CTP'
    print(np.nanmin(np.where(res1!=-999, res1, np.nan)),np.nanmax(res1),np.nanmin(np.where(res2!=-999, res2, np.nan)),np.nanmax(res2))
    im = sevproc.save_plot_cmap('/home/users/dhegedus/seviri_ml/cesium_cmap_test.png', res1, opts)
    
    print('---CTT---')
    opts.outvar = 'ctt' 
    opts.outvarunc = 'ctt_unc'
    ctt = preds.predict_ctt(vis006, vis008, ir016, ir039, ir062, ir073,
                        ir082, ir108, ir120, ir134, lsm, skt,
                        solzen=solzen, satzen=satzen,
                        undo_true_refl=False, correct_vis_cal_nasa_to_impf=3,
                        cldmask=cldmask)
    res1, res2, _, res_area, area_ext = sevproc.resample_data(ds, ctt, opts)
    opts.varname = 'CTT'
    print(np.nanmin(np.where(res1!=-999, res1, np.nan)),np.nanmax(res1),np.nanmin(np.where(res2!=-999, res2, np.nan)),np.nanmax(res2))
    im = sevproc.save_plot_cmap('/home/users/dhegedus/seviri_ml/cesium_ctt_test.png', res1, opts)
    plt.show()                   
    print('---CBH---')                   
    opts.outvar = 'cbh' 
    opts.outvarunc = 'cbh_unc'                    
    cbh = preds.predict_cbh(ir108, ir120, ir134, solzen=solzen, satzen=satzen, 
                        cldmask=cldmask)
    res1, res2, _, res_area, area_ext = sevproc.resample_data(ds, cbh, opts)
    opts.varname = 'CBH'
    print(np.nanmin(np.where(res1!=-999, res1, np.nan)),np.nanmax(res1),np.nanmin(np.where(res2!=-999, res2, np.nan)),np.nanmax(res2))
    im = sevproc.save_plot_cmap('/home/users/dhegedus/seviri_ml/cesium_cbh_test.png', res1, opts)
    
    
    print('---MLAY---')        
    opts.outvar = 'mlay'
    opts.outvarunc = 'mlay_unc'
    opts.outvarflag = 'mlay_flag'                    
    mlay = preds.predict_mlay(vis006, vis008, ir016, ir039, ir062, ir073,
                        ir082, ir108, ir120, ir134, lsm, skt,
                        solzen=solzen, satzen=satzen,
                        undo_true_refl=False, correct_vis_cal_nasa_to_impf=3, 
                        cldmask=cldmask)
    
    res1, res2, res3, res_area, area_ext = sevproc.resample_data(ds, mlay, opts)
    
   
    

    
    
    
    
    
    #ds_xr = sevproc.resample_data(ds, cma, cph, ctp, ctt, cbh, mlay, opts)
    
    sevproc.plot_ml_pred(cma, cph, ctp, ctt, cbh, mlay)
    
    



main_opts = sevproc.SEVIRI_HRIT()
main(main_opts)

              