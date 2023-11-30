from glob import glob
from datetime import datetime
import xarray as xr
import rioxarray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import pathlib
import sys
sys.path.append('/home/users/dhegedus/seviri_ql')
import plotting as sev_plot

def_outvar = ('cot', 'cph', 'ctp', 'ctt', 'cbh', 'mlay')
              
def_outflag = ( 'cloudmask','phase', 'mlay_flag')
              
def_outunc = ('cma_unc', 'cph_unc', 'ctp_unc', 'ctt_unc', 'cbh_unc', 'mlay_unc')

#def_varname = ('cot', 'cph', 'ctp', 'ctt', 'cbh', 'mlay', 'cloudmask','phase', 'mlay_flag', 'cma_uncertainty', 'cph_uncertainty', 'ctp_uncertainty', 'ctt_uncertainty', 'cbh_uncertainty', 'mlay_uncertainty')

def limdict(min_ctt=180.,
            max_ctt=310.,
            min_dctt=0.,
            max_dctt=50.,
            
            min_cbh=0,
            max_cbh=15000.,
            min_dcbh=0.,
            max_dcbh=50.,
            
            min_cph=0.,
            max_cph=1.,
            min_dcph=0.,
            max_dcph=50.,
            
            min_ctp=0.,
            max_ctp=1000.,
            min_dctp=0.,
            max_dctp=50.,
            
            min_cot=0.1,
            max_cot=100.,
            min_dcot=0.,
            max_dcot=50.,
            
            min_dmlay=0.,
            max_dmlay=50.
            ):

    lim_dict = {'COT': [min_cot, max_cot],
                'dCOT': [min_dcot, max_dcot],
                
                'CBH': [min_cbh, max_cbh],
                'dCBH': [min_dcbh, max_dcbh],
                
                'CTP': [min_ctp, max_ctp],
                'dCTP': [min_dctp, max_dctp],
                
                'CPH': [min_cph, max_cph],
                'dCPH': [min_dcph, max_dcph],
                
                'CTT': [min_ctt, max_ctt],
                'dCTT': [min_dctt, max_dctt],
                
                'dMLAY': [min_dmlay, max_dmlay]
                }

    return lim_dict


class SEVIRI_HRIT:
    def __init__(self,
                 in_dtstr='202311091200',
                 indir='/gws/nopw/j04/nrt_ecmwf_metop/eumetsat/H-000-MSG3/',
                 outdir_top='/home/users/dhegedus/seviri_ml/TEST/',
                 pixellimit=[1370, 2735, 2000, 3509],
                 era5tdir='/badc/ecmwf-era5t/data/oper/an_sfc/',
                 filename_lsm='/gws/nopw/j04/aerosol_cci/proud/GEO_FILES/MSG_000E_LSM.nc',
                 res_meth='nearest',
                 outvar=def_outvar,
                 outvarflag=def_outflag,
                 outvarunc=def_outunc,
                 varname='',
                 outlims=limdict(),
                 logscl=False,
                 title_stub='NCEO-L2-CLOUD-AEROSOL-SEVIRI_ORAC_MSG3_'
                  ):
        # Directory containing the ORAC pri + sec files
        self.in_dtstr=in_dtstr
        self.indir = indir
        self.era5tdir = era5tdir
        self.dater = datetime.strptime(self.in_dtstr, "%Y%m%d%H%M")
        self.subdir = self.dater.strftime("%Y/%m/%d/")
        self.pixellimit = pixellimit
        self.filename = glob(self.indir+'/'+self.subdir+'*'+self.in_dtstr+'*')
        #glob('{}/{}H-000-MSG3__-MSG3________-*{}*'.format(self.indir,self.subdir,self.in_dtstr))
        self.filename_lsm=filename_lsm
        self.filename_skt = glob(self.era5tdir+self.subdir+'ecmwf-era5t_oper_an_sfc_'+self.in_dtstr+'.skt.nc')[0]
        self.res_meth = res_meth
        self.outvar = outvar
        self.outvarflag = outvarflag
        self.outvarunc = outvarunc
        self.varname = varname
        self.outlims = outlims
        self.logscl = logscl
        self.outdir_top = outdir_top
        self.title_stub = title_stub


def set_output_files(opts, cesium=True, var_out_list=('cot', 'cph', 'ctp', 'ctt', 'cbh', 'mlay', 'cloudmask','PHS', 'mlay_flag', 'cma_unc', 'cph_unc', 'ctp_unc', 'ctt_unc', 'cbh_unc', 'mlay_unc')):
    """Define and create output filenames for quicklooks based on the date.
    Inputs:
     - odir: String, output directory.
     - pri_fname: String, the filename of the ORAC primary file.
     - offset: Int, filename position offset from 'SEVIRI_ORAC' that gives timestamp.
     - var_out_list: List of strings, variables to save.
    Returns:
     - out_fnames: Dictionary, output filenames for saving.
     - need_proc: Boolean, do we need to do processing. False if all files already present."""
    if cesium:
        outdir = f'{opts.outdir_top}seviri_ml/quick_look_cesium/{opts.subdir}'
    else:
        outdir = f'{opts.outdir_top}seviri_ml/quick_look_hires/{opts.subdir}'
       
    print(outdir) 
    print(glob(outdir))
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    
    out_fnames = {}
    for var in var_out_list:
        out_fnames[var] = f'{outdir}{opts.title_stub}{opts.in_dtstr}_{var}.png'
        print(out_fnames[var])
    return out_fnames

def test_files(fdict):
    """Determine if all required output files are present.
    Inputs:
     - flist: Dictinary of filenames to test.
    Returns:
     - Boolean, true of all files are present."""
    if all([os.path.isfile(fdict[k]) for k in fdict.keys()]):
        return True
    else:
        return False

             
def load_seviri_hrit(filename, pixellimit):
    from satpy import Scene
    from satpy.modifiers import angles
    ds = Scene(filenames=filename, reader='seviri_l1b_hrit', reader_kwargs={'include_raw_metadata': True})
    ds.load(['VIS006', 'VIS008', 'IR_016','IR_039', 'WV_062', 'WV_073',  'IR_087', 'IR_108', 'IR_120', 'IR_134'])
    area_def = ds['VIS006'].attrs['area']
    
    # Load channels
    vis006 = ds['VIS006'].data[pixellimit[2]:pixellimit[3], pixellimit[0]:pixellimit[1]]/100
    vis008 = ds['VIS008'].data[pixellimit[2]:pixellimit[3], pixellimit[0]:pixellimit[1]]/100
    ir016 = ds['IR_016'].data[pixellimit[2]:pixellimit[3], pixellimit[0]:pixellimit[1]]/100
    ir039 = ds['IR_039'].data[pixellimit[2]:pixellimit[3], pixellimit[0]:pixellimit[1]]
    ir062 = ds['WV_062'].data[pixellimit[2]:pixellimit[3], pixellimit[0]:pixellimit[1]]
    ir073 = ds['WV_073'].data[pixellimit[2]:pixellimit[3], pixellimit[0]:pixellimit[1]]
    ir082 = ds['IR_087'].data[pixellimit[2]:pixellimit[3], pixellimit[0]:pixellimit[1]]
    ir108 = ds['IR_108'].data[pixellimit[2]:pixellimit[3], pixellimit[0]:pixellimit[1]]
    ir120 = ds['IR_120'].data[pixellimit[2]:pixellimit[3], pixellimit[0]:pixellimit[1]]
    ir134 = ds['IR_134'].data[pixellimit[2]:pixellimit[3], pixellimit[0]:pixellimit[1]]
    
    
    # Load satelite and solar zenith angles
    solzen = angles.get_angles(ds['VIS006'])[3].values[pixellimit[2]:pixellimit[3], pixellimit[0]:pixellimit[1]]
    satzen = angles.get_angles(ds['VIS006'])[1].values[pixellimit[2]:pixellimit[3], pixellimit[0]:pixellimit[1]]
    
    return vis006, vis008, ir016, ir039, ir062, ir073, ir082, ir108, ir120, ir134, satzen, solzen, ds

def load_lsm(filename_lsm, pixellimit):
    # Land sea mask
    ds_lsm = xr.open_dataset(filename_lsm)
    lsm = ds_lsm['Land_Sea_Mask'].data[pixellimit[2]:pixellimit[3], pixellimit[0]:pixellimit[1]]
    return lsm
    
def load_skt(filename_skt, ds, pixellimit):
    from rasterio.enums import Resampling
    area_def = ds['VIS006'].attrs['area']
    ds_skt = xr.open_dataset(filename_skt)
    ds_skt.rio.write_crs("epsg:4326", inplace=True)
    ds_skt = ds_skt.rename({'longitude':'x','latitude':'y'})
    reprojected_skt = ds_skt['skt'].rio.reproject(dst_crs=area_def.crs_wkt, resampling=Resampling.bilinear)
    resulted_proj = reprojected_skt.isel(time=0).interp_like(ds['VIS006'])
    skt = resulted_proj.values[pixellimit[2]:pixellimit[3], pixellimit[0]:pixellimit[1]]
    return skt
    
def save_data_to_nc(ds, satzen, solzen, lsm, skt, outname):
    from satpy.modifiers import angles
    dsnew = ds.to_xarray(['VIS006', 'VIS008', 'IR_016','IR_039', 'WV_062', 'WV_073',  'IR_087', 'IR_108', 'IR_120', 'IR_134'])
    dsnew['satzen'] = angles.get_angles(ds['VIS006'])[3]
    dsnew['solzen'] = angles.get_angles(ds['VIS006'])[1]
    dsnew['skt'] = resulted_proj
    dsnew['lsm'] = (('y','x'), ds_lsm['Land_Sea_Mask'].values)
    dsnew.to_netcdf('./ml_trial.nc')

def resample_data(ds, vararray, opts, roi=50000, fill_value=np.nan):
    from pyresample import create_area_def, geometry, image
    from satpy import resample
    import xarray as xr
    ds_xr = ds[opts.pixellimit[2]:opts.pixellimit[3], opts.pixellimit[0]:opts.pixellimit[1]].to_xarray(['VIS006'])
    
    if len(vararray) == 2:
        ds_xr[opts.outvar] = (('y', 'x'), np.where(vararray[0] < 0, np.nan, vararray[0]))
        ds_xr[opts.outvarunc] = (('y', 'x'), np.where(vararray[1] <= 0, np.nan, vararray[1]))
    else:
        if opts.outvar == 'cph':
            ds_xr[opts.outvar] = (('y', 'x'), np.where(vararray[0] < 0, np.nan, vararray[0]))
            ds_xr[opts.outvarflag] = (('y', 'x'), np.where(vararray[1] < 0, np.nan, vararray[1]))
            ds_xr[opts.outvarunc] = (('y', 'x'), np.where(vararray[2] <= 0, np.nan, vararray[2]))
        else:
            ds_xr[opts.outvar] = (('y', 'x'), np.where(vararray[0] < 0, np.nan, vararray[0]))
            ds_xr[opts.outvarflag] = (('y', 'x'), np.where(vararray[1] < 0, np.nan, vararray[1]))
            ds_xr[opts.outvarunc] = (('y', 'x'), np.where(vararray[2] < 0, np.nan, vararray[2]))
    
    ds_xr = ds_xr.drop_vars('VIS006')

    lats = ds_xr['latitude']
    lats = np.where(lats > -90, lats, np.nan)
    lats = np.where(np.isfinite(lats), lats, np.nan)
    lons = ds_xr['longitude']
    lons = np.where(lons > -180, lons, np.nan)
    lons = np.where(np.isfinite(lons), lons, np.nan)
    lons = xr.DataArray(lons, dims=["y", "x"])
    lats = xr.DataArray(lats, dims=["y", "x"])
    
    lat_max = lats.max().values
    lat_min = lats.min().values
    lon_max = lons.max().values
    lon_min = lons.min().values
    pix_height = math.floor((lat_max-lat_min)/0.03)
    pix_width = math.floor((lon_max-lon_min)/0.03)
    
    indata_def = geometry.SwathDefinition(lats=lats, lons=lons)
    area_def = ds['VIS006'][opts.pixellimit[2]:opts.pixellimit[3], opts.pixellimit[0]:opts.pixellimit[1]].attrs['area']
    area_def2 = area_def
    xy_bbox = [ds['VIS006'][opts.pixellimit[2]:opts.pixellimit[3], opts.pixellimit[0]:opts.pixellimit[1]].x.min().data,
               ds['VIS006'][opts.pixellimit[2]:opts.pixellimit[3], opts.pixellimit[0]:opts.pixellimit[1]].y.min().data,
               ds['VIS006'][opts.pixellimit[2]:opts.pixellimit[3], opts.pixellimit[0]:opts.pixellimit[1]].x.max().data,
               ds['VIS006'][opts.pixellimit[2]:opts.pixellimit[3], opts.pixellimit[0]:opts.pixellimit[1]].y.max().data]

    area_def_crop, _, _ = ds._slice_area_from_bbox(area_def, area_def2, xy_bbox=xy_bbox)    
    area_def = create_area_def('test_area',
                               {'proj': 'latlong', 'lon_0': 0},
                               area_extent=(lon_min, lat_min, lon_max, lat_max),
                               width=pix_width,
                               height=pix_height)
                               
    res1 = resample.resample(area_def_crop,
                                ds_xr[opts.outvar],
                                area_def,
                                resampler=opts.res_meth,
                                reduce_data=False,
                                radius_of_influence=roi,
                                fill_value=fill_value,
                                cache_dir='/gws/pw/j07/rsgnceo/Data/seviri_msg3/nrt_processing/cache_dir/')
                                
    
    res2 = resample.resample(area_def_crop,
                                ds_xr[opts.outvarunc],
                                area_def,
                                resampler=opts.res_meth,
                                reduce_data=False,
                                radius_of_influence=roi,
                                fill_value=fill_value,
                                cache_dir='/gws/pw/j07/rsgnceo/Data/seviri_msg3/nrt_processing/cache_dir/')
                                
    if len(vararray) == 3:
        res3 = resample.resample(area_def_crop,
                                ds_xr[opts.outvarflag],
                                area_def,
                                resampler=opts.res_meth,
                                reduce_data=False,
                                radius_of_influence=roi,
                                fill_value=fill_value,
                                cache_dir='/gws/pw/j07/rsgnceo/Data/seviri_msg3/nrt_processing/cache_dir/')
                                
    else: 
        res3 = None

    return res1, res2, res3, area_def, (lon_min, lon_max, lat_min, lat_max)
    



def save_plot_phs(fname, phs_data, opts):
    from PIL import Image
    
    # Find bands and scale to percentile
    b1 = phs_data
    b2 = phs_data
    b3 = phs_data

    b1 = np.where(b1==2, 1, 0) #Blue = ice
    b2 = np.where(b2==np.nan, np.nan, 0) #Green = clear
    b3 = np.where(b3==1, 1, 0) #Red = liquid
    
    
    data = np.dstack((b1, b2, b3))
    pts = (data < 0).nonzero()
    data[pts] = 0
    
    data = np.round(np.where(data > 1, 1., data) * 255)
    data = data.astype(np.ubyte)
    
    save_fc = sev_plot.make_alpha(data[:, :, ::-1])
    save_fc[:,:,3] = np.where((save_fc[:, :,0]==0) & (save_fc[:, :,1]==0) & (save_fc[:, :,2]==0), 0, save_fc[:,:,3])
    #save_fc[:,:,3] = np.where((save_fc[:, :,0]==25) & (save_fc[:, :,1]==25) & (save_fc[:, :,2]==25), 0, save_fc[:,:,3])
    img = Image.fromarray(save_fc)      
    img.save(fname)
    return img
    
def save_plot_phs_ql(fname, data, opts, im, area_ext):
    """Save plot-ready phase class data to quicklook file."""
    import matplotlib.ticker as mticker
    import matplotlib.patches as mpatches
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import cartopy.crs as ccrs

    # Set up plot
    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()), dpi=300)
    cax = fig.add_axes([0.97, 0.175, 0.02, 0.64]) 
    ax.coastlines(lw=0.3)
    ax.set_xticks(np.arange(round(area_ext[0].item(),-1),round(area_ext[1].item(), -1),20), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(round(area_ext[2].item(),-1),math.ceil(area_ext[3].item()/10.0)*10,10), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    #ax.set_title(opts.title)
    cur_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('phasecmap', [[100,0,0], [0,0,100]], 2)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap=cur_cmap),
             cax=cax, orientation='vertical', label='Phase')
    cbar.ax.set_yticks(np.arange(0.25, 1.25, 0.5))
    cbar.ax.set_yticklabels( ['liquid', 'ice'])
    
    # Overlay image of data
    ax.imshow(im, extent=area_ext)
    
    # Save quicklook figure
    fig.savefig(fname, bbox_inches='tight')
   
def save_plot_cmap(fname, data, opts, area_ext=None, fill_value=np.nan, data_filt=None):
    from PIL import Image
    data_proc = np.copy(data)
    data_proc = np.where(data_proc==np.nan, fill_value, data_proc)
    # Get the colormap
    cur_cmap = sev_plot.assign_cmap_cld('/home/users/dhegedus/seviri_ql/cube1_0-1.csv')
    #cur_cmap = opts.cmap_cld.copy()
            
    # Find the correct range limits for a given variable, in log scale if needed
    if opts.logscl:
        rng_min = np.log10(opts.outlims[opts.varname][0])
        rng_max = np.log10(opts.outlims[opts.varname][1])
        data_proc = np.log10(data_proc)
        data_proc = np.where(np.isfinite(data_proc), data_proc, fill_value)
    else:
        rng_min = opts.outlims[opts.varname][0]
        rng_max = opts.outlims[opts.varname][1]
        print(rng_min, rng_max)
    
    # Set data lims for plotting and init mask
    #mask = data_proc.copy()
    #data_proc = np.where(data_proc < 0, fill_value, data_proc)
    data_proc = np.where(data_proc < rng_min, rng_min, data_proc)
    data_proc = np.where(data_proc > rng_max, rng_max, data_proc)
    data_proc = np.where(data_proc==np.nan, -999, data_proc)
    mask = data_proc.copy()
    # Populate mask
    mask = np.where(mask ==-999, 0, 255)
    plt.figure()
    pim=plt.imshow(data_proc)
    plt.colorbar(pim)
    # Normalise data
    data_proc = data_proc / rng_max
    
    # Make the image and save
    im = np.uint8(cur_cmap(data_proc) * 255)
    #im[:, :, 3] = mask
    plt.figure()
    pim=plt.imshow(im)
    plt.colorbar(pim)
    #if opts.aerosol_landsea:
    #    land = regionmask.defined_regions.natural_earth_v5_0_0.land_10
    #    lat = np.linspace(area_ext[2].item(),area_ext[3].item(), data_proc.shape[0])
    #    lon = np.linspace(area_ext[0].item(),area_ext[1].item(), data_proc.shape[1])
    #    mask2 = land.mask(lon, lat)
    #    mask = np.where(np.flipud(mask2)==0, 0,mask)
        
    #im[:, :, 3] = mask   
    img = Image.fromarray(im)
    img.save(fname)
    return im

def plot_ml_pred(cma, cph, ctp, ctt, cbh, mlay):
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    
    cot_prediction = np.where(cma[0] < 0, np.nan, cma[0])
    cldmask = np.where(cma[1] < 0, np.nan, cma[1])
    cma_uncertainty = np.where(cma[2] < 0, np.nan, cma[2])
    
    cph_prediction = np.where(cph[0] < 0, np.nan, cph[0])
    phase = np.where(cph[1] < 0, np.nan, cph[1])
    cph_uncertainty = np.where(cph[2] <= 0, np.nan, cph[2])
    
    pressure = np.where(ctp[0] < 0, np.nan, ctp[0])
    ctp_uncertainty = np.where(ctp[1] <= 0, np.nan, ctp[1])
    
    temperature = np.where(ctt[0] < 0, np.nan, ctt[0])
    ctt_uncertainty = np.where(ctt[1] <= 0, np.nan, ctt[1])
    
    baseheight = np.where(cbh[0] < 0, np.nan, cbh[0])
    baseheight_unc = np.where(cbh[1] <= 0, np.nan, cbh[1])
    
    mlay_prediction = np.where(mlay[0] < 0, np.nan, mlay[0])
    mlay_flag = np.where(mlay[1] < 0, np.nan, mlay[1])
    mlay_uncertainty = np.where(mlay[2] <0, np.nan, mlay[2])
    
    IPROJ = ccrs.Geostationary()
    OPROJ = ccrs.Geostationary()
    fig = plt.figure(figsize=(13,9))
    
    ax = fig.add_subplot(541, projection=OPROJ)
    ims = ax.imshow(cot_prediction, transform=IPROJ)
    ax.set_title('COT prediction')
    plt.colorbar(ims)
    
    ax = fig.add_subplot(542, projection=OPROJ)
    ims = ax.imshow(cldmask, transform=IPROJ, interpolation='none')
    ax.set_title('Binary CMA')
    plt.colorbar(ims)
    
    ax = fig.add_subplot(543, projection=OPROJ)
    ims = ax.imshow(cma_uncertainty, transform=IPROJ)
    ax.set_title('CMA uncertainty')
    plt.colorbar(ims)
    
    ax = fig.add_subplot(544, projection=OPROJ)
    ims = ax.imshow(cph_prediction, transform=IPROJ, interpolation='none')
    ax.set_title('CPH prediction')
    plt.colorbar(ims)
    
    ax = fig.add_subplot(545, projection=OPROJ)
    ims = ax.imshow(phase, transform=IPROJ, interpolation='none')
    ax.set_title('Binary CPH')
    plt.colorbar(ims)
    
    ax = fig.add_subplot(546, projection=OPROJ)
    ims = ax.imshow(cph_uncertainty, transform=IPROJ, interpolation='none')
    ax.set_title('CPH uncertainty')
    plt.colorbar(ims)
    
    ax = fig.add_subplot(547, projection=OPROJ)
    ims = ax.imshow(pressure, transform=IPROJ, interpolation='none')
    ax.set_title('CTP')
    plt.colorbar(ims)
    
    ax = fig.add_subplot(548, projection=OPROJ)
    ims = ax.imshow(ctp_uncertainty, transform=IPROJ, interpolation='none')
    ax.set_title('CTP uncertainty')
    plt.colorbar(ims)
    
    ax = fig.add_subplot(5,4,9, projection=OPROJ)
    
    ax = fig.add_subplot(5,4,10, projection=OPROJ)
    ims = ax.imshow(mlay_prediction, transform=IPROJ, interpolation='none')
    ax.set_title('MLAY prediction')
    plt.colorbar(ims)
    
    ax = fig.add_subplot(5,4,11, projection=OPROJ)
    ims = ax.imshow(mlay_flag, transform=IPROJ, interpolation='none')
    ax.set_title('MLAY flag')
    plt.colorbar(ims)
    
    ax = fig.add_subplot(5,4,12, projection=OPROJ)
    ims = ax.imshow(mlay_uncertainty, transform=IPROJ, interpolation='none')
    ax.set_title('MLAY uncertainty')
    plt.colorbar(ims)
    
    ax = fig.add_subplot(5,4,13, projection=OPROJ)
    ims = ax.imshow(temperature, transform=IPROJ, interpolation='none')
    ax.set_title('CTT')
    plt.colorbar(ims)
    
    ax = fig.add_subplot(5,4,14, projection=OPROJ)
    ims = ax.imshow(ctt_uncertainty, transform=IPROJ, interpolation='none')
    ax.set_title('CTT uncertainty')
    plt.colorbar(ims)
    
    ax = fig.add_subplot(5,4,15, projection=OPROJ)
    ims = ax.imshow(baseheight, transform=IPROJ, interpolation='none')
    ax.set_title('CBH')
    plt.colorbar(ims)
    
    ax = fig.add_subplot(5,4,16, projection=OPROJ)
    ims = ax.imshow(baseheight_unc, transform=IPROJ, interpolation='none')
    ax.set_title('CBH uncertainty')
    plt.colorbar(ims)
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    plt.savefig('./testplot.png')
    plt.show()
