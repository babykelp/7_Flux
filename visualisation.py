# %%
# @author=Renata Zigangirova zigangirovarenata@yandex.ru
# v.1 November 2023 

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, LinearRing, MultiPolygon
import xarray as xr
import natsort
from glob import glob

#for plotting 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import wesanderson

# %%
#create stable colormap for 28 classes (run once)
num_classes = 28
colors = np.random.rand(num_classes, 3)  # Generate random RGB values for each class
custom_cmap = plt.cm.colors.ListedColormap(colors)

cmap = plt.cm.twilight
new_cmap = mcolors.LinearSegmentedColormap.from_list(
    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0, b=0.9),
    cmap(np.linspace(0, 0.9, int(cmap.N * 0.9)))
)

# %%
def shift_geometry(geom):
    # function for bringing Chukotka back in a geodataframe
    if geom.is_empty:
        return geom

    if geom.geom_type == 'MultiPolygon':
        shifted_polys = [shift_geometry(p) for p in geom.geoms]
        return MultiPolygon([p for p in shifted_polys if not p.is_empty])

    if geom.geom_type == 'Polygon':
        ext = shift_geometry(geom.exterior)
        intiors = [shift_geometry(interior) for interior in geom.interiors]
        return Polygon(ext, intiors)

    if geom.geom_type == 'LinearRing':
        coords = np.array(geom.coords)
        for idx, (x, y) in enumerate(coords):
            if x < 0:  
                coords[idx] = (x + 360, y)
        return LinearRing(coords)

    return geom

def f_loadEnv(env_folder_path, astype='float16'):
    # Read all environmental data in the given folder and concatenate to a single -xarray- DataSet.
    # Data should be stored in -tif-, have unique projection and resolution. Output coodrinates -dim- should call -x- and -y-.
    # params:
    #   env_folder_path : -str- : path to the folder
    #   astype: 'float16' 'float32' 'float64' 'int' : how to load the data

    flist = natsort.natsorted(glob(env_folder_path + '*.tif'))
    print('Find files: ' , len(flist) )
    # Open each TIF file as a separate xarray DataArray
    try:
        arrays=[]
        names=[]
        for n,f in enumerate(flist):
            arrays.append( xr.open_dataset(f, engine="rasterio") ) #chunks='auto') )#.astype('float16') )
            if arrays[n].band.size != 1:
                for k in range(arrays[n].band.size):
                    names.append(f.split('\\')[-1].split('.tif')[0] + '_layer' + str(k))
            else:
                names.append( f.split('\\')[-1].split('.tif')[0] )
        ds = xr.concat(arrays, dim='band').astype('float16')
        ds = ds.assign_coords(band=names)
        print ('    load done')
        return ds
    except:
        print('Something fails')

def f_loadOcc(stations_path, env, sep=";", y='Latitude', x='Longitude', crs="EPSG:4326"):
    # Read the file with stations coordinates and attach the environmental parameters.
    # params:
    #   occ_csv_path : -str- : path to the csv file
    #   env : -xarray DataSet- : environmental data load with -f_loadEnv- function
    #   sep : -str- : csv columns separator
    #   y, x : -str- : columns' names for -Y- and -X-
    #   crs : -str- : code of the coordinates projection


    df = pd.read_csv(stations_path, sep=sep)
    print("Find in CSV # flux stations: ", len(df))
    # check the data storage type
    if type(df[x][0]) == str:
        df[y] = df[y].apply(lambda q: pd.to_numeric(q.replace(',', '.')))
        df[x] = df[x].apply(lambda q: pd.to_numeric(q.replace(',', '.')))
    df['IDcode'] = df.index + 1 # digital ID

    # get the -env- data in points
    x_loc = xr.DataArray(df[x], dims=['location'])
    y_loc = xr.DataArray(df[y], dims=['location'])
    data = env.sel(x=x_loc, y=y_loc, method='nearest')
    dff = data.to_dataframe().reset_index()
    del dff['x'], dff['y'], dff['spatial_ref']
    dff = dff.pivot(index = 'location', columns = 'band', values = 'band_data')
    df = df.join(dff) # location index + columns from source scv + data from all -env- band
    df = gpd.GeoDataFrame( df, geometry = gpd.points_from_xy(df.Longitude, df.Latitude), crs = crs)
    print('     load done. ')
    #print('     Output columns: ', df.columns)

    return df

def add_occ_to_csv(dataset_name, occ_values, file_name):
    """
    function for adding falues of stations to csv for easier access
    """
    if not os.path.exists(file_name) or os.stat(file_name).st_size == 0:
        df = pd.DataFrame({dataset_name: occ_values})
    else:
        df = pd.read_csv(file_name)
        df[dataset_name] = occ_values
    
    df.to_csv(file_name, index=False)

def extract_data(dataset):
    try:
        return dataset.band_data.to_numpy().ravel()
    except:
        return dataset.to_numpy().ravel()

def plot_stars(ax, x, y):
    ax.scatter(x, y, s=27, color='white', alpha=0.8, marker='*', zorder = 20)
    ax.scatter(x, y, s=13, color='black', alpha=0.9, marker='*', zorder = 20)

def clip_dataset(dataset, mask_geometry, mask_crs):
    return dataset.rio.clip(mask_geometry, mask_crs)

def plot_data_on_ax(ax, data, name, vmin, vmax):
    if vmin == True and vmax == True:
        if hasattr(data, 'band_data'):  # check if 'band_data' attribute exists
            if name[0] == 'P':
                mappable = data.band_data[0].plot.pcolormesh(ax=ax, add_colorbar=False) 
            else: 
                mappable = data.band_data[0].plot.pcolormesh(ax=ax, add_colorbar=False, cmap=new_cmap) 
        else:
            mappable = data.plot.pcolormesh(ax=ax, add_colorbar=False, cmap=new_cmap) 
        return mappable
    else:
        if hasattr(data, 'band_data'):  # check if 'band_data' attribute exists
            if name[0] == 'P':
                mappable = data.band_data[0].plot.pcolormesh(ax=ax, add_colorbar=False, vmin=vmin, vmax=vmax) 
            else: 
                mappable = data.band_data[0].plot.pcolormesh(ax=ax, add_colorbar=False, cmap=new_cmap, vmin=vmin, vmax=vmax) 
        else:
            mappable = data.plot.pcolormesh(ax=ax, add_colorbar=False, cmap=new_cmap, vmin=vmin, vmax=vmax) 
        return mappable

def plot_maps(dataset1, name1, units, maskRussia, stations, vmin, vmax):
    data_clipped = clip_dataset(dataset1, maskRussia.geometry.values, maskRussia.crs)
    
    fig,ax = plt.subplots(figsize = (8,3))
    mappable = plot_data_on_ax(ax, data_clipped, name1, vmin, vmax)
    cbar = plt.colorbar(mappable, label=units, ax=ax)
    cbar.set_label(units, rotation=270, labelpad=20)

    plot_stars(ax, stations.geometry.x, stations.geometry.y)

    ax.set_aspect('auto', adjustable='box')

    plt.title(name1)
    plt.show()

def mask_YNS (dataYNS, stID, dataYNSprob, maskRussia):
    dataYNSmask = stID.where((dataYNS.band_data[0]==1), 0)
    dataYNSmask = dataYNSmask.where((dataYNSprob.band_data[0]>=0.8), 0)

    daYNS_mid =  dataYNS.where((dataYNSprob.band_data[0]<0.8), 0)

    dataYNSmasked = clip_dataset(dataYNSmask, maskRussia.geometry.values, maskRussia.crs)
    daYNS_mid = clip_dataset(daYNS_mid, maskRussia.geometry.values, maskRussia.crs)
    dataYNS = clip_dataset(dataYNS, maskRussia.geometry.values, maskRussia.crs)
    
    return dataYNSmasked.where(dataYNSmasked != 0), dataYNS.where(daYNS_mid != 0)

def plot_maps_mask_stID(dataYNS, dataYNSprob, maskRussia, stations, legend, name, stID, class_labels):
    data_clipped, data_mid = mask_YNS(dataYNS, stID, dataYNSprob, maskRussia)
    fig, ax = plt.subplots(figsize = (8,3))
    
    #if you want to save the file
    #data_clipped.rio.to_raster(f'{keyoo}_YNS_Prob_StID.tif') 

    data_clipped.band_data.plot(ax=ax,zorder=10, cmap = custom_cmap, edgecolor="none", add_colorbar = False)

    # if you want to add regions with less than 0.8 probability of a station
    # data_mid.band_data.plot(ax=ax, cmap="binary", edgecolor="none", zorder=2, add_colorbar = False) 
    
    maskRussia.plot(ax=ax, color="lightgray", edgecolor="none", zorder=0)
    
    plot_stars(ax, stations.geometry.x, stations.geometry.y)

    ax.set_aspect('auto', adjustable='box')
     
    class_colors = [custom_cmap(i / len(class_labels)) for i in range(len(class_labels))]
    legend_patches = [Patch(color=color, label=label) for label, color in zip(class_labels, class_colors)]
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend(handles=legend_patches, title='ID вышки', loc='center left', bbox_to_anchor=(1, 0.5), ncol = 2)
    plt.title(legend[name])
    plt.show()

def plot_maps_mask(dataYNS, dataYNSprob, maskRussia, stations, legend, name):
    
    dataYNSmask = dataYNS.where((dataYNSprob.band_data[0]>=0.8), 0)
    daYNS_mid =  dataYNS.where((dataYNSprob.band_data[0]<0.8), 0)

    # 0 and 1 values
    dataYNSmask = clip_dataset(dataYNSmask, maskRussia.geometry.values, maskRussia.crs)
    dataYNSmask = dataYNSmask.where(dataYNSmask != 0)
    daYNS_mid = clip_dataset(daYNS_mid, maskRussia.geometry.values, maskRussia.crs)
    daYNS_mid = daYNS_mid.where(daYNS_mid != 0)

    fig, ax = plt.subplots(figsize = (8,3))
    dataYNSmask.band_data[0].plot(ax=ax,zorder=10, cmap = 'Reds', edgecolor="none", add_colorbar = False) 
    
    daYNS_mid.band_data.plot(ax=ax, cmap="binary", edgecolor="none", zorder=2, add_colorbar = False)
    maskRussia.plot(ax=ax, color="lightgray", edgecolor="none", zorder=0)
    
    plot_stars(ax, stations.geometry.x, stations.geometry.y)
    ax.set_aspect('auto', adjustable='box')
    legend_patches = ['Территории, представленные вышками с P > 0.8', 'Территории, представленные вышками с P < 0.8', 'Территории, не представленные вышками']
    class_colors = [plt.cm.Reds(1/2), plt.cm.binary(1/2), 'lightgray']
    legend_patches = [Patch(color=color, label=label) for label, color in zip(legend_patches, class_colors)]
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor = (0.5,-0.15))
    plt.title(legend[name])
    plt.show()

def generate_density_scatterplot(dataset1, dataset2, name_dataset1, name_dataset2, x_label, y_label):
    
    x = extract_data(dataset1)
    y = extract_data(dataset2)
    
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    
    df_occ = pd.read_csv("combined_occ_data.csv")
    occ_x = df_occ[name_dataset1].values
    occ_y = df_occ[name_dataset2].values

    fig, ax = plt.subplots()
    
    plt.hist2d(x, y, bins=(100, 100), cmap=new_cmap, cmin=1)
    plot_stars(ax, occ_x, occ_y)
    plt.colorbar(label='Density')
    plt.xlim(100,400)
    plt.ylim(35,150)
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xticks()
    plt.show()

def compute_statistics(LCOVER, lctYPE, data, name1, mask):
    """
    Compute the statistics from the data.
    """
    # Drop the 'band' dimension if it exists in the data dataset.
    if 'band' in data.dims:
        data = data.squeeze('band', drop=True)
    
    data = clip_dataset(data, mask.geometry.values, mask.crs)

    # Resample data to match LCover's resolution
    data = data.interp(x=LCOVER['x'], y=LCOVER['y'])
    
    # Using the .where method to mask the values
    datalc = data.where(LCOVER == lctYPE, np.nan)
    
    if hasattr(datalc, 'band_data'):  # check if 'band_data' attribute exists
        mean_value = datalc.band_data.mean().item()
        std_dev = datalc.band_data.std().item()
    else:
        mean_value = datalc.mean().item()
        std_dev = datalc.std().item()
    return [mean_value, std_dev]

def mask_LC (LC, dataYNS, dataYNSprob, maskRussia):

    dataYNSmask = dataYNS.where((dataYNSprob.band_data[0]>=0.8), 0)
    if 'band' in dataYNSmask.dims:
        dataYNSmask = dataYNSmask.squeeze('band', drop=True)
    # 0 and 1 values masked by russian border
    dataYNSmasked = clip_dataset(dataYNSmask, maskRussia.geometry.values, maskRussia.crs)
    dataYNSmasked = dataYNSmasked.interp(x=LC['x'], y=LC['y'])
    # return values of climate data where it is representative
    return clip_dataset(LC, maskRussia.geometry.values, maskRussia.crs).where(dataYNSmasked == 1)

# %%
# data storage locations

base = '/media/babykelp/Expansion/FLUXNETstations'

stations_path = base + "/FluxTower_pnt/Russian_Fluxes_EC_Towers_1998_2022_v6.csv"
space_border = base + "/cropLimit_plg/admin_level_3_InLand_WGS84.shp" # to manage the PAmask and thus PA random selection
clim_path = base + "/DataForWork/climate/"
soil_path = base + "/DataForWork/soil/"
topo_path = base + "/DataForWork/terrain/"
GPP_path = base + "/DataForWork/GPPmodis/"
ET_path = base + "/DataForWork/ETmodis/"
PCA_path = base + "/DataForWork/pca/"

PAmask_path = base + '/DataForWork/PAmasks/'
results_path = base + '/PREDICTED_OUTPUT/'

# %%
#loading mask and stations shapefiles
mask_WGS84 = gpd.read_file(space_border)
mask_WGS84['geometry'] = mask_WGS84['geometry'].apply(shift_geometry)
stations = gpd.read_file(base + '/FluxTower_pnt/FluxTower_v6_pnt.shp')

# %% [markdown]
# ### Data preparation for plotting

# %%
# choose 2 most important parameters from each factor

clim3 = xr.open_dataset(clim_path + 'wc2.0_bio_30s_03_RU.tif', engine = 'rasterio')
clim15 = xr.open_dataset(clim_path + 'wc2.0_bio_30s_15_RU.tif', engine = 'rasterio')

soil0 = xr.open_dataset(soil_path + 'crop_soc_3layers.tif', engine = 'rasterio').band_data[0]
soilN = xr.open_dataset(soil_path + 'crop_nitrogen_3layers.tif', engine = 'rasterio').band_data[0]

terrTPI = xr.open_dataset(topo_path + 'MeritDEM_MultiscTPI_RU.tif', engine = 'rasterio')
terr3sec = xr.open_dataset(topo_path + 'MeritDEM_3sec_mosaic_RU.tif', engine = 'rasterio')

GPPmean9 = xr.open_dataset(GPP_path + 'MOD17_GPP_MONTH_mean9_RU_interp.tif', engine = 'rasterio')
GPPmean7 = xr.open_dataset(GPP_path + 'MOD17_GPP_MONTH_mean7_RU_interp.tif', engine = 'rasterio')

ETmean9 = xr.open_dataset(ET_path + 'MOD16_ET_MONTH_mean9_RU_interp.tif', engine = 'rasterio')
ETmean7 = xr.open_dataset(ET_path + 'MOD16_ET_MONTH_mean7_RU_interp.tif', engine = 'rasterio')

pca5 = xr.open_dataset(PCA_path + 'PCA_ClimSoilTopo_comp5.tif', engine = 'rasterio')
pca6 = xr.open_dataset(PCA_path + 'PCA_ClimSoilTopo_comp6.tif', engine = 'rasterio')

# load model results YES/NO Station mode
climYNS = xr.open_dataset(results_path + 'climate_YNS_class_10RUNS.tif', engine = 'rasterio')
climYNSprob = xr.open_dataset(results_path + 'climate_YNS_prob_10RUNS.tif', engine = 'rasterio')

soilYNS = xr.open_dataset(results_path + 'soil_YNS_class_10RUNS.tif', engine = 'rasterio')
soilYNSprob = xr.open_dataset(results_path + 'soil_YNS_prob_10RUNS.tif', engine = 'rasterio')

terrYNS = xr.open_dataset(results_path + 'terrain_YNS_class_10RUNS.tif', engine = 'rasterio')
terrYNSprob = xr.open_dataset(results_path + 'terrain_YNS_prob_10RUNS.tif', engine = 'rasterio')

gppYNS = xr.open_dataset(results_path + 'GPPmodis_YNS_class_10RUNS.tif', engine = 'rasterio')
gppYNSprob = xr.open_dataset(results_path + 'GPPmodis_YNS_prob_10RUNS.tif', engine = 'rasterio')

etYNS = xr.open_dataset(results_path + 'ETmodis_YNS_class_10RUNS.tif', engine = 'rasterio')
etYNSprob = xr.open_dataset(results_path + 'ETmodis_YNS_prob_10RUNS.tif', engine = 'rasterio')

pcaYNS = xr.open_dataset(results_path + 'pca_YNS_class_10RUNS.tif', engine = 'rasterio')
pcaYNSprob = xr.open_dataset(results_path + 'pca_YNS_prob_10RUNS.tif', engine = 'rasterio')

# load model results Station ID mode
clim_stID = xr.open_dataset(results_path + 'climate_stID_class_10RUNS.tif', engine = 'rasterio')
GPP_stID = xr.open_dataset(results_path + 'GPPmodis_stID_class_10RUNS.tif', engine = 'rasterio')
ET_stID = xr.open_dataset(results_path + 'ETmodis_stID_class_10RUNS.tif', engine = 'rasterio')
soil_stID = xr.open_dataset(results_path + 'soil_stID_class_10RUNS.tif', engine = 'rasterio')
terr_stID = xr.open_dataset(results_path + 'terrain_stID_class_10RUNS.tif', engine = 'rasterio')
pca_stID = xr.open_dataset(results_path + 'pca_stID_class_10RUNS.tif', engine = 'rasterio')

# %%
# # for visualisation purposes we want to know values of the parameter in the station pixel(and for neighbouring pixels)
# env_folder_path = topo_path #change for desired factor
# # load environmental data
# env = f_loadEnv(env_folder_path)
# # load station location and collect -env- data in those points
# occ = f_loadOcc(stations_path, env,
#                 sep=";", y='Latitude', x='Longitude', crs="EPSG:4326")
# # add and store them in a csv so you won't need to load them again
# add_occ_to_csv("ETmean7", occ['MOD16_ET_MONTH_mean7_RU_interp'], "combined_occ_data.csv")

# %%
#format of a tuple for all the plots is:
#                (xr.DataArray or xr.Dataset of parameter №1, 
#                 xr.DataArray or xr.Dataset of parameter №2, 
#                 name of the parameter №1 from combined_occ_data.csv, 
#                 name of the parameter №2 from combined_occ_data.csv,
#                 label of the axis for parameter №1,
#                 label of the axis for parameter №2,
#                 units of parameter №1,
#                 units of parameter №2,
#                 xr.DataArray of models YES/NO Station results,
#                 xr.DataArray of models YES/NO Station probability results, 
#                 xr.DataArray of models Station ID results)
data_pairs = [
    # (clim3, clim15, 'clim3', 'clim15',"Изотермальность", "Сезонность осадков (стандартное отклонение)", '%', 'мм', climYNS, climYNSprob, clim_stID),
    # (soil0, soilN, 'soil0', 'soiln','Органический углерод почвы (0-5 см)', 'Почвенный азот (0-5 см)', 'дг/кг', 'сг/кг', soilYNS, soilYNSprob, soil_stID),
    # (terrTPI, terr3sec, 'terr_TPI', 'terr_3sec', 'Топографический индекс позиции (TPI)', 'Высота над уровнем моря MERIT DEM', 'м', 'м', terrYNS, terrYNSprob,  terr_stID),
    # (GPPmean7,GPPmean9, 'GPPmean7', 'GPPmean9','GPP. Июль', 'GPP. Сентябрь', 'кг C/м²','кг C/м²', gppYNS, gppYNSprob, GPP_stID),
    (ETmean7, ETmean9, 'ETmean7', 'ETmean9','Среднемесячное суммарное испарение. Июль', 'Среднемесячное суммарное испарение. Сентябрь', ' кг/м²/8сут', 'кг/м²/8сут', etYNS, etYNSprob, ET_stID),
    # (pca5,pca6, 'pca5', 'pca6','PCA: компонент 5 (Климат + Почвы + Рельеф)', 'PCA: компонент 6 (Климат + Почвы+ Рельеф)', None, None, pcaYNS, pcaYNSprob, pca_stID)
]

translated_legend = {
    "clim3": "Климат",
    "soil0": "Почвы",
    "GPPmean7": "Валовая первичная продукция",
    "ETmean7": "Суммарное испарение",
    "pca5": "Метод главных компонент",
    'terr_TPI': "Рельеф"
}

# %% [markdown]
# ### Maps for whole Russian territory

# %%
#just plotting the values of the source parameters here
plotted_datasets = set()
for data_tuples in data_pairs:
    for m in range(0,2):
        if data_tuples[m+2] not in plotted_datasets:
            plot_maps(data_tuples[m], data_tuples[m+4], data_tuples[m+6],  mask_WGS84, stations, True, True) #change last 2 parameters to vmin and vmax if needed
            plotted_datasets.add(data_tuples[m+2])

# %% [markdown]
# ### Maps for FLuxnet sites represented parts only

# %%
# plotting maps of represented parts only colored with stations ID
df = pd.read_csv(stations_path, delimiter= ';')
class_labels = df['ID'].tolist()

plotted_datasets = set()
for data_tuples in data_pairs:
    if data_tuples[2] not in plotted_datasets:
        plot_maps_mask_stID(data_tuples[-3], data_tuples[-2], mask_WGS84, stations, translated_legend, data_tuples[2], data_tuples[-1], class_labels)
        plotted_datasets.add(data_tuples[2])

# %%
# plotting maps of represented parts only Y/N Stations
df = pd.read_csv(stations_path, delimiter= ';')
class_labels = df['ID'].tolist()

plotted_datasets = set()
for data_tuples in data_pairs:
    if data_tuples[2] not in plotted_datasets:
        plot_maps_mask(data_tuples[-3], data_tuples[-2], mask_WGS84, stations, translated_legend, data_tuples[2])
        plotted_datasets.add(data_tuples[2])

# %% [markdown]
# ### Density scatterplots for all Russia

# %%
# creating density scatterplots for two chosen parameters for all russian territory
for dataset_orig1, dataset_orig2, name1, name2, x_label, y_label, units1, units2, daYNS, daYNSprob, stID in data_pairs:
    
    dataset1_clipped = clip_dataset(dataset_orig1, mask_WGS84.geometry.values, mask_WGS84.crs)
    dataset2_clipped = clip_dataset(dataset_orig2, mask_WGS84.geometry.values, mask_WGS84.crs)
    
    generate_density_scatterplot(dataset1_clipped, dataset2_clipped, name1, name2, x_label, y_label)

# %%
# Density scatterplot for > 0.8 probability only 
for dataset_orig1, dataset_orig2, name1, name2, x_label, y_label, units1, units2, daYNS, daYNSprob, stID in data_pairs:

    dataset1 = mask_YNS(daYNS, dataset_orig1, daYNSprob, mask_WGS84)
    dataset2 = mask_YNS(daYNS, dataset_orig2, daYNSprob, mask_WGS84)

    generate_density_scatterplot(dataset1[0], dataset2[0], name1, name2, x_label, y_label)

# %% [markdown]
# ### Land Cover

# %%
# code for counting how many pixels were represented by each land cover
# result is excel file

# Open the LCover dataset
LCover = xr.open_dataset(base + '/DataForWork/other/Schepaschenko_LULCmap/Russia_LC_2009_WGS84_RU.tif', engine='rasterio').band_data[0]
lc_legend_df = pd.read_csv(base + "/DataForWork/other/Schepaschenko_LULCmap/LC_legend.csv")
# Placeholder for results
all_results = {}
unique_YNS = set()

for dataset_orig1, _, name1, name2, x_label, _, units1, _, daYNS, daYNSprob, _ in data_pairs:
    # Avoid processing the same YNS data multiple times
    if name1 in unique_YNS or name2 in unique_YNS:
        continue
    unique_YNS.add(name1)
    unique_YNS.add(name2)
    print(name1)
    # Intersect with YES pixels
    intersected = mask_YNS(daYNS, dataset_orig1, daYNSprob, mask_WGS84)
    # Drop the 'band' dimension if it exists in the data dataset.
    LC_intersected = mask_LC(LCover, daYNS, daYNSprob, mask_WGS84).band_data
    plt.show()
    results_for_current_pair = []
    for _, LCtype in lc_legend_df.iterrows():
        lc_value  = LCtype['VALUE']
        print(lc_value)
        land_cover = LCtype['Land cover']
        
        # Compute statistics for the entire Russia territory
        stats_orig = compute_statistics(LCover, lc_value, dataset_orig1, name1, mask_WGS84)
        
        # Compute statistics for the territory represented by towers
        stats_towers = compute_statistics(LCover, lc_value, intersected, name1, mask_WGS84)

        # Aggregate results
        result = {
            "Код LC": lc_value,
            "LandCover Type": land_cover,
            "Общее число пикселей": np.sum(LCover.values == lc_value),
            f'Среднее значение {name1}, {units1}': stats_orig[0],
            f'Стандартное отклоненение {name1}, {units1}':  stats_orig[1]
        }
        if hasattr(LC_intersected, 'band_data'):  # check if 'band_data' attribute exists
            result["Число пикселей представленных вышками"] = np.sum(LC_intersected.values == lc_value)
        else:
            result["Число пикселей представленных вышками"] = np.sum(LC_intersected.values == lc_value)

        result[f'Среднее значение {name1} на территории, представленной вышками, {units1}'] = stats_towers[0]
        result[f'Стандартное отклоненение {name1} на территории, представленной вышками, {units1}'] = stats_towers[1]
        results_for_current_pair.append(result)

    all_results[name1] = pd.DataFrame(results_for_current_pair)

# Save each dataframe to a different sheet in Excel
with pd.ExcelWriter(base + "/results_visualisation/tables/representation_LC.xlsx") as writer:
    for name, df in all_results.items():
        df.to_excel(writer, sheet_name=name)

# %%
## code for bar chart of representation by Land Cover Type

# Load the data from the files
representation_LC_data = pd.ExcelFile(base + "/results_visualisation/tables/representation_LC.xlsx")

# Aggregate the data by land cover type
landcover_counts = representation_LC_data.parse(representation_LC_data.sheet_names[0])['Число установленных вышек'].tolist()

# Extract necessary columns from the representation_LC file
landcover_types = representation_LC_data.parse(representation_LC_data.sheet_names[0])['LandCover Type'].tolist()
total_pixels = representation_LC_data.parse(representation_LC_data.sheet_names[0])['Общее число пикселей'].tolist()
tower_pixels = {sheet: representation_LC_data.parse(sheet)['Число пикселей представленных вышками'].tolist() for sheet in representation_LC_data.sheet_names}
del tower_pixels["terr_TPI"]

# Let's regenerate the chart with the reversed order of land cover types
russian_labels = [
    "Лес, сосна", "Лес, ель и пихта", "Лес, лиственница", 
    "Лес, сибирская сосна", "Лес, берёза или осина", 
    "Лес, широколиственные породы", "Лес, другие виды", 
    "Редкий лес", "Редкий лес", "Водно-болотные угодья", 
    "Сельскохозяйственные угодья", "Сельскохозяйственные угодья", 
    "Сельскохозяйственные угодья", "Травянистые сообщества", 
    "Кустарниковые сообщества", "Вода", "Непродуктивные территории"
]

# Define the translated legend
translated_legend = {
    "clim3": "Климат",
    "soil0": "Почвы",
    "GPPmean7": "Валовая первичная продукция",
    "ETmean7": "Суммарное испарение",
    "pca5": "Метод главных компонент"
}
# Reverse the order of landcover types for plotting
# Since we've added two more labels, we need to reverse the list again
russian_labels_reversed = russian_labels[::-1]
total_pixels_reversed = total_pixels[::-1]
tower_pixels_reversed = {k: v[::-1] for k, v in tower_pixels.items()}

# Plotting parameters
gray_bar_width = 0.8
bar_width = 0.12
offset = (len(representation_LC_data.sheet_names) - 1) * bar_width / 2

# Plotting
fig, ax1 = plt.subplots(figsize=(9, 8))

# Setting positions for each landcover type in reversed order
y_positions_reversed = range(len(russian_labels_reversed))

# Plot total pixels for each landcover type in reversed order
ax1.barh(y_positions_reversed, total_pixels_reversed, label='Общее число пикселей', color='gray', edgecolor=None, height=gray_bar_width, alpha=0.5)
colors = wesanderson.film_palette('The French Dispatch')
colors_br = wesanderson.film_palette('The Grand Budapest Hotel')
# Plot tower pixels from each sheet in reversed order
for i, (sheet, pixels) in enumerate(tower_pixels_reversed.items()):
    label = translated_legend.get(sheet, sheet)  # Get the translation if available, else use the original key
    color_index = i % len(colors)  # Cycle through colors if there are more sheets than colors
    ax1.barh([pos + bar_width * i - offset for pos in y_positions_reversed], pixels, label=label, edgecolor='black', height=bar_width, color=colors[color_index])

# Update the y-axis labels with the reversed Russian translations
ax1.set_yticks(y_positions_reversed)
ax1.set_yticklabels(russian_labels_reversed)

ax1.set_xlim(0, 10.2*1e6)
ax1.set_xlabel('Число пикселей')
# ax1.set_title('Репрезентативность разных типов ландшафтов по Д.Г. Щепащенко')

# Adjust the legend order
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(reversed(handles), reversed(labels), loc='lower right', bbox_to_anchor=(1, 0.5))

# Adding the second axis for the landcover counts with adjusted range and synchronized gridlines
ax2 = ax1.twiny()
ax2.set_xlim(0, 5.1)
# Scatter plot for landcover counts should also be reversed
landcover_counts_reversed = landcover_counts[::-1]
ax2.scatter(landcover_counts_reversed, y_positions_reversed, color = colors_br[1], marker='o')
ax2.set_xlabel('Число установленных вышек', color = colors_br[1])
ax2.tick_params(axis='x', labelcolor = colors_br[1])

# Synchronize gridlines
ax1.grid(True, which='both', axis='x')
ax2.grid(False)

plt.tight_layout()
plt.show()


