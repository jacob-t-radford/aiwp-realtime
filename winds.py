import sys
import json
import subprocess as sp
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool
import logging
import geojsoncontour
import geojson
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units
from functools import partial

def shift_lons(lon,model):
    if model in ['fcnv2','pw','gc','gfs','au']:
        return lon - 360 if lon > 180 else lon
    else:
        return lon

def run_processor(step):
    stepstr = str(step*6).zfill(3)
    for lev in range(0,13):
        level = str(data['level'][lev].values)
        features = []
        for i in range(latitudes.shape[0]):
            for j in range(longitudes.shape[0]):
                lat = latitudes[i]
                lon = shift_lons(longitudes[j],model)
                feature = geojson.Feature(
                        geometry=geojson.Point((float(lon), float(lat))),
                        properties={
                                "direction": float(direction[step,lev,i,j]),
                                "magnitude": float(magnitude[step,lev,i,j])
                        }
                )
                features.append(feature)
        feature_collection = geojson.FeatureCollection(features)
        outfile = f'{outdir}/{model}_{init_cond}_windbarbs{level}_{stepstr}.geojson.simple'
        with open(outfile, 'w') as f:
            geojson.dump(feature_collection,f)

    features = []
    for i in range(latitudes.shape[0]):
        for j in range(longitudes.shape[0]):
            lat = latitudes[i]
            lon = shift_lons(longitudes[j],model)
            feature = geojson.Feature(
                    geometry=geojson.Point((float(lon), float(lat))),
                    properties={
                            "direction": float(direction10[step,i,j]),
                            "magnitude": float(magnitude10[step,i,j])
                    }
            )
            features.append(feature)
    feature_collection = geojson.FeatureCollection(features)
    outfile = f'{outdir}/{model}_{init_cond}_windbarbs10_{stepstr}.geojson.simple'
    with open(outfile, 'w') as f:
        geojson.dump(feature_collection,f)

if __name__ == "__main__":
    year, month, day, init, model, init_cond, model_file, geojson_path = sys.argv[1:9]
    outdir = Path(f"{geojson_path}/{year}{month}{day}_{init}/{model}_{init_cond}")
    outdir.mkdir(parents=True, exist_ok=True)

    data = xr.open_dataset(model_file)
    uwinds = data['u'][:,:,::10,::10]
    vwinds = data['v'][:,:,::10,::10]
    uwinds10 = data['u10'][:,::10,::10]
    vwinds10 = data['v10'][:,::10,::10]

    longitudes = data['longitude'].values[::10]
    latitudes = data['latitude'].values[::10]
    
    magnitude = np.sqrt(uwinds**2 + vwinds**2)*1.94384
    direction = (np.arctan2(vwinds, uwinds) * 180 / np.pi + 360) % 360
    magnitude10 = np.sqrt(uwinds10**2 + vwinds10**2)*1.94384
    direction10 = (np.arctan2(vwinds10, uwinds10) * 180 / np.pi + 360) % 360

    magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf = 0.0)
    direction = np.nan_to_num(direction, nan=0.0, posinf=0.0, neginf = 0.0)
    magnitude10 = np.nan_to_num(magnitude10, nan=0.0, posinf=0.0, neginf=0.0)
    direction10 = np.nan_to_num(direction10, nan=0.0, posinf=0.0, neginf=0.0)
    
    with Pool(processes=14) as pool:
        pool.map(run_processor, range(0,41))
