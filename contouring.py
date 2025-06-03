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
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------
# Constants and configuration
# ---------------------------------------------
PRECIP_COLORS = [
    "#80ff00", "#00cd00", "#008b00", "#104e8b", "#1e8fff", "#00b3ee",
    "#00eeee", "#8968cd", "#902cee", "#8b008b"
]
TOTAL_PRECIP_COLORS = [
    "#80ff00", "#00cd00", "#008b00", "#104e8b", "#1e8fff", "#00b3ee",
    "#00eeee", "#8968cd", "#902cee", "#8b008b", "#8b0000", "#cd0000",
    "#ee4000", "#ff7f00", "#cd8500", "#ffd900", "#ffff00", "#ffaeb9"
]

LEVELS = [1000,925,850,700,500,300,250,200]

myvars = {
    "windmag10": {
        "type": "sfc",
        "sfc": {
            "contourlevels": [0.01] + list(np.linspace(1.50, 30, 20)),
            "colormap": "viridis",
            "units": "m/s",
            "extend": "max",
            "filter": 1
        }
    },
    "t2": {
        "type": "sfc",
        "sfc": {
            "contourlevels": list(np.linspace(-50, 50, 101)),
            "colormap": "viridis",
            "units": "°C",
            "extend": "both",
            "filter": 1
        }
    },
    "sp": {
        "type": "sfc",
        "sfc": {
            "contourlevels": list(np.linspace(750, 1050, 101)),
            "colormap": "viridis",
            "units": "hPa",
            "extend": "both",
            "filter": 1
        }
    },
    "msl": {
        "type": "sfc",
        "sfc": {
            "contourlevels": list(np.linspace(950, 1050, 101)),
            "colormap": "viridis",
            "units": "hPa",
            "extend": "both",
            "filter": 1
        }
    },
    "tcwv": {
        "type": "sfc",
        "sfc": {
            "contourlevels": [0.01] + list(np.linspace(1, 70, 70)),
            "colormap": "viridis",
            "units": "kg/m^2",
            "extend": "max",
            "filter": 1
        }
    },
    "apcp": {
        "type": "sfc",
        "sfc": {
            "contourlevels": [0.01, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00],
            "colormap": "viridis",
            "units": "in",
            "extend": "max",
            "filter": 1
        }
    },
    "apcptotal": {
        "type": "sfc",
        "sfc": {
            "contourlevels": [0.01, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00, 10.00, 15.00, 20.00],
            "colormap": "viridis",
            "units": "in",
            "extend": "max",
            "filter": 1
        }
    },
    "windmag": {
        "type": "pl",
        **{level: {
            "contourlevels": [0.01] + list(np.linspace(start, stop, 20)),
            "colormap": "viridis",
            "units": "m/s",
            "extend": "max",
            "filter": 1
        } for level, (start, stop) in zip(
            [1000, 925, 850, 700, 500, 300, 250, 200],
            [(1.50, 30), (1.50, 30), (1.50, 30), (2.00, 40), (2.50, 50), (3.50, 70), (3.50, 70), (3.50, 70)]
        )}
    },
    "z": {
        "type": "pl",
        **{level: {
            "contourlevels": list(np.linspace(*bounds, 21)),
            "colormap": "viridis",
            "units": "m",
            "extend": "both",
            "filter": 1
        } for level, bounds in zip(
            [1000, 925, 850, 700, 500, 300, 250, 200],
            [(-400, 400), (200, 1000), (900, 1700), (2300, 3300), (4800, 6000), (8000, 10000), (9250, 11250), (10500, 13500)]
        )}
    },
    "t": {
        "type": "pl",
        **{level: {
            "contourlevels": list(np.linspace(*bounds, 21)),
            "colormap": "viridis",
            "units": "°C",
            "extend": "both",
            "filter": 1
        } for level, bounds in zip(
            [1000, 925, 850, 700, 500, 300, 250, 200],
            [(-40, 40), (-40, 40), (-30, 30), (-40, 20), (-50, 0), (-70, -20), (-70, -30), (-70, -40)]
        )}
    },
    "r": {
        "type": "pl",
        **{level: {
            "contourlevels": [0.01] + list(np.linspace(5, 100, 20)),
            "colormap": "viridis",
            "units": "m/s",
            "extend": "max",
            "filter": 3
        } for level in [1000, 925, 850, 700, 500, 300, 250, 200]}
    },
    "w": {
        "type": "pl",
        **{level: {
            "contourlevels": list(np.linspace(-1, 1, 21)),
            "colormap": "bwr",
            "units": "m/s",
            "extend": "both",
            "filter": 1
        } for level in [1000, 925, 850, 700, 500, 300, 250, 200]}
    }
}


def prepare_lons_lats(model):
    if model in ["ifs", "aifs"]:
        lons = np.arange(-180, 180, 0.25)
        lons_extended = np.arange(-180, 180.25, 0.25)
    else:
        lons = np.arange(0, 360, 0.25)
        lons_extended = np.arange(0, 360.25, 0.25)
    lats = np.arange(-90, 90.25, 0.25)
    return lons, lons_extended, lats

def calculate_derived_variables(data,model):
    windmag10 = np.sqrt(data['u10']**2 + data['v10']**2)
    data['windmag10'] = (('time', 'latitude', 'longitude'), windmag10.data)

    windmag = np.sqrt(data['u']**2 + data['v']**2)
    data['windmag'] = (('time', 'level', 'latitude', 'longitude'), windmag.data)

    if model in ['gc','pw','au','aifs']:

        spfh = data['q']
        pres_vals = data['level'].values
        pres = np.zeros((41,13,721,1440))
        for i, lev in enumerate(pres_vals):
            pres[:, i, :, :] = lev
        temp_C = data['t'].values - 273.15
        rh = relative_humidity_from_specific_humidity(
            pres * units.hPa, temp_C * units.degC, spfh
        ).to('percent').magnitude.astype('float32')
        data['r'] = (('time', 'level', 'latitude', 'longitude'), rh)

    if model in ["gc","gfs","ifs","aifs"]:

        apcptotal = np.cumsum(data['apcp'].values, axis=0)
        data['apcptotal'] = (('time', 'latitude', 'longitude'), apcptotal)

    return data

def plot(varlev,variabledata_extended,contourlevels,cmap,extend):
    figure = plt.figure()
    ax = figure.add_subplot(111)

    if varlev not in ['apcp','apcptotal']:
        contours = ax.contourf(lons_extended,lats,variabledata_extended,levels=contourlevels,cmap=cmap,extend=extend)
    elif varlev=='apcp':
        contours = ax.contourf(lons_extended,lats,variabledata_extended,levels=contourlevels,colors=PRECIP_COLORS,extend=extend)
    elif varlev=='apcptotal':
        contours = ax.contourf(lons_extended,lats,variabledata_extended,levels=contourlevels,colors=TOTAL_PRECIP_COLORS,extend=extend)
    plt.close()
        
    return contours

def geojson(contours,outfile):
    geojsoncontour.contourf_to_geojson(contourf=contours,geojson_filepath=outfile,ndigits=2)

def empty_geojson(filename):
    empty_geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    with open(filename, "w") as f:
        json.dump(empty_geojson, f)

def simplify(geojsonfile):
    command = f"{mapshaper} {geojsonfile} -simplify dp 30% -o {geojsonfile}.simple"
    try:
        sp.run(command,shell=True,check=True)
    except:
        empty_geojson(f"{geojsonfile}.simple")

def run_processor(step):
    for var in myvars.keys():
        if var in data.data_vars:
            if myvars[var]["type"]=="pl":
                for lev in LEVELS:
                    contourlevels = myvars[var][lev]['contourlevels']
                    cmap = myvars[var][lev]['colormap']
                    units = myvars[var][lev]['units']
                    extend = myvars[var][lev]['extend']
                    sigma = myvars[var][lev]['filter']
                    varlev = var + str(lev)
                    fhour = str(step*6).zfill(3)

                    levidx = np.where(data.level.values == lev)[0][0] 
                    variabledata = data[var][step,levidx,:,:]
                    
                    if var=='z':
                        variabledata = variabledata/9.80665
                    elif var=='t':
                        variabledata = variabledata - 273.15

                    variabledata = gaussian_filter(variabledata,sigma=sigma,mode='wrap')
                    variabledata = np.flipud(variabledata)
                    
                    variabledata_extended = np.zeros((721,1441))
                    variabledata_extended[:,:-1] = variabledata
                    variabledata_extended[:,-1] = variabledata[:,0]

                    contours = plot(varlev,variabledata_extended,contourlevels,cmap,extend)

                    outfile = f"{outdir}/{model}_{init_cond}_{varlev}_{fhour}.geojson"
                    geojson(contours,outfile)
                    simplify(outfile)

            elif myvars[var]["type"]=="sfc":
                contourlevels = myvars[var]['sfc']['contourlevels']
                cmap = myvars[var]['sfc']['colormap']
                units = myvars[var]['sfc']['units']
                extend = myvars[var]['sfc']['extend']
                sigma = myvars[var]['sfc']['filter']
                varlev = var
                fhour = str(step*6).zfill(3)

                variabledata = data[var][step,:,:]

                if var=='t2':
                    variabledata = variabledata - 273.15
                elif var=='apcp' or var=='apcptotal':
                    variabledata = variabledata * 39.3701
                elif var=='msl':
                    variabledata = variabledata/100.
            
                variabledata = gaussian_filter(variabledata,sigma=sigma,mode='wrap')
                variabledata = np.flipud(variabledata)

                variabledata_extended = np.zeros((721,1441))
                variabledata_extended[:,:-1] = variabledata
                variabledata_extended[:,-1] = variabledata[:,0]

                contours = plot(varlev,variabledata_extended,contourlevels,cmap,extend)

                outfile = f"{outdir}/{model}_{init_cond}_{varlev}_{fhour}.geojson"
                geojson(contours,outfile)
                simplify(outfile)

        else:
            if myvars[var]["type"]=="pl":
                for lev in LEVELS:
                    varlev = var + str(lev)
                    outfile = f"{outdir}/{model}_{init_cond}_{varlev}_{fhour}"
                    empty_geojson(f"{outfile}.simple")
            elif myvars[var]["type"]=="sfc":
                varlev = var
                outfile = f"{outdir}/{model}_{init_cond}_{varlev}_{fhour}"
                empty_geojson(f"{outfile}.simple")


if __name__ == "__main__":
    year, month, day, init, model, init_cond, model_file, geojson_path = sys.argv[1:9]
    outdir = Path(f"{geojson_path}/{year}{month}{day}_{init}/{model}_{init_cond}")
    outdir.mkdir(parents=True, exist_ok=True)
    mapshaper = f"node_modules/.bin/mapshaper"

    lons, lons_extended, lats = prepare_lons_lats(model)
    
    data = xr.open_dataset(model_file)
    data = calculate_derived_variables(data, model)
    data.load()

    with Pool(processes=14) as pool:
        pool.map(run_processor, range(0,41))
