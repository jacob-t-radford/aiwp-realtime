# ** Don't modify any of this **

import numpy as np
import xarray as xr
import pygrib as pg
import math
import datetime
import sys
from netCDF4 import Dataset as DS

# Define the main function to convert GRIB files to NetCDF
def grib2nc(infile, initconditions, model, date, time):

    # Mapping of variable names and their descriptions/units
    varmap = {
        "u10": ['10 metre U wind component', 'm s-1'],
        "v10": ['10 metre V wind component', 'm s-1'],
        "t2": ['2 metre temperature', 'K'],
        "msl": ['Pressure reduced to MSL', 'Pa'],
        "u100": ['100 metre U wind component', 'm s-1'],
        "v100": ['100 metre V wind component', 'm s-1'],
        "sp": ['Surface pressure', 'Pa'],
        "tcwv": ['Precipitable water', 'kg m-2'],
        "apcp": ['6-hr accumulated precipitation', 'm'],
        "u": ['U component of wind', 'm s-1'],
        "v": ['V component of wind', 'm s-1'],
        "t": ['Temperature', 'K'],
        "z": ['Geopotential', 'm2 s-2'],
        "r": ['Relative humidity', '%'],
        "q": ['Specific humidity', 'kg kg-1'],
        "w": ['Vertical velocity', 'Pa s-1']
    }

    # Mapping from ECMWF to GFS names
    ec2gfsmap = {
        "u": "u",
        "v": "v",
        "w": "w",
        "z": "z",
        "q": "q",
        "r": "r",
        "t": "t",
        "10u": "u10",
        "10v": "v10",
        "100u": "u100",
        "100v": "v100",
        "2t": "t2",
        "msl": "msl",
        "sp": "sp",
        "tcwv": "tcwv",
        "tp": "apcp"
    }

    # Initialize lists to store variables and levels
    unique_pl_vars = []
    unique_sfc_vars = []
    levels = []

    # Open the GRIB file and extract grid shape and lat/lon coordinates
    with pg.open(infile) as grib:
        y_shape, x_shape = grib[1].values.shape
        lats, lons = grib[1].latlons()
        lats = lats[:, 0]
        lons = lons[0, :]

        # Identify pressure level and surface variables
        for grb in grib:
            if grb.levelType == "pl":
                if grb.level not in levels:
                    levels.append(grb.level)
                if grb.shortName not in unique_pl_vars:
                    unique_pl_vars.append(grb.shortName)
            else:
                if grb.shortName not in unique_sfc_vars:
                    if grb.shortName in ['lsm','z']:
                        continue
                    unique_sfc_vars.append(grb.shortName)
        levels.sort(reverse=True)
        levelmap = {level: c for c, level in enumerate(levels)}

        # Create NetCDF file and define dimensions
        f = DS(f"{infile}.nc", 'w', format='NETCDF4')
        f.createDimension('time', 41)
        f.createDimension('level', 13)
        f.createDimension('longitude', 1440)
        f.createDimension('latitude', 721)

        # Generate time values based on initialization date and time
        initdt = datetime.datetime.strptime(f"{date}{time}", "%Y%m%d%H00")
        times = np.array([
            int((initdt + datetime.timedelta(hours=int(i))).timestamp())
            for i in np.arange(0, 246, 6)
        ])

        # Create coordinate variables in NetCDF
        create_variable_nochunk(f, 'time', ('time',), times, {
            'long_name': 'Date and Time', 'units': 'seconds since 1970-1-1',
            'calendar': 'standard'
        })
        create_variable_nochunk(f, 'longitude', ('longitude',), lons, {
            'long_name': 'Longitude', 'units': 'degree'
        })
        create_variable_nochunk(f, 'latitude', ('latitude',), lats, {
            'long_name': 'Latitude', 'units': 'degree'
        })
        create_variable_nochunk(f, 'level', ('level',), np.array(levels), {
            'long_name': 'Isobaric surfaces', 'units': 'hPa'
        })

        # Define variables in NetCDF for both pressure level and surface variables
        for variable in unique_pl_vars + unique_sfc_vars:
            print(variable)
            dims = ('time', 'level', 'latitude', 'longitude') if variable in unique_pl_vars else ('time', 'latitude', 'longitude')
            chunksizes = (1, 1, y_shape, x_shape) if 'level' in dims else (1, y_shape, x_shape)
            gfsequivalent = ec2gfsmap[variable]
            create_variable(
                f, gfsequivalent, dims, None, {
                    'long_name': varmap[gfsequivalent][0],
                    'units': varmap[gfsequivalent][1]
                },
                chunksizes
            )

        # Populate NetCDF variables with data from GRIB
        grib.seek(0)
        for grb in grib:
            shortName = grb.shortName
            timestep = int(grb.step / 6)
            level = grb.level
            levelType = grb.levelType
            if (shortName == 'z' and levelType == 'sfc') or shortName not in ec2gfsmap.keys():
                continue
            gfsequivalent = ec2gfsmap[shortName]
            vals = grb.values

            if levelType == 'pl':
                levelind = levelmap[level]
                f.variables[gfsequivalent][timestep, levelind, :, :] = vals
            elif levelType == 'sfc':
                f.variables[gfsequivalent][timestep, :, :] = vals

        # Add global attributes to the NetCDF file
        f.Conventions = 'CF-1.8'
        f.version = '3_2025-02-20'
        f.model_name = model
        f.initialization_model = initconditions
        f.initialization_time = initdt.strftime('%Y-%m-%dT%H:%M:%S')
        f.first_forecast_hour = "0"
        f.last_forecast_hour = f"240"
        f.forecast_hour_step = f"6"
        f.creation_time = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        f.close()

# Functions to create variables in the NetCDF file
def create_variable(f, name, dimensions, data, attrs, chunksizes):
    dtype = 'i4' if name in ['time', 'level'] else 'f4'
    var = f.createVariable(name, dtype, dimensions, chunksizes=chunksizes)
#    var = f.createVariable(name, dtype, dimensions, compression='zlib', complevel=4, chunksizes=chunksizes)
    if data is not None:
        var[:] = data
    for attr_name, attr_value in attrs.items():
        var.setncattr(attr_name, attr_value)

def create_variable_nochunk(f, name, dimensions, data, attrs):
    dtype = 'i4' if name in ['time', 'level'] else 'f4'
    var = f.createVariable(name, dtype, dimensions)
 #   var = f.createVariable(name, dtype, dimensions, compression='zlib', complevel=4)
    var[:] = data
    for attr_name, attr_value in attrs.items():
        var.setncattr(attr_name, attr_value)

infile = sys.argv[1]
initconditions = sys.argv[2]
model = sys.argv[3]
date = sys.argv[4]
time = sys.argv[5]

#Call the function
grib2nc(infile, initconditions, model, date, time)
