# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a script to convert ROMS history file output to ADH Time Series
ASCII XY1 x-y series control cards

Author: Brandy Armstrong
Date of Revision: 05/28/2021
Affiliation: The University of Southern Mississippi
Modeling Ocean, Sediment, Engineering, Atmosphere (MOSEA)


"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

## USER INPUT REQUIRED
# Identify ADH XY, latitude and longitude, input locations
lat_in = [30.2492, 30.3]
lon_in = [-89.0919, -89.1]

# Identify start and end date, define time slice
# the initial time of each series must be less than or equal to the initial 
# time of the simulation, and final time of each series must be equal to or 
# greater than the final time of the simulation. 
# msb-COAWST output is hourly: start time must be on the hour or choose 
# the hour prior to actual start time, end_time must be on the hour or 
# choose the hour after run is to complete.
start_date = np.datetime64('2019-02-28T01:00:00.000000000') # initial time of the simulation
end_date = np.datetime64('2019-02-28T22:00:00.000000000')   # final time of the simulation  

# Identify name and location for output file
output_file="/home/brandy/Python/test_AdH.txt"

# Identify COAWST output files or url
nc_file='http://127.0.0.1:8081/thredds/dodsC/mgwork/model_runs/msbCOAWST/2019/concorde_ext_his_20190228.nc'
#nc_file='http://127.0.0.1:8081/thredds/dodsC/msbCOAWST/msbCOAWST_best.ncd'
#nc_fmrc='http://127.0.0.1:8081/thredds/dodsC/msbCOAWST/msbCOAWST_fmrc.ncd'
#nc_fmrc='http://127.0.0.1:8081/thredds/dodsC/home/mgWork/thredds_home/model_runs/msbCOAWST/msbCOAWST_Forecast_best.ncd'
#open dataset with xarray & turn on chunking to activate dask and parallelize read/write
ds = xr.open_dataset(nc_file, chunks={'ocean_time': 1})

#pick out some of the variables to include as coordinates
ds = ds.set_coords(['Cs_r', 'Cs_w', 'hc', 'h', 'Vtransform', 'lat_rho', 'lon_rho'])
# Extract longitude and latitude from NetCDF file
lat_rho=ds['lat_rho'].data
lon_rho=ds['lon_rho'].data
# select a subset of variables
variables = ['salt', 'temp', 'zeta','ubar', 'vbar']

# Extract time from NetCDF file, identify indices of time between start and end dates
# Identify start and end date, define time slice
start_ind = np.where(ds['ocean_time'].data == start_date)
end_ind = np.where(ds['ocean_time'].data == end_date)
time_ts=ds['ocean_time'].data-start_date
time_tst=time_ts[start_ind[0][0]:end_ind[0][0]]
time_tsh=time_tst/(time_ts[1]-time_ts[0])
# calculate vertical coordinates
if ds.Vtransform == 1:
    Zo_rho = ds.hc * (ds.s_rho - ds.Cs_r) + ds.Cs_r * ds.h
    z_rho = Zo_rho + ds.zeta * (1 + Zo_rho/ds.h)
elif ds.Vtransform == 2:
    Zo_rho = (ds.hc * ds.s_rho + ds.Cs_r * ds.h) / (ds.hc + ds.h)
    z_rho = ds.zeta + (ds.zeta + ds.h) * Zo_rho

ds.coords['z_rho'] = z_rho.transpose()   # needing transpose seems to be an xarray bug

# AdH information
# Series number will be assigned

# Identify the time value unit flag
# 0=seconds, 1=minutes, 2=hours, 3=days, 4=weeks
time_unit=2

# Calculate the number of points
no_pts=end_ind[0][0]-start_ind[0][0]


# Identify ADH variables
ubar_ts=[]
vbar_ts=[]
zeta_ts=[]
salt_ts=[]
temp_ts=[]
# integrate over s_rho
saltt=ds.salt.integrate("s_rho")
tempt=ds.temp.integrate("s_rho")
#time_series_ts=np.linspace(1,5*len(lat_in),num=5*len(lat_in))
# 2d variables, Find eta_rho and xi_rho for each input location
# Find the index of the grid point nearest a specific lat/lon.
xi_id=[] 
eta_id=[] 
time_series=0
for idx, val in enumerate(lat_in):
    abslat = np.abs(ds.lat_rho-lat_in[idx])
    abslon = np.abs(ds.lon_rho-lon_in[idx])
    c = np.maximum(abslon, abslat)
    ([xi_ids], [eta_ids]) = np.where(c == np.min(c))
    xi_id +=[xi_ids]
    eta_id +=[eta_ids]
    
    # plot to check location (make sure eta and xi are not switched)
    ds.ubar_eastward.isel(ocean_time=1).plot(x='lon_rho',y='lat_rho')
    plt.plot(ds.lon_rho[xi_id,eta_id],ds.lat_rho[xi_id,eta_id],'rx')
    plt.plot(lon_in,lat_in,'bx')
    
    # 2D variables to XY: ubar, vbar, and zeta
    # ubar, depth averaged u (east) velocity
    ubar_ts=ds.ubar_eastward.isel(ocean_time=slice(start_ind[0][0], end_ind[0][0]-start_ind[0][0]-1,1), eta_rho=eta_id[idx], xi_rho=xi_id[idx])
    # vbar, depth averaged v (north) velocity
    vbar_ts=ds.vbar_northward.isel(ocean_time=slice(start_ind[0][0], end_ind[0][0]-start_ind[0][0]-1,1), eta_rho=eta_id[idx], xi_rho=xi_id[idx])
    # zeta, sea surface height
    zeta_ts=ds.zeta.isel(ocean_time=slice(start_ind[0][0], end_ind[0][0]-start_ind[0][0]-1,1), eta_rho=eta_id[idx], xi_rho=xi_id[idx])
    
    # Depth integrated 3D variables: salt and temp
    # salt, depth averaged salinity
    salt_ts=saltt.isel(ocean_time=slice(start_ind[0][0], end_ind[0][0]-start_ind[0][0]-1,1), eta_rho=eta_id[idx], xi_rho=xi_id[idx])

    # temp, depth averaged temperature
    temp_ts=tempt.isel(ocean_time=slice(start_ind[0][0], end_ind[0][0]-start_ind[0][0]-1,1), eta_rho=eta_id[idx], xi_rho=xi_id[idx])
    

    # Write AdH Time Series XY1 data cards, ASCII
    # Create a DataFrame for each variable
  
    # write data to delimited text file output_file defined above
    file=open(output_file,"a")
    time_series=time_series+1
    L1="XY1\t"+str(time_series)+"\t"+str(no_pts)+"\t"+str(time_unit)+"\t0\n"
    file.write(L1)
    for a, am in zip(time_tsh,ubar_ts.values):
        file.write("{}\t{}\n".format(a,am))
    file.write("\n")
    time_series=time_series+1
    L1="XY1\t"+str(time_series)+"\t"+str(no_pts)+"\t"+str(time_unit)+"\t0\n"
    file.write(L1)
    for a, am in zip(time_tsh,vbar_ts.values):
        file.write("{}\t{}\n".format(a,am))
    file.write("\n")
    time_series=time_series+1
    L1="XY1\t"+str(time_series)+"\t"+str(no_pts)+"\t"+str(time_unit)+"\t0\n"
    file.write(L1)
    for a, am in zip(time_tsh,zeta_ts.values):
        file.write("{}\t{}\n".format(a,am))
    file.write("\n")
    time_series=time_series+1
    L1="XY1\t"+str(time_series)+"\t"+str(no_pts)+"\t"+str(time_unit)+"\t0\n"
    file.write(L1)
    for a, am in zip(time_tsh,salt_ts.values):
        file.write("{}\t{}\n".format(a,am))
    file.write("\n")
    time_series=time_series+1
    L1="XY1\t"+str(time_series)+"\t"+str(no_pts)+"\t"+str(time_unit)+"\t0\n"
    file.write(L1)
    for a, am in zip(time_tsh,temp_ts.values):
        file.write("{}\t{}\n".format(a,am))    
    file.write("\n")
    file.close()
    




