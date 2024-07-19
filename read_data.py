import netCDF4
import numpy as np

# 指定.nc文件的路径
file_path = './GEO.fp.asm.tavg3_2d_chm_Nx.nc4'

# 打开.nc文件
nc = netCDF4.Dataset(file_path, mode='r')

# 若变量是地理坐标相关的，获取经纬度信息
latitudes = nc.variables['lat'][:]
longitudes = nc.variables['lon'][:]

# 获取CO2信息
co2cl = np.squeeze(nc.variables['CO2CL'][:])
co2em = np.squeeze(nc.variables['CO2EM'][:])
co2sc = np.squeeze(nc.variables['CO2SC'][:])

length = len(latitudes) * len(longitudes)
geo_data = []
co2_data = np.zeros((length, 3))
for i in range(len(latitudes)):
    for j in range(len(longitudes)):
        geo_data.append((latitudes[i], longitudes[j]))
        co2_data[i * len(latitudes) + j, :] = np.array([co2cl[i, j], co2em[i, j], co2sc[i, j]])

co2 = np.concatenate((co2cl, co2em, co2sc), axis=-1)

print('done')
