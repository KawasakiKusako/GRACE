import netCDF4
import random

# 指定.nc文件的路径
file_path = './GEO.fp.asm.tavg3_2d_chm_Nx.nc4'

# 打开.nc文件
nc = netCDF4.Dataset(file_path, mode='r')

# 若变量是地理坐标相关的，获取经纬度信息
latitudes = nc.variables['lat'][:]
longitudes = nc.variables['lon'][:]

length = len(latitudes) * len(longitudes)
data_idxs = [i for i in range(length)]

train_ratio = 0.75
random.shuffle(data_idxs)  # 先打乱列表以确保随机性
train_set = data_idxs[:int(train_ratio * length)]
test_set = list(set(data_idxs) - set(train_set))

with open("./data/train.txt", "w") as file:
    for item in train_set:
        file.write(str(item) + "\n")

with open("./data/test.txt", "w") as file:
    for item in test_set:
        file.write(str(item) + "\n")

print('done')

