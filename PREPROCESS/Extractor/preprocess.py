from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

datas = []

for data in os.listdir(sys.argv[1]):
    if ".dgl" in data:
        datas.append((os.path.abspath(os.path.join(sys.argv[1], data)), float(data.split("_")[-1].split(".dgl")[0])))

X = np.array([data[1] for data in datas]).reshape(-1,1)

robuster_scaler = preprocessing.RobustScaler(quantile_range=(25, 75))
X_robuster = robuster_scaler.fit_transform(X)

for idx in range(0, len(datas)):
    data = datas[idx]
    target_dir = os.path.join(sys.argv[2], (data[0][:data[0].rfind("_")+1] + str(float(X_robuster[idx])) + ".dgl").split("/")[-1])
    cmd = "ln -s {} {}".format(data[0], target_dir)
    os.system("ln -s {} {}".format(data[0], target_dir))
