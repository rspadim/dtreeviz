import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from dtreeviz.trees import *

df_cars = pd.read_csv("data/cars.csv")
X = df_cars.drop('MPG', axis=1)
y = df_cars['MPG']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


max_depth = 2
t = rtreeviz(X_train.WGT, y_train,
             max_depth=max_depth,
             feature_name='Vehicle Weight',
             target_name='MPG',
             # x_test=X_test.WGT, y_test=y_test,
             fontsize=14)
plt.savefig(f"/tmp/dectree-depth-{max_depth}.svg", bbox_inches=0, pad_inches=0)
plt.show()
