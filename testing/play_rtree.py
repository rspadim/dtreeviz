import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df_cars = pd.read_csv("data/cars.csv")
X = df_cars.drop('MPG', axis=1)
y = df_cars['MPG']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from dtreeviz.trees import rtreeviz
t = rtreeviz(X_train.WGT, y_train, max_depth=2, feature_name='Vehicle Weight', target_name='MPG')
plt.text(3090,40,f"Decision tree model, $R^2$={t.score(X_test[['WGT']],y_test):.3f}", fontsize=14)
plt.show()
