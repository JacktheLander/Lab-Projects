import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer

wellbeing = pd.read_csv('city_wellbeing.csv')
X = wellbeing[['NAI']].values
y = wellbeing[['WBI_Physical']].values

# Transofrm input feature into matrix needed for the linear spline model
spline = SplineTransformer(degree=1, n_knots=3, knots='quantile')
splinefit = spline.fit(X)
X_1 = spline.fit_transform(X)

LSRModel = LinearRegression()
LSRModel.fit(X_1,y)

# Linear spline model weights
print(LSRModel.intercept_)
print(LSRModel.coef_)

yPred = LSRModel.predict(X_1)
yPred[0:5]

# Plot the data, linear model, and spline model
x_vals = np.linspace(min(wellbeing['NAI']), max(wellbeing['NAI']), 100)
x_valsT = splinefit.transform(x_vals.reshape(-1,1))
y_vals = LSRModel.predict(x_valsT)

p = sns.regplot(data=wellbeing, x="NAI", y="WBI_Physical", ci=False, line_kws={'ls':'--', "color": "black"})
plt.plot(x_vals, y_vals, color='#ff7f0e', linewidth=3)
p.set_xlabel('Natural amenities index', fontsize=14)
p.set_ylabel('Physical well-being index', fontsize=14)
plt.show()

LSRModel.predict(splinefit.transform([[-0.76]]))

## Spline with 5 Knots
# Transofrm input feature into matrix needed for the spline regression model
spline = SplineTransformer(degree=1, n_knots=5, knots='quantile')
splinefit = spline.fit(X)
X_2 = spline.fit_transform(X)

LSRModel = LinearRegression()
LSRModel.fit(X_2,y)

x_vals = np.linspace(min(wellbeing['NAI']), max(wellbeing['NAI']), 100)
x_valsT = splinefit.transform(x_vals.reshape(-1,1))
y_vals = LSRModel.predict(x_valsT)

p = sns.regplot(data=wellbeing, x="NAI", y="WBI_Physical", ci=False, line_kws={'ls':'--', "color": "black"})
plt.plot(x_vals, y_vals, color='#3ca02c', linewidth=3)
p.set_xlabel('Natural amenities index', fontsize=14)
p.set_ylabel('Physical well-being index', fontsize=14)
plt.show()
