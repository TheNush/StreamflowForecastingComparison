from math import sqrt
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def adjust(df):
	length = (len(df)//32) * 32
	df = df[:length]
	return df

def normalize(x):
	return((x-np.min(x))/(np.max(x) - np.min(x)))

discharges = pd.read_csv("C:/Users/TheNush07/Desktop/Work/Projects/StreamFlow Forecasting/Datasets/Navalgund.csv", header=0)
discharges = adjust(discharges)
print(len(discharges))

print(len(discharges))
 
for j in range(1,6):
	for i in range(2,7):
		# i = ""
		details = pd.read_csv("C:/Users/TheNush07/Desktop/Work/Projects/StreamFlow Forecasting/Datasets/Decomps/Navalgund/L{}/details-db{}.csv".format(j, i), header=0)
		details = adjust(details)
		approx = pd.read_csv("C:/Users/TheNush07/Desktop/Work/Projects/StreamFlow Forecasting/Datasets/Decomps/Navalgund/L{}/approxs-db{}.csv".format(j, i), header=0)
		approx = adjust(approx)
		y_train = (discharges.iloc[1:, 0])
		x_train = normalize((pd.concat([approx, details], axis=1)[:-1]))
		print(x_train.shape)
		x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, shuffle=False)
		y_val = y_val.reset_index(drop=True)
		model = svm.SVR(kernel='rbf', C=8000, epsilon=1, gamma=3, verbose=1)	# Navalgund
		# model = svm.SVR(kernel='sigmoid', C=800, epsilon=0.01, gamma=0.1, verbose=1)	# Khanapur
		# model = svm.SVR(kernel='sigmoid', C=800, epsilon=0.01, gamma=0.1, verbose=1) # Cholachguda
		model.fit((x_train), (y_train))
		preds_test = model.predict(x_val)
		preds_train = model.predict(x_train)

		r2_train = r2_score(y_train, preds_train)
		print("Train R2 :  ", r2_train)
		r2_test = r2_score(y_val, preds_test)
		print("Test R2 :   ", r2_test)
		mse_train = mean_squared_error(y_train, preds_train)
		# print("Train RMSE: ", sqrt(mse_train))
		mse_test = mean_squared_error(y_val, preds_test)
		# print("Test RMSE:  ", sqrt(mse_test))
		# print("Train NRMSE:", (sqrt(mse_train)/(np.std(y_train))))
		# print("Test NRMSE: ", (sqrt(mse_test)/(np.std(y_val))))
		
		fig, axs = plt.subplots(1)
		axs.plot(y_val, label="Actual Discharge")
		axs.plot(preds_test, label='Forecasted Discharge')
		axs.set_title('db{} Level {}'.format(i,j))
		axs.set_xlabel("TIME(Day)")
		axs.set_ylabel("Discharge(m^3/s)")
		axs.legend(('Actual Discharge', 'Forecast Discharge'), fontsize=12)
		# fig.savefig('C:/Users/TheNush07/Desktop/Work/Projects/StreamFlow Forecasting/Results/Cholachguda/db/Line/L{}-db{}_2.png'.format(j, i))

		fig, axs = plt.subplots(1)
		axs.scatter(y_val, preds_test, label='Forecasted v/s True')
		b, m = np.polynomial.polynomial.polyfit(y_val, preds_test, 1)
		print("Slope: ", m)
		print("Intercept: ", b)
		axs.plot(y_val, m * y_val + b, color='black', label='y = {0:.2f}x + {0:.2f}'.format(m, b))
		axs.set_title('db{} Level {}'.format(i,j))
		axs.set_yticks([0, 100, 200, 300, 400, 500, 600, 700], minor=False)
		axs.set_ylabel(r'Forecasted Discharge(m^3/s)')
		axs.set_xlabel(r'Actual Discharge(m^3/s)')
		axs.legend(('y = %0.2fx + %0.2f \nR Squared Score: %0.2f' % (m, b, r2_test), 'Forecasted v/s Actual'), fontsize=12)
		# axs.text(0.2, 0.95, r'Regression Line: y = {0:.2f}x + {0:.2f}/nR Squared Score: {0:.2f}'.format(m, b, r2_test), fontsize=10,ha='center' , va='top', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}, transform=axs.transAxes)
		# axs.legend(('Forecasted v/s Actual', 'Regression Line: y = {0:.2f}x + {0:.2f}', 'R Squared Score: {0:.2f}'.format(m, b, r2_test),), 'best')
		# fig.savefig('C:/Users/TheNush07/Desktop/Work/Projects/StreamFlow Forecasting/Results/Cholachguda/db/Scatter/L{}-db{}_2.png'.format(j, i))
		# plt.close('all')
		plt.show()
		# print("/n")
		print("Completed - L{} db{}".format(j, i))
