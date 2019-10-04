import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Prepare data
runoff = pd.read_csv('C:/Users/TheNush07/Desktop/Work/Projects/StreamFlow Forecasting/Datasets/Navalgund.csv', header=0)
# Remove last 7 points so as to perform SWT
runoff = np.array(runoff[:-7])
# coeffs = np.array(pywt.swt(runoff, 'db2', level=3, axis=0))

wavelets = ['db', 'haar', 'coif', 'sym']

def decompose(wavelet):
	for j in range(2,8):
		wlt = wavelet + str(j)
		# print(wavelet)
		for i in range(1,6):
			coeffs = np.array(pywt.swt(runoff, wlt, level=i, axis=0))
			approxs = pd.DataFrame({'Discharges':coeffs[i-1, 0, :, 0]})
			details = pd.DataFrame(coeffs[:i, 1, :, 0])
			details = details.transpose()
			# print(details.shape)
			print(wlt)

			approxs.to_csv('C:/Users/TheNush07/Desktop/Work/Projects/StreamFlow Forecasting/Datasets/Decomps/Navalgund/L{}/approxs-{}.csv'.format(i, wlt), index=False)
			details.to_csv('C:/Users/TheNush07/Desktop/Work/Projects/StreamFlow Forecasting/Datasets/Decomps/Navalgund/L{}/details-{}.csv'.format(i, wlt), index=False)

# run decompose function for different wavelet families
decompose(wavelets[3])
