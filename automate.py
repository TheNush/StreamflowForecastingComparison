import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Prepare data
runoff = pd.read_csv('C:/Users/TheNush07/Desktop/Work/Projects/StreamFlow Forecasting/Datasets/Cholachguda.csv', header=0)
# Remove last 7 points so as to perform SWT
length = (len(runoff)//32) * 32
runoff = np.array(runoff[:length])
# coeffs = np.array(pywt.swt(runoff, 'db2', level=3, axis=0))

wavelets = ['db', 'haar', 'coif', 'sym']

wavelets_dict = {}
wavelets_dict[wavelets[0]] = [2,8]
wavelets_dict[wavelets[1]] = [1,2]
wavelets_dict[wavelets[2]] = [1,6]
wavelets_dict[wavelets[3]] = [2,8]

# print(wavelets_dict[wavelets[0]])

def decompose(wavelet, mems):
	for j in range(mems[0],mems[1]):
		if wavelet == 'haar':
			wlt = wavelet
		else:
			wlt = wavelet + str(j)
		# print(wavelet)
		for i in range(1,6):
			coeffs = np.array(pywt.swt(runoff, wlt, level=i, axis=0))
			approxs = pd.DataFrame({'Discharges':coeffs[i-1, 0, :, 0]})
			details = pd.DataFrame(coeffs[:i, 1, :, 0])
			details = details.transpose()
			# print(details.shape)
			print(wlt)
			approxs.to_csv('C:/Users/TheNush07/Desktop/Work/Projects/StreamFlow Forecasting/Datasets/Decomps/Cholachguda/L{}/approxs-{}_2.csv'.format(i, wlt), index=False)
			details.to_csv('C:/Users/TheNush07/Desktop/Work/Projects/StreamFlow Forecasting/Datasets/Decomps/Cholachguda/L{}/details-{}_2.csv'.format(i, wlt), index=False)

for i in range(0, 4):
	decompose(wavelets[i], wavelets_dict[wavelets[i]])
	print('Decomposition of {}'.format(wavelets[i]))
