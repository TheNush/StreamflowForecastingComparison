# Streamflow Forecasting Comparison

Forecasting streamflow based on past data is very crucial in flood control and helps in water resources management. The code in this repo 
aims at comparing the performance of single models(currently SVR) in forecasting streamflow for different wavelets used for data
decomposition. 

# File Functionality
*automate.py*

  This script performs a stationary wavelet transform on the streamflow data based on the input levels. In this case, each member of each 
  wavelet family is used to decompose the streamflow data for levels 1 to 5. The resulting details and approximations are stored as .csv     files which are further used as inputs for the Support Vector Regressor. 
  
*mat.py*

  This script uses the .csv files prepared by *automate.py* file as inputs to the SVR model. The forecasted values are compared with
  expected values and accuracies are reported. The required graphs are also plotted. 

