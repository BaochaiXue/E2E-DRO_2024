# End-to-End Distributionally Robust Optimization
#
# Prepared by:    Giorgio Costa (gc2958@columbia.edu)
# Last revision:  31-Oct-2021
#
####################################################################################################
## Import libraries
####################################################################################################
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import pandas_datareader as pdr

# Import E2E_DRO functions
my_path = "/Users/giorgio/Library/Mobile Documents/com~apple~CloudDocs/Documents/Google Drive/Research Projects/2021/E2E DRL/E2E-DRO"
import sys
sys.path.append(my_path)
from e2edro import e2edro as dro
from e2edro import e2enom as nom
from e2edro import LossFunctions as lf
from e2edro import RiskFunctions as rf

# Imoprt 'reload' to update E2E_DRO libraries while in development
from importlib import reload 
reload(dro)
reload(nom)

####################################################################################################
# Load data
####################################################################################################

#---------------------------------------------------------------------------------------------------
# Option 1: Generate synthetic data
#---------------------------------------------------------------------------------------------------
torch.manual_seed(1)
# Number of observations T, features m and outputs n
T, m, n = 112, 8, 15

# 'True' prediction bias and weights
a = torch.rand(n)
b = torch.randn(m,n)

# Syntehtic features
X = torch.randn(T, m)

# Synthetic outputs
Y_test = a + X @ b + 0.2*torch.randn(T,n)

# Convert them to Variable type for use with torch library
X, Y = Variable(X), Variable(Y_test)

T = 100
X, Y = X[0:T], Y[0:T]

#---------------------------------------------------------------------------------------------------
# Option 2: Load data from Kenneth French's data library 
# https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html 
#---------------------------------------------------------------------------------------------------
# Data frequency: type '_daily' for daily frequency, or '' for monthly frequency
freq = ''
start = '1990-01-01'
end = '2021-09-30'

# Download data 
asset_df = pdr.get_data_famafrench('10_Industry_Portfolios'+freq, start=start, end=end)[0]
factor_df = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3'+freq, start=start, end=end)[0]
rf_df = factor_df['RF']
factor_df = factor_df.drop(['RF'], axis=1)
mom_df = pdr.get_data_famafrench('F-F_Momentum_Factor'+freq, start=start, end=end)[0]
st_df = pdr.get_data_famafrench('F-F_ST_Reversal_Factor'+freq, start=start, end=end)[0]
lt_df = pdr.get_data_famafrench('F-F_LT_Reversal_Factor'+freq, start=start, end=end)[0]

# Concatenate factors as a pandas dataframe
factor_df = pd.concat([factor_df, mom_df, st_df, lt_df], axis=1)

# Convert pd.dataframes to torch.variables and trim the dataset
X = Variable(torch.tensor(factor_df.values / 100, dtype=torch.double))[0:400,:]
Y = Variable(torch.tensor(asset_df.values / 100, dtype=torch.double))[0:400,0:10]

# Partition dataset into training and testing sets
X_train, X_test = X[:200], X[200:]
Y_train, Y_test = Y[:200], Y[200:]

# Declare number of features n_x, number of assets n_y and number of observations n_obs
# n_obs is the number of observations given to the NN for distributional analysis
n_x, n_y, n_obs = X.shape[1], Y.shape[1], 100

####################################################################################################
# Neural net training and testing
####################################################################################################
#---------------------------------------------------------------------------------------------------
# E2E Nominal neural net
#---------------------------------------------------------------------------------------------------
# Neural net object
nom_net = nom.e2e(n_x, n_y, n_obs, prisk=rf.p_var).double()

# Train neural net
nom_net.net_train(X_train, Y_train, epochs=50, perf_loss=lf.single_period_over_var_loss)

# Test neural net
p_nom = nom_net.net_test(X_test, Y_test)

# Print parameter values and gradients
for name, param in nom_net.named_parameters():
    print(name, param.grad.data)
    print(name, param.data)
    
# Save/load trained model
model_path = my_path+"/saved_models/nom_net"
torch.save(nom_net, model_path)
test = torch.load(model_path)

#---------------------------------------------------------------------------------------------------
# E2E DRO neural net
#---------------------------------------------------------------------------------------------------
# Neural net object
dro_net = dro.e2edro(n_x, n_y, n_obs, prisk=rf.p_var, dro_layer=dro.tv).double()

# Train neural net
dro_net.net_train(X_train, Y_train, epochs=50, perf_loss=lf.single_period_over_var_loss)

# Test neural net
p_dro = dro_net.net_test(X_test, Y_test)

# Print parameter values and gradients
for name, param in dro_net.named_parameters():
    print(name, param.grad.data)
    print(name, param.data)
    
# Save/load trained model
model_path = my_path+"/saved_models/dro_net"
torch.save(dro_net, model_path)
test = torch.load(model_path)

dro.e2edro()