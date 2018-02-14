import pandas as pd
import numpy as np

full = pd.read_csv('horrorPlots.csv')
horror = pd.read_csv('horrorTrain.csv')
# drop 
testing = full.drop(full.index[horror.index])
testing.to_csv('horrorNoLabels.csv', index = False)

label = [0, 1, 1, 1, 1, 0, 1, 0, 0, 1,
         1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
         0, 0, 0, 0, 1, 0, 1, 1, 1, 1,
         1, 0, 0, 0, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
         1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 
         1, 1, 0, 0, 1, 0, 0, 1, 1, 1,
         0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 
         0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
         1, 1, 0, 1, 1, 1, 0, 1, 0, 1]

horror['category'] = label
horror.to_csv('horrorWithLabels.csv', index = False)
