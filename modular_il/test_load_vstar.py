import keras
from sklearn.preprocessing import StandardScaler
from learn_vf_from_demo import Expert_Vf
import cPickle
import numpy as np


hopper_vstar = cPickle.load(open("Est_Hopper_Vstar",'rb'));
raw_obs = np.random.rand(11)
est_vstar = hopper_vstar.predict(raw_obs, 2)[0][0]  
#where 2 is the time step of this raw observation
print est_vstar








