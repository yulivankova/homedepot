
"""
__file__

	run_all.py

__description___
	
	This file generates all the features in one shot.

__adopted from__

	Chenglong Chen < c.chenglong@gmail.com >

"""

import os

#################
## Preprocesss ##
#################

cmd = "python ./0_preprocess.py"
os.system(cmd)

#######################
## Generate features ##
#######################

cmd = "python ./1_feature_engineering.py"
os.system(cmd)

####################
## Create K folds ##
####################

cmd = "python ./2_gen_kfold.py"
os.system(cmd)

