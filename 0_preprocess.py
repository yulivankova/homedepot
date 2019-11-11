import sys
import _pickle as cPickle
import pandas as pd
from param_config import config
from spell_check_dict import spell_check_dict

###############
## Load Data ##
###############
print("Loading data...")
df_train = pd.read_csv(config.original_train_data_path,encoding=config.encoding)
df_test = pd.read_csv(config.original_test_data_path,encoding=config.encoding)
df_product_descriptions = pd.read_csv(config.original_product_descriptions_data_path,encoding=config.encoding)
df_attributes = pd.read_csv(config.original_attributes_data_path,encoding=config.encoding)

# number of train/test samples
num_train, num_test = df_train.shape[0], df_test.shape[0]

print("Number of training samples: %d" %num_train)
print("Number of testing samples: %d" %num_test)

#####################
#### Fixing Typos####
#####################
print("Fixing Typos in df_train.search_term ...")

f = lambda x: spell_check_dict.get(x, x)
for search in spell_check_dict:
    df_train.loc[(df_train['search_term']==search),'search_term'] = df_train.loc[(df_train['search_term']==search),'search_term'].map(f)

#Fixing typos in df_test.search_term
f= lambda x: spell_check_dict.get(x, x)
for search in spell_check_dict:
    df_test.loc[(df_test['search_term']==search),'search_term'] = df_test.loc[(df_test['search_term']==search),'search_term'].map(f)

######################################
## Merge all data into one dataframe ##
######################################
print("Merging all data into one frame...")
df_all = df_train.append(df_test, ignore_index=True)
df_all = pd.merge(df_all, df_product_descriptions, how='left', on='product_uid')

################
## Save Data ##
###############
print("Saving data...")
with open(config.output_folder+"/Dataset.pkl","wb") as f:
    cPickle.dump(df_all,f,-1)
print("dataframe has %r rows(samples) and %r columns(features)" % (df_all.shape))
print("Done.")
