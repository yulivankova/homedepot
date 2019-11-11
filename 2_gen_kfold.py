import sys
import _pickle as cPickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from param_config import config
import pandas as pd

if __name__ == "__main__":

    ## load data
    df_train = pd.read_csv(config.original_train_data_path,encoding=config.encoding)
    key = "relevance"

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_train[key])

    skf = [0]*config.n_runs
    for run in range(config.n_runs):
        random_seed = 2016 + 1000 * (run+1)
        skf[run] = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=random_seed)
        for fold, (trainInd,validInd) in enumerate(skf[run].split(df_train.values, y)):
            print("================================")
            print("Index for run: %s, fold: %s" % (run+1, fold+1))
            print("Train (num = %s)" % len(trainInd))
            print(trainInd[:10])
            print("Valid (num = %s)" % len(validInd))
            print(validInd[:10])
    with open("%s/stratifiedKFold.%s.pkl" % (config.output_folder, key), "wb") as f:
        cPickle.dump(skf, f, -1)
