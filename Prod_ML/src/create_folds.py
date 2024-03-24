#Imports
import pandas as pd
from sklearn.model_selection import KFold

if __name__ == "__main__":

    df=pd.read_csv(r"../input/Train.csv")

    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = KFold(n_splits=5)

    for  fold, (train_idx, val_idx) in enumerate(kf.split(X=df)):
        df.loc[val_idx,'kfold'] = fold

    df.to_csv(r"../input/train_folds.csv",index=False)    