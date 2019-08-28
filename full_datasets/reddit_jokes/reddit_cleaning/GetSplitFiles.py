import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.utils import resample
import os

base_path = os.path.join("full_datasets", "reddit_jokes")
all_features = pd.read_csv(os.path.join(base_path, "reddit_full_data.csv"), encoding="utf-8", index_col=0)
print(all_features.head())

CUTOFF = 200
all_features["score"] = (all_features["score"] > CUTOFF).astype(int)
all_features["score"].value_counts()

all_features["text"] = all_features["title"] + "_____" + all_features["selftext"]
all_features = all_features.dropna()

all_features["text"] = all_features["text"].apply(lambda s: s.replace('\n', ''))
all_features["text"] = all_features["text"].apply(lambda s: s.replace('\t', ''))
all_features["text"] = all_features["text"].apply(lambda s: s.replace('\r', ''))

def upsample(df_majority, df_minority):
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=df_majority.shape[0],    # to match majority class
                                     random_state=42) # reproducible results

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    data_train = df_upsampled
    # Display new class counts
    return data_train

def downsample(df_majority, df_minority):
    # downsample majority class
    df_majority_upsampled = resample(df_majority, 
                                     replace=False,     # sample with replacement
                                     n_samples=df_minority.shape[0],    # to match majority class
                                     random_state=42) # reproducible results

    # Combine majority class with upsampled minority class
    df_downsampled = pd.concat([df_minority, df_majority_upsampled])
    data_train = df_downsampled
    # Display new class counts
    return data_train

data_train, data_split = train_test_split(all_features, test_size=0.3, stratify=all_features["score"], random_state=17)
print(data_train.shape, data_split.shape)
data_train.reset_index(inplace=True, drop=True)
data_val, data_test = train_test_split(data_split, test_size=0.5, stratify=data_split["score"], random_state=17)
print(data_val.shape, data_test.shape)

# now to sample
print("Sampling")
df_majority = data_val[data_val["score"]==0]
df_minority = data_val[data_val["score"]==1]
data_val = downsample(df_majority, df_minority)
print(data_val.shape, "Is the shape of the validation")

df_majority = data_test[data_test.score==0]
df_minority = data_test[data_test.score==1]
data_test = downsample(df_majority, df_minority)
print(data_test.shape, "Is the shape of the test")

df_majority = data_train[data_train.score==0]
df_minority = data_train[data_train.score==1]
data_train = upsample(df_majority, df_minority)
data_train = shuffle(data_train)
print(data_train.shape, "Is the shape of the train")

print(data_train["score"].value_counts())

data_train.to_csv(os.path.join(base_path, "reddit_cleaning", "output/train.csv"), encoding="utf-8", header=True)
data_test.to_csv(os.path.join(base_path, "reddit_cleaning", "output/test.csv"), encoding="utf-8", header=True)
data_val.to_csv(os.path.join(base_path, "reddit_cleaning", "output/dev.csv"), encoding="utf-8", header=True,)

print("Done splitting")
