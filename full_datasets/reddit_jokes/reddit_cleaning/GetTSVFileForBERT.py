import pandas as pd
import os

base_path = os.path.join("full_datasets", "reddit_jokes", "reddit_cleaning")
train = pd.read_csv(os.path.join(base_path, "output/train.csv"), encoding="utf-8")
dev = pd.read_csv(os.path.join(base_path, "output/dev.csv"), encoding="utf-8")
test = pd.read_csv(os.path.join(base_path, "output/test.csv"), encoding="utf-8")

print(train.head())

def ready_for_bert(given_df, keep_column="text"):
    data = given_df.copy(deep=True)
    data[keep_column] = data[keep_column].apply(lambda s: s.replace('\n', ''))
    data[keep_column] = data[keep_column].apply(lambda s: s.replace('\t', ''))
    data[keep_column] = data[keep_column].apply(lambda s: s.replace('\r', ''))
    data["same_letter"] = "a"
    data = data[["score", "same_letter", keep_column]]
    return data

train_full = ready_for_bert(train)
test_full = ready_for_bert(test)
dev_full = ready_for_bert(dev)

train_body = ready_for_bert(train, "title")
test_body = ready_for_bert(test, "title")
dev_body = ready_for_bert(dev, "title")

train_punch = ready_for_bert(train, "selftext")
test_punch = ready_for_bert(test, "selftext")
dev_punch = ready_for_bert(dev, "selftext")

train_full.head()

print(train_full["score"].value_counts())

train_full.to_csv(os.path.join(base_path, "output/output_for_bert/full/train.tsv"), encoding="utf-8", header=False)
test_full.to_csv(os.path.join(base_path, "output/output_for_bert/full/test.tsv"), encoding="utf-8", header=False)
dev_full.to_csv(os.path.join(base_path, "output/output_for_bert/full/dev.tsv"), encoding="utf-8", header=False,)


train_body.to_csv(os.path.join(base_path, "output/output_for_bert/body/train.tsv"), encoding="utf-8", header=False)
test_body.to_csv(os.path.join(base_path, "output/output_for_bert/body/test.tsv"), encoding="utf-8", header=False)
dev_body.to_csv(os.path.join(base_path, "output/output_for_bert/body/dev.tsv"), encoding="utf-8", header=False,)


train_punch.to_csv(os.path.join(base_path, "output/output_for_bert/punch/train.tsv"), encoding="utf-8", header=False)
test_punch.to_csv(os.path.join(base_path, "output/output_for_bert/punch/test.tsv"), encoding="utf-8", header=False)
dev_punch.to_csv(os.path.join(base_path, "output/output_for_bert/punch/dev.tsv"), encoding="utf-8", header=False,)

