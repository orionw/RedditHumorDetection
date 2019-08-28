import pandas as pd

def split_on_score(df):
    pos = df[df["score"] == 1]
    neg = df[df["score"] == 0]
    pos.drop("score", axis=1, inplace=True)
    neg.drop("score", axis=1, inplace=True)
    assert(len(pos) + len(neg) == len(df))
    return pos, neg

for string in ["full", "punch", "body"]:
    train = pd.read_csv("output/output_for_bert/{}/train.csv".format(string), encoding="utf-8", names=['score', 'same_letter', "text"], index_col=None)
    dev = pd.read_csv("output/output_for_bert/{}/dev.csv".format(string), encoding="utf-8", names=['score', 'same_letter', "text"], index_col=None)
    test = pd.read_csv("output/output_for_bert/{}/test.csv".format(string), encoding="utf-8", names=['score', 'same_letter', "text"], index_col=None)

    train.drop("same_letter", axis=1, inplace=True)
    dev.drop("same_letter", axis=1, inplace=True)
    test.drop("same_letter", axis=1, inplace=True)
    # This is where we split into positive and negative for the CNN
    train_pos, train_neg = split_on_score(train)
    test_pos, test_neg = split_on_score(test)
    dev_pos, dev_neg = split_on_score(dev)
    
    train_pos.to_csv("output/output_for_bert/{}/train_pos.txt".format(string), encoding="utf-8", header=None, index=None, sep=' ', mode='w')
    test_pos.to_csv("output/output_for_bert/{}/test_pos.txt".format(string), encoding="utf-8", header=None, index=None, sep=' ', mode='w')
    dev_pos.to_csv("output/output_for_bert/{}/dev_pos.txt".format(string), encoding="utf-8", header=None, index=None, sep=' ', mode='w')


    train_neg.to_csv("output/output_for_bert/{}/train_neg.txt".format(string), header=None, index=None, sep=' ', mode='w')
    test_neg.to_csv("output/output_for_bert/{}/test_neg.txt".format(string), header=None, index=None, sep=' ', mode='w')
    dev_neg.to_csv("output/output_for_bert/{}/dev_neg.txt".format(string), header=None, index=None, sep=' ', mode='w',)

    print(train_pos.head())

print("Output has been created")

