import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

short_jokes = pd.read_csv("shortjokes.csv", header=None, encoding="utf-8")

print(short_jokes.head())
# average sentence length
average_sentence = np.mean(short_jokes[1].apply(lambda s: len(s.split("."))))

news_list = []
with open("news.2007.en.shuffled") as mytxt:
    for index, line in enumerate(mytxt):
            news_list.append(line)
            if index > 231657*7:  # max len of lines needed
                break
                
print(len(news_list))
            
index = 0
total_index = 0
news = []
while index < 231657:
    if not index % 10000:
        print(index)
    current_chunk = ""
    count = random.randint(1,round(average_sentence*2))
    for sent in range(count):
        current_chunk += news_list[total_index]
        total_index += 1
    news.append({"text": current_chunk, "score":0})
    index += 1

print(news[0])
len(news)

# assign these to be "not funny"
news_df = pd.DataFrame(news)
print(news_df.shape)
print(news_df.head())

# assign these to be "funny"
short_jokes["score"] = 1
short_jokes = short_jokes.iloc[:, 1:]
short_jokes.columns = ["text", "score"]
print(short_jokes.head())
print(short_jokes.columns)

def replace_contraction(text):
    contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'can not'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                         (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text

def replace_links(text, filler=' '):
        text = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*',
                      filler, text).strip()
        return text
    
def remove_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def cleanText(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = replace_contraction(text)
    text = replace_links(text, "link")
    text = remove_numbers(text)
    text = re.sub(r'[,”@#$%^&*)“(|/><";:\'\\}{]',"",text)
    text = text.lower()
    text = text.replace('\t', '')
    text = text.replace('\r', '')
    text = text.replace('\n', '')
    return text

news_df["text"] = news_df["text"].apply(lambda x: cleanText(x))
short_jokes["text"] = short_jokes["text"].apply(lambda x: cleanText(x))

full = pd.concat([news_df, short_jokes])
# Display new class counts
print(full.groupby('score').count())

print(full.isnull().any(axis=1).sum())
full["same_letter"] = "a"
full =  full[['score', 'same_letter', 'text']]

data_train, data_val = train_test_split(full, test_size=0.25)
data_dev, data_test = train_test_split(data_val, test_size=0.5)
data_train.reset_index(inplace=True, drop=True)
data_dev.reset_index(inplace=True, drop=True)
data_test.reset_index(inplace=True, drop=True)

data_train.to_csv("shortjokes/train.csv", encoding="utf-8", header=False,)
data_dev.to_csv("shortjokes/dev.csv", encoding="utf-8", header=False,)
data_test[["text"]].to_csv("shortjokes/test.csv", encoding="utf-8", header=True,)
