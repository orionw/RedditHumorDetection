#  This script will take the full data and regenerate the TSV files needed for the model.
#  Unfortunately, due to how new I was at this, I didn't set seeds in all the locations that are needed. Fortunately, I did save the original data splits which are located in the `data` folder. However, recreating this (as is currently in these files) shows results that are ~1% off in either direction and are consistent with reported paper results.
pip3 install -r requirements.txt
# process the data
python3 full_datasets/reddit_jokes/reddit_cleaning/GetSplitFiles.py
python3 full_datasets/reddit_jokes/reddit_cleaning/GetTSVFileForBERT.py
cp full_datasets/reddit_jokes/reddit_cleaning/output/output_for_bert/full/*.tsv data/
# remove the validation set
rm data/dev.tsv
# move the test set for evaluation
mv data/test.tsv data/dev.tsv
