# Humor Detection
## Code and Datasets for the Paper "Humor Detection: A Transformer Gets the Last Laugh" by Orion Weller and Kevin Seppi
The repository contains the following:
 - A way to regenerate the results found in the paper, by running `bash run_bert.sh`.  
 - The full datasets referenced in the paper (short jokes, puns, and the reddit dataset) are located in `full_datasets` whereas the `data` folder contains the split files used for training and testing.  The file `create_data.sh` will create the splits (slightly different from the ones used in the paper - see `create_data.sh`).
- pytorch_pretrained_bert contains files used by the model - these files are from the [huggingface repo](https://github.com/huggingface/pytorch-transformers#Training-large-models-introduction,-tools-and-examples) and are NOT up to date with the current `pytorch-transformers` repo.  

**This repository is is not maintained and will not be updated.**

## How to cite this paper:
Authors of scientific papers who use this repository are encouraged to cite the following paper:
```
@ARTICLE{humorDetection2019,
  title={Humor Detection: A Transformer gets the Last Laugh},
  author={Weller, Orion and Seppi, Kevin},
  journal={"Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing"},
  month=Nov,
  year = "2019",
}
```


