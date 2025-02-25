
## Dataset Structure 
After downloading the data files for a dataset, we put them under the `raw_data` folder of each dataset. We expect the following file structure, where `open_domain_data` folder contains the preprocessed data and the retrieval corpus.
```
data/
│── hotpotqa/
│   ├── raw_data/
│   │   ├── hotpot_train_v1.1.json
│   │   ├── hotpot_dev_distractor_v1.json
│   │   ├── enwiki-20171001-pages-meta-current-withlinks-abstracts
│   ├── open_domain_data/
│   │   ├── train_qa_pairs.json
│   │   ├── dev_qa_pairs.json
│   │   ├── test_qa_pairs.json
│   │   ├── corpus.json
│   │   ├── qrels.json
```

## Preprocessing Steps 

### HotPotQA 
For HotPotQA dataset, we use the [corpus](https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2) released alongside the dataset for retrieval. Since the test set is non-public, we use its full development set as test set. Moreover, we randomly sample 500 questions from its original training set as development set and use the rest as training set. 

### 2WikiMultiHopQA and MuSiQue
We follow the procedure in [IRCoT](https://arxiv.org/abs/2212.10509) to construct retrieval corpus for these two datasets, where we combine all the paragraphs for all questions in the training, development and test sets. Similar to HotPotQA, we also use their original development sets as test sets and randomly sample 500 questions from their original training sets as development sets. 

### Bamboogle, WebQA and NQ 
For these datasets, we only use their test tests for evaluation. We use the Wikipedia dump released by DPR as the retrieval corpus. 
