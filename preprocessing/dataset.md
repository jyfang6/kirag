
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
│
│── 2wikimultihopqa/
│   ├── raw_data/
│   │   ├── train.json
│   │   ├── dev.json
│   │   ├── test.json
│   ├── open_domain_data/
│   │   ├── train_qa_pairs.json
│   │   ├── dev_qa_pairs.json
│   │   ├── test_qa_pairs.json
│   │   ├── corpus.json
│   │   ├── qrels.json
|
│── musique/
│   ├── raw_data/
│   │   ├── musique_ans_v1.0_train.jsonl
│   │   ├── musique_ans_v1.0_dev.jsonl
│   │   ├── musique_ans_v1.0_test.jsonl
│   ├── open_domain_data/
│   │   ├── train_qa_pairs.json
│   │   ├── dev_qa_pairs.json
│   │   ├── test_qa_pairs.json
│   │   ├── corpus.json
│   │   ├── qrels.json
|
│── bamboogle/
│   ├── raw_data/
│   │   ├── Bamboogle_Prerelease.tsv
│   ├── open_domain_data/
│   │   ├── test_qa_pairs.json
|
│── webqa/
│   ├── raw_data/
│   │   ├── webquestions-test.qa.csv
│   ├── open_domain_data/
│   │   ├── test_qa_pairs.json
|
│── wikipedia/
│   ├── psgs_w100.tsv
```

For datasets with public test sets (Bamboogle, WebQA and NQ), we report performance on their full test set. For datasets with non-public test sets (HotPotQA, 2WikiMultiHopQA, MuSiQue), we use their full development set as test sets and randomly sample 500 examples from their training set as development sets. 