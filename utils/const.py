
from dataset.collators import E5Collator, BGECollator
from dataset.corpus import Wikipedia, HotPotQA, WikiMultiHopQA, MuSiQue

COLLATOR_MAP = {
    "E5Retriever": E5Collator, 
    "BGERetriever": BGECollator
}

CORPUS_MAP = {
    "wikipedia": Wikipedia,
    "hotpotqa": HotPotQA, 
    "2wikimultihopqa": WikiMultiHopQA, 
    "musique": MuSiQue
}