import json
import os
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
import numpy as np

from utils import filter_word
from doc_db import DocDB

PROCESS_DB = None

def init(db_class, db_opts):
    global PROCESS_DB
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)

def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)

def tokenize(text):
    return [w for w in word_tokenize(text) if not filter_word(w)] 

def get_similarity(args):
    query = args[0]
    corpus = args[1]
    if corpus is None or len(corpus) == 0:
        return []
    bm25 = BM25Okapi(corpus)
    doc_scores = bm25.get_scores(query)
    return doc_scores.tolist()


if __name__ == "__main__":
    db_class = DocDB
    db_options = {'db_path': '../database'}
    processes = ProcessPool(
        20,  # TODO move to flags
        initializer=init,
        initargs=(db_class, db_options)
    )
    
    with open(os.path.join('..', '..', 'DrQA', 'results_2grams_train.json'), 'r') as f:
        data = json.load(f)
        
        
    BATCH_SIZE = 64
    batches = [(data[i: i + BATCH_SIZE]) for i in range(0, len(data), BATCH_SIZE)]
    
    sims = [] # bm25 scores for all documents
    qlens = [] # no of valid words in question len(question_tokens)
    title_overlap = []
    doc_overlap = []

    for batch in tqdm(batches):
        questions = [d[0] for d in batch]
        documents = [doc for d in batch for doc in d[2]]
        scores  = [d[3] for d in batch]
        batch_offsets = [len(d[2]) for d in batch]
        
        question_tokens = processes.map(tokenize, questions)

        doc_texts = processes.map(fetch_text, documents)
        doc_tokens = processes.map(tokenize, doc_texts)
    
        doc_batch = []
        start = 0
        for i in range(len(batch)):
            doc_batch.append(doc_tokens[start:start+batch_offsets[i]])
            start = start+batch_offsets[i]
            title_overlap.append([len(set(question_tokens[i]).intersection(set(tokenize(doc_title)))) for doc_title in batch[i][2]])
            doc_overlap.append([len(set(question_tokens[i]).intersection(set(dt))) for dt in doc_batch[-1]])

        similarities = processes.map(get_similarity, zip(question_tokens, doc_batch))
        sims.extend(similarities)
        qlens.extend([len(q) for q in question_tokens])
    
    with open('enriched.json', 'w') as f:
        data_new = list(zip(data,sims, qlens, title_overlap, doc_overlap))
        json.dump(data_new, f)
    assert len(doc_overlap) == len(data)
    assert len(sims) == len(data)
    assert len(qlens) == len(data)
    assert len(title_overlap) == len(data)
    