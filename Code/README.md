# Experiments

Please understand that we lack the time and resources to maintain this code repository. The code here is mainly for transparency and to validate the findings in our paper. The code is not optimized for production settings.

In order to run an annotation experiment proceed as follows: 

1. predict label candidates for a batch using ``./qa/predict.py``
2. simulate annotations on that batch using ``./annotation/annotate.py``
3. retrain QA model on annotated samples using ``./bert/run_squad.py`` 


A bash script executing that experiment could look like the following:

```bash
export BERT_BASE_DIR=

#python ./qa/predict.py --vocab_file=${BERT_BASE_DIR}/vocab.txt --bert_config_file=${BERT_BASE_DIR}/bert_config.json --output_dir=./data/reader_output/ --init_checkpoint=/tmp/trainonsquad/ --do_predict=True --predict_file=./data/nq_train_questions_title.json  --retriever_model=./data/wikidump/database-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz --doc_db=./data/wikidump/database --out_name=logits_train_final --batch_number=0 --max_seq_length=512 --predict_batch_size=32
#python ./qa/predict.py --vocab_file=${BERT_BASE_DIR}/vocab.txt --bert_config_file=${BERT_BASE_DIR}/bert_config.json --output_dir=./data/reader_output/ --init_checkpoint=/tmp/trainonsquad/ --do_predict=True --predict_file=./data/nq_train_questions_title.json  --retriever_model=./data/wikidump/database-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz --doc_db=./data/wikidump/database --out_name=logits_train_final --batch_number=1 --max_seq_length=512 --predict_batch_size=32

k=100
while [[ ${k} -lt 108 ]]
do
    python annotation/annotate.py --current-batch ${k} --mc-feature-template logits_train_final-feat-batch-{}.txt --out-name final
    
    rm -r /tmp/finetune
    mkdir /tmp/finetune
    python ./bert/run_squad.py  --vocab_file="${BERT_BASE_DIR}"/vocab.txt --bert_config_file="${BERT_BASE_DIR}"/bert_config.json --init_checkpoint=/tmp/trainonsquad --do_train=True --train_file=./final.json --do_predict=False --train_batch_size=16 --learning_rate=3e-5 --num_train_epochs=2.0 --max_seq_length=368 --doc_stride=128 --output_dir=/tmp/finetune
    k=$((k+1))
    python ./qa/predict.py --vocab_file=${BERT_BASE_DIR}/vocab.txt --bert_config_file=${BERT_BASE_DIR}/bert_config.json --output_dir=./data/reader_output/ --init_checkpoint=/tmp/finetune/ --do_predict=True --predict_file=./data/nq_train_questions_title.json  --retriever_model=./data/wikidump/database-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz  --max_seq_length=512 --predict_batch_size=16 --doc_db=./data/wikidump/database --out_name=logits_train_final --batch_number=${k}
done 
```