import numpy as np
import json
from tqdm import tqdm
from keras.preprocessing import sequence
import keras
from keras.models import Sequential, Model
from keras.engine.input_layer import Input
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Conv1D, Embedding, SeparableConv1D
from keras.layers.merge import Concatenate
from keras.datasets import imdb
import os
import argparse
import logging

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('--ir-top-n', type=int, default=25, help='top-x features to use for ir state description')
parser.add_argument('--ir-feature-path', type=str, default='./data/retriever_output', help="Path to ir feature file")

parser.add_argument('--mc-top-n', type=int, default=5, help='top-x features to use for mc state description')
parser.add_argument('--mc-feature-path', type=str, default='./data/reader_output', help="Path to mc feature files")
parser.add_argument('--mc-feature-template', type=str, default='logits_train-feat-batch-{}.txt', help="Path to mc feature files")

parser.add_argument('--current-batch', type=int, default=2, help='top-x features to use for mc state description')

parser.add_argument('--num-choices', type=int, default=5, help='number of choices semi supervision sees')

parser.add_argument('--oracle', type=int, default=0, help='number of choices semi supervision sees')
parser.add_argument('--out-name', type=str, default='generated_samples', help="Path to mc feature files")

max_length = 10000

class COSTS:
    IR_MANUAL = 0.75
    IR_TOPX = 0.25
    MC_MANUAL = 0.75
    MC_TOPX = 0.25

if __name__ == '__main__':
    args = parser.parse_args()

    all_original_samples = {}
    with open('data/nq_squad_format_train.json', 'r') as f:
        original_samples = json.load(f)
        for orig in original_samples['data']:
            orig['full_annotation'] = True
            question = orig['paragraphs'][0]['qas'][0]['question']
            all_original_samples[question[:-1]] = orig
            all_original_samples[question] = orig

    new_samples = []
    log = []
    in_batch_num = 0
    all_questions = []
    val_file = os.path.join(args.mc_feature_path, args.mc_feature_template.format(args.current_batch))
    print('Starting with {}..'.format(val_file))
    with open(val_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            odata = data['output']

            lowest = 10
            i = 0
            sample = None
            for s in odata['samples']:
                if s['target'] == 1:
                    lowest = i
                    doc = ' '.join(s['doc_tokens'])
                    answer = s['original_answer']
                    answer_start =  doc.find(answer)
                    if answer_start == -1:
                        answer_start =  doc.lower().find(answer.lower())
                    sample =  {'full_annotation': False, 'paragraphs': [{'context': doc, 'qas': [
                                            {'question': data['original']['question'], 'answers': [
                                                {'text': answer, 'answer_start': answer_start}]}]}]}
                    break
                i += 1

            label = 1 if lowest < 5 else 0
            if label == 1:
                assert sample is not None
                new_samples.append(sample)
            else:
                new_samples.append(all_original_samples[data['original']['question']])
            log.append({'chosen_action': label, 'cost': COSTS.IR_TOPX if label == 1 else COSTS.IR_MANUAL, 'optimal': True, 'in_batch': args.current_batch, 'in_batch_num': in_batch_num, question: data['original']['question']})

    ds_file = '{}.json'.format(args.out_name)
    old_data = []
    if os.path.exists(ds_file):
        with open(ds_file, 'r') as f:
            old_data = json.load(f)['data']
    old_data.extend(new_samples)
    with open(ds_file, 'w') as f:
        json.dump({'data': old_data}, f)

    ds_file = '{}.log'.format(args.out_name)
    old_data = []
    if os.path.exists(ds_file):
        with open(ds_file, 'r') as f:
            old_data = json.load(f)
    old_data.extend(log)
    with open(ds_file, 'w') as f:
        json.dump(old_data, f)
