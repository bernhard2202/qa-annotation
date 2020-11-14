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

max_length = 1000


def policy(x, beta):
    return (x*beta)/((beta-1)*x+1)


class COSTS:
    IR_MANUAL = 0.75
    IR_TOPX = 0.25
    MC_MANUAL = 0.75
    MC_TOPX = 0.25

if __name__ == '__main__':
    args = parser.parse_args()

    all_features = ['span_score', 'start_logit', 'end_logit', 'paragraph_score', 'span_len', 'first_occ', 'num_occ',
                    'avg_span_score', 'max_span_score', 'avg_start_logit', 'max_start_logit', 'min_start_logit',
                    'min_end_logit', 'max_end_logit', 'avg_end_logit', 'min_span_score', 'avg_paragraph_score',
                    'max_paragraph_score', 'min_paragraph_score', ]


    all_x_train0 = []
    all_x_train1 = []
    for i in range(0, args.current_batch):
        print('Starting with {}..'.format(i))
        with open(os.path.join(args.mc_feature_path, args.mc_feature_template.format(i)), 'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                odata = data['output']

                X = np.zeros((len(all_features) * args.mc_top_n))
                d = data['output']['samples']
                for j in range(min(args.mc_top_n, len(d))):
                    for k, feat in enumerate(all_features):
                        if feat in d[j]:
                            X[j * len(all_features) + k] = d[j][feat]

                lowest = 10
                j = 0
                for s in odata['samples']:
                    if s['target'] == 1:
                        lowest = j
                        break
                    j += 1

                label = 1 if lowest < 5 else 0

                if label == 1:
                    all_x_train1.append(X)
                if label == 0:
                    all_x_train0.append(X)

    budget = len(all_x_train1)

    beta = len(all_x_train1)/len(all_x_train0)
    all_x_train0 = all_x_train0[-budget:]
    all_x_train1 = all_x_train1[-budget:]

    all_x_train = []
    all_y_train = []

    for a,b in zip(all_x_train1,all_x_train0):
        all_x_train.append(a)
        all_y_train.append(1)
        all_x_train.append(b)
        all_y_train.append(0)

    all_x_train = np.stack(all_x_train, axis=0)

    all_original_samples = {}
    with open('data/nq_squad_format_train.json', 'r') as f:
        original_samples = json.load(f)
        for orig in original_samples['data']:
            orig['full_annotation'] = True
            question = orig['paragraphs'][0]['qas'][0]['question']
            all_original_samples[question[:-1]] = orig
            all_original_samples[question] = orig

    all_questions = []
    all_x_test = []
    all_y_test = []
    test_samples_original = []
    test_samples_ds = []
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
                    answer_start = doc.find(answer)
                    if answer_start == -1:
                        answer_start = doc.lower().find(answer.lower())
                    sample = {'full_annotation': False, 'paragraphs': [{'context': doc, 'qas': [
                                            {'question': data['original']['question'], 'answers': [
                                                {'text': answer, 'answer_start': answer_start}]}]}]}
                    break
                i += 1

            label = 1 if lowest < 5 else 0
            if label == 1:
                assert sample is not None
                test_samples_ds.append(sample)
            else:
                test_samples_ds.append(None)
            test_samples_original.append(all_original_samples[data['original']['question']])

            all_y_test.append(label)
            all_questions.append(data['original']['question'])
            X = np.zeros((len(all_features) * args.mc_top_n))
            d = data['output']['samples']
            for j in range(min(args.mc_top_n, len(d))):
                for k, feat in enumerate(all_features):
                    if feat in d[j]:
                        # TODO log transform..
                        X[j * len(all_features) + k] = d[j][feat]
            all_x_test.append(X)
    all_x_test = np.stack(all_x_test, axis=0)
    print(np.shape(all_x_test))

    dropout_prob = 0.3
    batch_size = 32
    hidden_dims = 128
    
    input_shape = ((len(all_features) * args.mc_top_n),)

    model_input = Input(shape=input_shape)
    z = Dense(hidden_dims, activation="relu")(model_input)
    z = Dropout(dropout_prob)(z)
    z = Dense(int(hidden_dims/2), activation="relu")(z)
    z = Dropout(dropout_prob)(z)
    model_output = Dense(1, activation="sigmoid")(z)

    model = Model(model_input, model_output)

    # Training parameters
    num_epochs = 25
    learning_rate = 0.0001

    def my_cuystom_loss(y_true, y_pred):
        return keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)

    optim = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=my_cuystom_loss, optimizer=optim, metrics=["accuracy"])
    # weights = class_weight.compute_class_weight('balanced',
    #                                            np.unique(all_y_train),
    #                                            all_y_train)
    # weights = {0: weights[0], 1: weights[1]}
    model.fit(all_x_train, all_y_train, batch_size=batch_size,
              epochs=num_epochs, validation_data=(all_x_test, all_y_test),
              verbose=1, shuffle=True)

    action_log = [] 

    y_pred_train = model.predict(all_x_train, )
    new_samples = []
    costs = []
    y_pred = model.predict(all_x_test, )
    for i in range(len(y_pred)):
        action = 1 if policy(y_pred[i], beta) >= (COSTS.IR_TOPX/COSTS.IR_MANUAL) else 0
        cost = 0
        if action == 0:
            cost = COSTS.IR_MANUAL
            new_samples.append(test_samples_original[i])
        else:
            new_samples.append(test_samples_ds[i] if all_y_test[i] == 1 else test_samples_original[i])
            cost = COSTS.IR_TOPX if all_y_test[i] == 1 else COSTS.IR_TOPX + COSTS.IR_MANUAL
        costs.append(cost)
        optimal = all_y_test[i] == action
        action_log.append({'chosen_action': action, 'p': float(policy(y_pred[i], beta)), 'payed_cost': cost, 'optimal': optimal, 'in_batch': args.current_batch, 'in_batch_num': i, 'question': all_questions[i] })

    print(np.mean(costs))
    
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
    old_data.extend(action_log)
    with open(ds_file, 'w') as f:
        json.dump(old_data, f)
