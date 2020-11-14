import json
import os
import operator
import numpy as np
from sklearn.metrics import classification_report

data = []
with open('v1.0-simplified_simplified-nq-train.jsonl', 'r') as f:
    for line in f:
        dat = json.loads(line)

        sa_annotation = dat['annotations'][0]['short_answers']
        if len(sa_annotation) > 0:
            sa_annotation = sa_annotation[0]
            short_answer = dat['document_text'].split(' ')[sa_annotation['start_token']:sa_annotation['end_token']]
            short_answer = ' '.join(short_answer)
            short_answer_length = len(short_answer)
        else:
            continue
        assert dat['annotations'][0]['yes_no_answer'] == "NONE"

        long_answ_ind = dat['annotations'][0]['long_answer']['candidate_index']
        long_answ_ind_original = long_answ_ind

        long_answ_cand = dat['long_answer_candidates'][long_answ_ind]
        long_answ_cand_orig = long_answ_cand

        while long_answ_cand['top_level'] == False and long_answ_ind > 0:
            long_answ_ind -= 1
            long_answ_cand = dat['long_answer_candidates'][long_answ_ind]

        rel_offset = sa_annotation['start_token'] - long_answ_cand['start_token']

        long_answer = dat['document_text'].split(' ')[long_answ_cand['start_token']:long_answ_cand['end_token']]
        rel_char_offset = len(' '.join(long_answer[0: rel_offset]))
        if rel_offset > 0:
            rel_char_offset += 1
        long_answer = ' '.join(long_answer)

        assert long_answer[rel_char_offset:rel_char_offset + short_answer_length] == short_answer

        assert long_answ_cand['top_level'] is True, long_answ_cand
        assert long_answ_cand['start_token'] <= long_answ_cand_orig['start_token']
        assert long_answ_cand['end_token'] >= long_answ_cand_orig['end_token']

        question_text = dat['question_text']
        if question_text[-1] == '?':
            pass
        else:
            question_text = question_text + '?'

        data.append({'title': dat['document_url'], 'paragraphs': [{"context": long_answer,
                                                                   "qas": [{"id": dat['example_id'],
                                                                            "question": question_text,
                                                                            "answers": [{
                                                                                "answer_start": rel_char_offset,
                                                                                "text": short_answer}]}]}]})
        if len(data) % 5000 == 0:
            print(len(data))

with open('nq_squad_format_train.json', 'w') as f:
    json.dump({'data': data}, f)
print(len(data))
