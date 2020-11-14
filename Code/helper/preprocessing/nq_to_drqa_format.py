import json
import urllib.parse

data = []
without_long = 0
without_short = 0
with_short = 0
with open('../data/v1.0-simplified_nq-dev-all.jsonl', 'r') as f:
    for line in f:
        dat = json.loads(line)
        non_null_long_answer = 0
        non_null_short_answer = 0
        for annotation in dat['annotations']:
            if (annotation['long_answer']['candidate_index']) >= 0:
                non_null_long_answer += 1
                if len(annotation['short_answers']) > 0:
                    non_null_short_answer += 1
        if non_null_long_answer < 2:
            data.append({'question': dat['question_text'], 'doc': dat['document_title'], 'has_long': 0, 'has_short': 0})
            without_long += 1
            without_short += 1
        else:
            if non_null_short_answer < 2:
                without_short += 1
                data.append(
                    {'question': dat['question_text'], 'doc': dat['document_title'], 'has_long': 1, 'has_short': 0})
            else:
                with_short += 1
                data.append(
                    {'question': dat['question_text'], 'doc': dat['document_title'], 'has_long': 1, 'has_short': 1})
print(without_long / len(data))
print(without_short / len(data))
print(with_short / len(data))

with open('../data/nq_dev_questions_title.json', 'w') as f:
    json.dump(data, f)

data = []
without_long = 0
without_short = 0
with_short = 0
with open('../data/v1.0-simplified_simplified-nq-train.jsonl', 'r') as f:
    for line in f:
        dat = json.loads(line)
        assert len(dat['annotations']) == 1, len(dat['annotations'])
        url_info = urllib.parse.parse_qs(dat['document_url'][dat['document_url'].find('?') + 1:])
        dat['document_title'] = url_info['title'][0].replace('_', ' ')

        if dat['annotations'][0]['long_answer']['candidate_index'] == -1:
            data.append({'question': dat['question_text'], 'doc': dat['document_title'], 'answer': -1, 'has_long': 0,
                         'has_short': 0})
            without_long += 1
            without_short += 1
        else:
            sa_annotations = dat['annotations'][0]['short_answers']
            if len(sa_annotations) > 0:
                answers = []
                for sa_annotation in sa_annotations:
                    short_answer = dat['document_text'].split(' ')[
                                   sa_annotation['start_token']:sa_annotation['end_token']]
                    short_answer = ' '.join(short_answer)
                    answers.append(short_answer)
                data.append(
                    {'question': dat['question_text'], 'doc': dat['document_title'], 'answer': answers, 'has_long': 1,
                     'has_short': 1})
                with_short += 1
            else:
                without_short += 1
                data.append(
                    {'question': dat['question_text'], 'doc': dat['document_title'], 'answer': -1, 'has_long': 1,
                     'has_short': 0})
        if (with_short + without_short) % 5000 == 0:
            print(with_short + without_short)
print(without_long / len(data))
print(without_short / len(data))
print(with_short / len(data))

with open('../data/nq_train_questions_title.json', 'w') as f:
    json.dump(data, f)
