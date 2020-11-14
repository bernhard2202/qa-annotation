import json
import urllib.parse

all_docs = {}
with open('../data/v1.0-simplified_nq-dev-all.jsonl', 'r') as f:
    for line in f:
        dat = json.loads(line)
        if dat['document_title'] not in all_docs:
            tags = ['<H1>', '<H2>', '<H3>', '<Tr>', '<Td>', '<Ul>', '<Th>', '</Th>', '<Li>', '<Table>', '<P>', '<Br>', '</H1>', '</H2>', '</H3>', '</Tr>', '</Td>', '</Ul>', '</Li>', '</Table>', '</P>', '</Br>']

            doc_text = ' '.join([t['token'] for t in dat['document_tokens'] if not t['html_token'] or t['token'] in tags])
            if doc_text.find('<H2> References </H2>') > 0:
                doc_text = doc_text[:doc_text.find('<H2> References </H2>')]

            if doc_text.find('About Wikipedia') > 0:
                doc_text = doc_text[:doc_text.find('About Wikipedia')]

            tokens = doc_text.split(' ')
            text = ' '.join([t if t not in tags else '\n' for t in tokens])
            all_docs[dat['document_title']] = text

with_short = 0
with open('../data/v1.0-simplified_simplified-nq-train.jsonl', 'r') as f:
    for line in f:
        dat = json.loads(line)
        url_info = urllib.parse.parse_qs(dat['document_url'][dat['document_url'].find('?') + 1:])
        dat['document_title'] = url_info['title'][0].replace('_', ' ')
        if dat['document_title'] not in all_docs:
            doc_text = dat['document_text']
            if doc_text.find('<H2> References </H2>') > 0:
                doc_text = doc_text[:doc_text.find('<H2> References </H2>')]
            if doc_text.find('About Wikipedia') > 0:
                doc_text = doc_text[:doc_text.find('About Wikipedia')]

            tags = ['<H1>', '<H2>', '<H3>', '<Tr>', '<Td>', '<Ul>', '<Th>', '</Th>', '<Li>', '<Table>', '<P>', '<Br>',
                    '</H1>', '</H2>', '</H3>', '</Tr>', '</Td>', '</Ul>', '</Li>', '</Table>', '</P>', '</Br>']
            tokens = doc_text.split(' ')
            text = ' '.join([t if t not in tags else '\n' for t in tokens])

            all_docs[dat['document_title']] = text
            # print(' '.join([t['token'] for t in dat['document_tokens'] if not t['html_token']]))

with open('../data/all_docs.json', 'w') as f:
    json.dump(all_docs, f)