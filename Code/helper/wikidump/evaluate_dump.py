import sqlite3
import os
import json

db_path = './database'   # path to the full database
connection = sqlite3.connect(db_path, check_same_thread=False)
c = connection.cursor()
c.execute("SELECT id FROM documents")
results = [r[0] for r in c.fetchall() ]
c.close()

results_map = {}
for r in results:
    results_map[r] = 1

for fn in ['nq_dev_questions_title.json','nq_train_questions_title.json']:
    print(fn)
    has_long = 0
    has_short = 0
    tot = 0
    with open(os.path.join('../..', 'DrQA', fn), 'r') as f:
        found = 0 
        data = json.load(f)
        for d in data:
            if d['doc'] in results_map:
                found += 1
            else: 
                tot += 1
                has_long += d['has_long']
                has_short += d['has_short']

        print(found/len(data))
        print()
        print(has_long/tot)
        print(has_short/tot)
        print()
