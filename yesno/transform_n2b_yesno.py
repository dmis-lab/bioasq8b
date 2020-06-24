import json,time
import numpy as np
import os
import argparse

from collections import Counter

parser = argparse.ArgumentParser(description='Shape the answer')
parser.add_argument('--nbest_path', type=str, help='location of nbest_predictions.json')
parser.add_argument('--output_path', type=str, help='location of output path')
args = parser.parse_args()

    
### Setting basic strings 

#### Checking nbest_BioASQ-test prediction.json
if not os.path.exists(args.nbest_path):
    print("No file exists!\n#### Fatal Error : Abort!")
    raise

#### Reading Pred File
with open(args.nbest_path, "r") as reader:
    test=json.load(reader)


qidDict=dict()
if True: # multi output
    for multiQid in test:
        assert len(multiQid)==(24+4) # all multiQid should have length of 24 + 3
        if not multiQid[:-4] in qidDict:
            qidDict[multiQid[:-4]]=[test[multiQid]]
        else :
            qidDict[multiQid[:-4]].append(test[multiQid])
else: # single output
    qidDict={qid:[test[qid]] for qid in test}    

entryList=[]
print(len(qidDict), 'number of questions')
for qid in qidDict:
    yesno_prob = {'yes': [], 'no': []}
    yesno_cnt = 0
    for ans, prob in qidDict[qid]:
        yesno_prob['yes'] += [float('{:.3f}'.format(prob[0]))] # For sigmoid
        yesno_cnt += 1
    
    mean = lambda x: sum(x)/len(x)
    final_answer = 'yes' if mean(yesno_prob['yes']) > 0.5 else 'no'

    entry={u"type": "yesno",
    u"id": qid,
    u"ideal_answer": ["Dummy"],
    u"exact_answer": final_answer,
    }
    entryList.append(entry)
finalformat={u'questions':entryList}

if os.path.isdir(args.output_path):
    outfilepath=os.path.join(args.output_path, "BioASQform_BioASQ-answer.json") # For unified output name
else:
    outfilepath=args.output_path

with open(outfilepath, "w") as outfile:
    json.dump(finalformat, outfile, indent=2)
    print("outfilepath={}".format(outfilepath))