import argparse
import json
import numpy as np
from operator import itemgetter
import os
import pandas as pd
import string


parser = argparse.ArgumentParser(description='Shape the answer')
parser.add_argument('--nbest_path', help='location of nbest_predictions.json')
parser.add_argument('--output_path',  help='location of output')

parser.add_argument('--threshold', type=float, help='prob', default=0.3)
parser.add_argument('--check_num_len', type=int, help='', default=4)
parser.add_argument('--len_limit', type=int, help='', default=10)
parser.add_argument('--use_softmax', '-s', action='store_true')
parser.add_argument('--min_prob', type=float,
                    help='minimal prob if softmax', default=0.02)
parser.add_argument('--stop_prob', type=float,
                    help='stop accumulated prob if softmax', default=0.8)
parser.add_argument('--verbose', '-v', action='store_true')
args = parser.parse_args()


threshold = args.min_prob if args.use_softmax else args.threshold

print(args.__dict__)
print('Threshold', threshold)

numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']
int_numbers = [str(i+1) for i in range(len(numbers))]


def find_num(question_text):
    words = question_text.split()
    for widx, w in enumerate(words):
        if w in numbers:
            # Dirty rule for the following question
            # Name the phase 3 clinical trials for tofacitinib in colitis.
            if widx == 0 or 'phase' != words[widx - 1]:
                return numbers.index(w) + 1
        elif w in int_numbers:
            if widx == 0 or 'phase' != words[widx - 1]:
                return int_numbers.index(w) + 1

        # TODO performance comparison
        # check only first N words
        if widx >= args.check_num_len:
            break

    return None


def preprocess_phrase(p):
    p = p.replace('\u2028', '')
    p = p.replace('\u2029', '')
    # p = p.strip()
    return p


def filter_incomplete(a_text):
    if '(' in a_text and ')' not in a_text:
        return True

    if ')' in a_text and '(' not in a_text:
        return True

    first_char = a_text.strip()[0]
    if first_char in string.punctuation:
        # print('first_char is punctuation', a_text.strip())
        return True

    last_char = a_text.strip()[-1]
    if last_char in string.punctuation:
        return True

    return False


def resolve_a_and_b(answers, question_id):

    if len(answers) < 3:
        return answers

    answer0_set = set([answer[0].lower() for answer in answers])

    res_answers = list()
    for answer in answers:

        if ' and ' not in answer[0] \
                and ', ' not in answer[0] \
                and '/' not in answer[0]:
            res_answers.append(answer)
            continue

        delm_start = answer[0].find(' and ')
        if -1 == delm_start:
            delm_start = answer[0].find(', ')
            if -1 == delm_start:
                delm_start = answer[0].find('/')

                if delm_start == -1:
                    res_answers.append(answer)
                    continue

                b_start = delm_start + 1
            else:
                b_start = delm_start + 2
        else:
            b_start = delm_start + 5

        answer_a = answer[0][:delm_start]
        answer_b = answer[0][b_start:]

        answer_a = answer_a.strip()
        answer_b = answer_b.strip()

        if answer_a and answer_b \
                and answer_a.lower() in answer0_set \
                and answer_b.lower() in answer0_set:
            print('qid', question_id, 'Remove', answer[0],
                  'already in', answer0_set)
            continue

        # if '' != answer_a and answer_a not in answer0_set:
        #     res_answers.append([answer_a])
        #
        # if '' != answer_b and answer_b not in answer0_set:
        #     res_answers.append([answer_b])

        res_answers.append(answer)

    if len(res_answers) == 0:
        print('qid', question_id, 'No answer!')
        return answers

    return res_answers


def split_a_and_b(answers):
    if len(answers) < 3:
        return answers

    # found = False

    additional_answers = list()
    for answer in answers:
        # print(ans)

        if ' and ' not in answer[0]:
            continue

        delm_start = answer[0].find(' and ')

        answer_a = answer[0][:delm_start]
        answer_b = answer[0][delm_start+5:]

        # print('answer', answer[0])
        # print('answer_a', answer_a)
        # print('answer_b', answer_b)

        if '' != answer_a.strip():
            additional_answers.append(answer_a.strip())
        if '' != answer_b.strip():
            additional_answers.append(answer_b.strip())

        # if a in answers and b in answers:
        #     found = True

    for added_answer in additional_answers:
        found_exist = False
        for answer in answers:
            if added_answer == answer[0]:
                found_exist = True
                break

        if found_exist:
            continue

        answers.append([added_answer])

    return answers


def split_all(answers, question_id):
    res_answers = list()
    answer0_set = set()

    for answer in answers:
        if ' and/or ' not in answer[0] \
                and ' and ' not in answer[0] \
                and ', ' not in answer[0] \
                and '/' not in answer[0]:
            answer0_set.add(answer[0].lower())
            res_answers.append(answer)
            continue

        if ' and/or ' in answer[0]:
            snames = get_split_names(answer[0], ' and/or ')
            if len(snames) > 0:
                print('qid', question_id, 'Split', answer[0], '->', snames)
                for n in snames:
                    if n.lower() in answer0_set:
                        continue
                    answer0_set.add(n.lower())
                    res_answers.append([n])
        elif ' and ' in answer[0]:
            snames = get_split_names(answer[0], ' and ')
            if len(snames) > 0:
                print('qid', question_id, 'Split', answer[0], '->', snames)
                for n in snames:
                    if n.lower() in answer0_set:
                        continue
                    answer0_set.add(n.lower())
                    res_answers.append([n])
        elif ', ' in answer[0]:
            snames = get_split_names(answer[0], ', ')
            if len(snames) > 0:
                print('qid', question_id, 'Split', answer[0], '->', snames)
                for n in snames:
                    if n.lower() in answer0_set:
                        continue
                    answer0_set.add(n.lower())
                    res_answers.append([n])
        elif '/' in answer[0]:
            snames = get_split_names(answer[0], '/')
            if len(snames) > 0:
                print('qid', question_id, 'Split', answer[0], '->', snames)
                for n in snames:
                    if n.lower() in answer0_set:
                        continue
                    answer0_set.add(n.lower())
                    res_answers.append([n])

    return res_answers


def get_split_names(super_answer, delm):
    paren_start = super_answer.find('(')
    if -1 < paren_start:
        paren_end = super_answer.find(')')
        if -1 < paren_end:
            abbr = ''
            for c in super_answer[:paren_start]:
                if ord('A') <= ord(c) <= ord('Z'):
                    abbr += c
            print('Found paren:', super_answer[paren_start+1:paren_end],
                  'abbreviation:', abbr)
            if abbr == super_answer[paren_start+1:paren_end]:
                return [super_answer]

    split_names = list()
    names = super_answer.split(delm)
    for nm in names:

        if nm and nm[-1] in '()':
            nm = nm[:-1]

        if nm and nm[0] in '()':
            nm = nm[1:]

        nm = nm.strip()
        if not nm:
            continue

        if ' (' in nm:
            if 1 < len(nm.split(' (')):
                split_names.extend(nm.split(' ('))
                continue
        if ') ' in nm:
            if 1 < len(nm.split(') ')):
                split_names.extend(nm.split(') '))
                continue

        split_names.append(nm)
    return split_names


def split_w_slash(answers):
    answer0_set = set()
    for answer in answers:
        answer0_set.add(answer[0].lower())

    res_answers = list()
    for answer in answers:
        if '/' not in answer[0]:
            res_answers.append(answer)
            continue

        snames = get_split_names(answer[0], '/')
        if snames:
            print('Split', answer[0], '->', snames)
            for n in snames:
                if n.lower() in answer0_set:
                    continue
                res_answers.append([n])


def remove_suffix(answers, question_id):
    res_answers = list()
    valid_ans_set = set()
    for answer in answers:
        if answer[0][-1] in ',®':
            n = answer[0][:-1]
            print('qid', question_id, 'Remove suffix', answer[0], '->', n)
        else:
            n = answer[0]
        if n not in valid_ans_set:
            res_answers.append([n])
            valid_ans_set.add(n)
    return res_answers


def softmax(probs):
    exp_probs = np.exp(probs)
    return exp_probs / np.sum(exp_probs)


# Setting basic strings
# Info : This script is only for factoid question


# Checking nbest_BioASQ-test prediction.json
if not os.path.exists(args.nbest_path):
    print("No file exists!\n#### Fatal Error : Abort!")
    raise ValueError("No file exists!: " + args.nbest_path)

# Reading Pred File
with open(args.nbest_path, "r") as reader:
    nbest_pred_dict = json.load(reader)

qidDict = dict()
qidQtextDict = dict()

for multiQid in nbest_pred_dict:
    assert len(multiQid) == (24+4)

    realQid = multiQid[:-4]

    # each snippet
    merged_q_answers = list()

    for a in nbest_pred_dict[multiQid]:

        # check ends with , or .
        # check ()
        if filter_incomplete(a['text']):
            # print('\tFilter', a['text'], sep='\t')
            continue

        found_same_answer = False

        for ma in merged_q_answers:
            if a['text'].lower() != ma['text'].lower():
                continue

            found_same_answer = True

            if args.verbose:
                print('\tSUM prob in a snippet:', ma['text'],
                      '{:.4f} -> {:.4f}'.format(
                          ma['probability'],
                          ma['probability'] + a['probability']))

            ma['probability'] += a['probability']
            ma['freq'] += 1
            break

        if not found_same_answer:
            a['freq'] = 1
            merged_q_answers.append(a)

    if realQid not in qidDict:
        qidDict[realQid] = [merged_q_answers]
        q_text = merged_q_answers[0]['question_text']
        found_num = find_num(q_text)
        if found_num is not None:
            print('Found', found_num, 'qid', realQid, 'q_text', q_text)
        qidQtextDict[realQid] = (q_text, found_num)
    else:
        qidDict[realQid].append(merged_q_answers)


entryList = list()
n_answersList = list()
for qid in qidDict:
    jsonList = list()
    answer_set = set()
    for jsonele in qidDict[qid]:  # value of qidDict is a list of list

        # Merge prob
        for e in jsonele:

            found = e['text'].lower() in answer_set

            if found:
                for aa in jsonList:

                    if e['text'].lower() != aa['text'].lower():
                        continue

                    if args.verbose:
                        print('\t\tSUM prob inter snippets', aa['text'],
                              '{:.4f} -> {:.4f}'.format(
                                  aa['probability'],
                                  aa['probability'] + e['probability']))

                    aa['probability'] += e['probability']
                    if 'snp_freq' in aa:
                        aa['snp_freq'] += 1
                    else:
                        aa['snp_freq'] = 1
            else:
                if 'snp_freq' in e:
                    e['snp_freq'] += 1
                else:
                    e['snp_freq'] = 1
                jsonList.append(e)
                answer_set.add(e['text'].lower())

    # # snp avg. prob
    # for aa in jsonList:
    #
    #     if aa['snp_freq'] == 1:
    #         continue
    #
    #     print('\tAVG prob inter snp', aa['text'],
    #           aa['probability'], '->',
    #           aa['probability'] / aa['snp_freq'])
    #
    #     aa['probability'] /= aa['snp_freq']

    assert len(jsonList) > 0

    if args.use_softmax:
        prob_list = [aa['probability'] for aa in jsonList]
        final_probs = softmax(prob_list)
        for fp, aa in zip(final_probs, jsonList):
            aa['probability'] = fp

    # print(len(jsonList))  # debug

    # if not args.multi_output:
    qidDf = pd.DataFrame(jsonList)
    # else: # args.multi_output==True

    # for factoid
    """
    fullAnsList=qidDf.sort_values(
    by='probability', axis=0, ascending=False)['text'].tolist() 
    ansList=[]
    for ans in fullAnsList:
        if ans not in ansList:
            ansList.append(ans)
        if len(ansList)==5:
            break
    """

    _, found_num = qidQtextDict[qid]
    if found_num is not None:
        descendList = sorted(jsonList, key=itemgetter('probability'),
                             reverse=True)
        lowercaseSet = set()  # wj
        tempList = list()  # wj
        for ans in descendList:  # wj

            if ans['probability'] <= threshold:
                continue

            ans['text'] = preprocess_phrase(ans['text'])
            if ans['text'].lower() not in lowercaseSet:
                lowercaseSet.add(ans['text'].lower())
                tempList.append(ans)
            if len(tempList) == found_num:
                break

        if len(tempList) == 0:
            if len(descendList) > 0:
                firstAns = descendList[0]
                firstAns['text'] = preprocess_phrase(firstAns['text'])
                tempList.append(firstAns)
            else:
                print('No answer!!!', qid)

        descendList = tempList  # wj
        ansList = [a['text'] for a in descendList]
    else:
        # fullAnsList = \
        #     set(qidDf[qidDf['probability'] > args.threshold]['text'].tolist())
        has_freq = 'freq' in qidDf
        fullAnsList = \
            set(qidDf[(qidDf['probability'] / qidDf['freq'] if has_freq
                       else qidDf['probability']) > threshold][
                    'text'].tolist())
        ansList = [a for a in fullAnsList]

        if len(ansList) == 0:
            descendList = sorted(jsonList, key=itemgetter('probability'),
                                 reverse=True)
            if len(descendList) > 0:
                ansList = [preprocess_phrase(descendList[0]['text'])]
            else:
                print('No answer!!!', qid)
                ansList = []
        elif len(ansList) == 1:
            ansList[0] = preprocess_phrase(ansList[0])
        else:
            lowercaseSet = set()
            tempList = list()
            for ans in ansList:
                ans = preprocess_phrase(ans)
                if ans.lower() in lowercaseSet:
                    continue
                lowercaseSet.add(ans.lower())
                tempList.append(ans)
            ansList = tempList

    assert len(qid) == 24

    assert len(ansList) > 0

    exact_answer = [[ans] for ans in ansList if ans != " "]

    exact_answer = resolve_a_and_b(exact_answer, qid)
    exact_answer = split_all(exact_answer, qid)
    exact_answer = remove_suffix(exact_answer, qid)

    # delete too long answers
    exact_answer = [a for a in exact_answer
                    if args.len_limit >= len(a[0].split())]

    assert len(exact_answer) > 0

    exact_answer = sorted(exact_answer, key=lambda ea: ea[0])

    entry = {
        u"type": "list",
        # u"body":qas,
        u"id": qid,  # must be 24 char
        u"ideal_answer": ["Dummy"],
        u"exact_answer": exact_answer,
    }
    entryList.append(entry)
    n_answersList.append(len(exact_answer))

# sort by question id
entryList = sorted(entryList, key=itemgetter(u"id"))

print('#questions', len(entryList))
if 0 < len(n_answersList):
    print('avg #answers {:.1f}'.format(sum(n_answersList) / len(n_answersList)))

with open(args.output_path, "w") as outfile:
    json.dump({u'questions': entryList}, outfile, indent=2, sort_keys=True)
