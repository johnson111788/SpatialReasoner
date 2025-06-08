import os

import numpy as np
import pandas as pd

################
dataset_name = '3DSRBench'
results_path = './VLMEvalKit/outputs'
results_file = f'results_{dataset_name}.csv'
################

LABELS = ['A', 'B', 'C', 'D']
mapping = {
    'location': ['location_above', 'location_closer_to_camera', 'location_next_to'],
    'height': ['height_higher'],
    'orientation': ['orientation_in_front_of', 'orientation_on_the_left', 'orientation_viewpoint'],
    'multi_object': ['multi_object_closer_to', 'multi_object_facing', 'multi_object_viewpoint_towards_object', 'multi_object_parallel', 'multi_object_same_direction']}
types = ['height', 'location', 'orientation', 'multi_object']
subtypes = sum([mapping[k] for k in types], [])

file_mapping = {}
for model in os.listdir(results_path):
    file = os.path.join(results_path, model, f'{model}_{dataset_name}_openai_result.xlsx')
    if os.path.isfile(file):
        file_mapping[model] = file

if os.path.isfile(results_file):
    saved_df = pd.read_csv(results_file)
    results_csv = []
    for idx, row in saved_df.iterrows():
        results_csv.append(row.tolist())
else:
    results_csv = []

# Compute model results
for model in sorted(list(file_mapping.keys())):
    found = False
    for m in results_csv:
        if m[0] == model:
            print(f"Skipping {model} as it is already computed.")
            found = True
    if found:
        continue

    file = file_mapping[model]
    df = pd.read_excel(file)

    results = {}
    for i in range(len(df.index)):
        row = df.iloc[i].tolist()

        assert row[12] in [0, 1], row[12]

        if row[1][-2] == '-':
            qid = row[1][:-2]
        else:
            qid = row[1]

        if qid in results:
            results[qid][0] = results[qid][0] * row[12]
        else:
            results[qid] = [row[12], row[8]]

        assert row[8] in subtypes, row[8]

    curr_results = [np.mean([results[k][0] for k in results])]
    # print(len([results[k][0] for k in results]))
    for t in types:
        # print(t, len([results[k][0] for k in results if results[k][1] in mapping[t]]))
        curr_results.append(np.mean([results[k][0] for k in results if results[k][1] in mapping[t]]))
    for t in subtypes:
        curr_results.append(np.mean([results[k][0] for k in results if results[k][1] == t]))
    # exit()

    curr_results = [model] + [np.round(v*100, decimals=1) for v in curr_results]

    results_csv.append(curr_results)

# Compute a random baseline
found = False
for m in results_csv:
    if m[0] == 'random':
        print(f"Skipping {model} as it is already computed.")
        found = True
if not found:
    file = file_mapping[model]
    df = pd.read_excel(file)
    results = {}
    for i in range(len(df.index)):
        row = df.iloc[i].tolist()
        assert row[12] in [0, 1], row
        if row[1][-2] == '-':
            qid = row[1][:-2]
        else:
            qid = row[1]
        if isinstance(row[4], float):
            hit = int(np.random.randint(2) == 0)
        else:
            hit = int(np.random.randint(4) == 0)
        if qid in results:
            results[qid][0] = results[qid][0] * hit
        else:
            results[qid] = [hit, row[8]]
        assert row[8] in subtypes, row[8]
    curr_results = [np.mean([results[k][0] for k in results])]
    for t in types:
        curr_results.append(np.mean([results[k][0] for k in results if results[k][1] in mapping[t]]))
    for t in subtypes:
        curr_results.append(np.mean([results[k][0] for k in results if results[k][1] == t]))
    curr_results = ['random'] + [np.round(v*100, decimals=1) for v in curr_results]
    results_csv.append(curr_results)

pd.DataFrame(columns=['model', 'overall']+types+subtypes, data=results_csv).to_csv(results_file, index=False)
