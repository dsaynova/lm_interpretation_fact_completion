import json
from collections import defaultdict, Counter
import requests
import time
import pickle
import argparse

def wiki_label_from_Q(q):
    url = 'https://query.wikidata.org/sparql'
    query1 = '''
    SELECT ?valueLabel
    WHERE
    {
    wd:'''
    query2=''' wdt:P31 ?value.
    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
    } 
    LIMIT 1  '''
    query = query1+q+query2
    r = requests.get(url, params = {'format': 'json', 'query': query})
    data = r.json()
    time.sleep(1)
    if len(data['results']['bindings']) == 1: return data['results']['bindings'][0]['valueLabel']['value']
    else: return "not found"

def read_jsonl_file(filename: str):
    dataset = []
    with open(filename) as f:
        for line in f:
            loaded_example = json.loads(line)
            dataset.append(loaded_example)
    return dataset

def main(data_folder, output_folder):
    res = read_jsonl_file(f"{data_folder}/trex/P495.jsonl")
    p495_subjects = Counter()
    for r in res:
        label = wiki_label_from_Q(r["sub_uri"])
        p495_subjects[label]+=1

    remain=0
    for k, v in p495_subjects.items():
        if v>10: print(k, v)
        else: remain+=1
    print("Remain: ", remain)
    with open(f"{output_folder}/p495_subject_types.pickle", 'wb') as outputfile:
        pickle.dump(p495_subjects, outputfile)
    
    
    
    res = read_jsonl_file(f"{data_folder}/trex/P740.jsonl")
    p740_subjects = Counter()
    for r in res:
        label = wiki_label_from_Q(r["sub_uri"])
        p740_subjects[label]+=1

    remain=0
    for k, v in p740_subjects.items():
        if v>10: print(k, v)
        else: remain+=1
    print("Remain: ", remain)
    with open(f"{output_folder}/P740_subject_types.pickle", 'wb') as outputfile:
        pickle.dump(p740_subjects, outputfile)
                
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder", type=str
    )
    parser.add_argument(
        "--output_folder", type=str
    )

    args = parser.parse_args()

    main(
        args.data_folder,
        args.output_folder
    )
    