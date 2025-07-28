import argparse
import json
import os
import re
from collections import defaultdict, Counter
import requests
import time
import pickle
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import numpy

def wiki_check(rel, option):
    url = 'https://query.wikidata.org/sparql'
    query1 = '''
    SELECT ?item ?value ?valueLabel
    {
      ?item wdt:'''
    query2 = ''' ?value.
      ?value ?label "'''
    query3 = '''"@en .

      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
    }
    LIMIT 1
    '''
    query = query1+rel+query2+option.strip()+query3
    r = requests.get(url, params = {'format': 'json', 'query': query})
    data = r.json()
    time.sleep(1)
    if len(data['results']['bindings']) == 1: return True
    else: return False
    
    
def wiki_subject_check(sub):
    url = 'https://query.wikidata.org/sparql'
    query1 = '''
    SELECT ?item ?value ?valueLabel
    {
      ?item (wdt:P19|wdt:P20|wdt:P27|wdt:P101|wdt:P1376|wdt:P740|wdt:P495) ?value.
      ?item ?label "'''
    query2 = '''"@en .

      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
    }
    LIMIT 1
    '''
    query = query1+sub.strip()+query2
    r = requests.get(url, params = {'format': 'json', 'query': query})
    data = r.json()
    time.sleep(1)
    if len(data['results']['bindings']) == 1: return True
    else: return False
    

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

    
def main(data_folder, output_folder):
    SUBJECTS = ["DND_human_male", "DND_human_female", "Russian", "AnimeAndManga", 
            "Books", "Newspapers", "Magazines", "Town_Central_Africa", 
            "Town_Central_America", "Town_Central_Asia", "Town_East_Asia", 
            "Town_East_Europe", "Town_Middle_Eastern", "Town_West_Europe", "French", 
            "German", "Korean", "Japanese", "Music_group", "Company"]
    for source in SUBJECTS:
        print(source)
        subjects_all = []
        real=[]
        with open(f"{data_folder}/{source}.pickle", "rb")  as f:
            subjects_all = pickle.load(f)

        for n in tqdm(subjects_all): 
            if wiki_subject_check(n):
                real.append(n)
        with open(f'{output_folders}/{source}_real.txt', 'w+') as outputfile:
            for i in real:
                outputfile.write(i)
                outputfile.write('\n')
                
                
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
    