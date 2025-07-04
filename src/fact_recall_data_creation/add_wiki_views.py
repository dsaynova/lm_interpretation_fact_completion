import argparse
import pandas as pd
from tqdm import tqdm
import requests

def add_wiki_views(args, data):
    # Customize this to your person. Used for the Wikipedia pageviews API.
    HEADERS = {
    'User-Agent': 'popularity/wikidata5m (email)'
    }
    # Wikipedia 2019 popularity rates are quite relevant as e.g. T-REx is from 2018. This query is formatted to that.
    REQUEST_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/INPUT/monthly/20190101/20191228"

    def get_avg_monthly_views(response):
        assert len(response['items'])==12, "Should have monthly views for a whole year."
        tot_views = 0
        for item in response['items']:
            tot_views += item["views"]
        return tot_views/12
    
    print("Collecting views for subjects...")
    all_subjects = data.sub_label.unique()
    sub_views = {}
    for subject in tqdm(all_subjects):
        request_url = REQUEST_URL.replace("INPUT", subject.replace(" ", "_"))
        response = requests.get(request_url, headers=HEADERS)
        tmp_sub_views = None
        if response.ok and len(response.json()["items"])==12:
            tmp_sub_views = get_avg_monthly_views(response.json())
        sub_views[subject] = tmp_sub_views
        
    data["sub_view_rates"] = data.sub_label.apply(lambda val: sub_views[val])

    print("Collecting views for objects...")
    # get views for objects
    all_objects = data.obj_label.unique()
    obj_views = {}
    for o in tqdm(all_objects):
        # the wikipedia page view API is case sensitive
        # make sure that all objects are capitalized, e.g. "algebra" should be "Algebra"
        request_url = REQUEST_URL.replace("INPUT", o.capitalize().replace(" ", "_"))
        response = requests.get(request_url, headers=HEADERS)
        tmp_obj_views = None
        if response.ok and len(response.json()["items"])==12:
            tmp_obj_views = get_avg_monthly_views(response.json())
        obj_views[o] = tmp_obj_views
        
    data["obj_view_rates"] = data.obj_label.apply(lambda val: obj_views[val])
    
    return data
    
def main(args):
    print("Collecting Wikipedia page views...")
    data = pd.read_json(args.srcfile, lines=True)
    data = add_wiki_views(args, data)
    print("Wikipedia page views added!")
    
    data.to_json(args.output_file, lines=True, orient="records")
    print("Data saved!")
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--srcfile",
        required=True,
        type=str,
        help="Source file.",
    )
    argparser.add_argument(
        "--output_file",
        required=True,
        type=str,
        help="Filename to save processed data to.",
    )
    args = argparser.parse_args()

    print(args)
    main(args)