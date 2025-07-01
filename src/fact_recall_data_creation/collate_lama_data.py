import argparse
import pandas as pd
import os

RELATIONS = {"P19": "place of birth",
            "P20": "place of death",
            "P27": "country of citizenship",
            "P101": "field of work",
            "P495": "country of origin",
            "P740": "location of formation",
            "P1376": "capital of"}

REL_TO_GOOGLE_RE = {"P19": "place_of_birth_test",
                    "P20": "place_of_death_test"}

def collate_data(args):
    data = pd.DataFrame()

    print("Processing TREx data...")
    # read TREX data
    for relation in RELATIONS:
        tmp_data = pd.read_json(os.path.join(args.srcdir_trex, relation+".jsonl"), lines=True)
        tmp_data["evidences"] = tmp_data.evidences.apply(lambda val: [ev["masked_sentence"].replace("[MASK]", ev["obj_surface"]) for ev in val])
        tmp_data["source"] = "TREx_UHN"
        data = pd.concat((data, tmp_data), ignore_index=True)
        
    print("Processing Google RE data...")
    # read Google RE data
    for rel_key, rel_val in REL_TO_GOOGLE_RE.items():
        tmp_data = pd.read_json(os.path.join(args.srcdir_google_re, rel_val+".jsonl"), lines=True)
        tmp_data = tmp_data[["evidences", "sub_w", "sub_label", "obj_w", "obj_label", "obj_aliases", "uuid"]]
        tmp_data = tmp_data.rename(columns={"sub_w": "sub_uri", "obj_w": "obj_uri"})
        tmp_data["predicate_id"] = rel_key
        tmp_data["evidences"] = tmp_data.evidences.apply(lambda ev_list: [val["considered_sentences"] for val in ev_list])
        tmp_data["source"] = "Google_RE_UHN"
        data = pd.concat((data, tmp_data), ignore_index=True)
        
    return data
    
def main(args):
    data = collate_data(args)
    print("Data collated!")
    
    data.to_json(args.output_file, lines=True, orient="records")
    print("Data saved!")
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--srcdir_trex",
        required=True,
        type=str,
        help="Source directory. Should be TREx_alpaca.",
    )
    argparser.add_argument(
        "--srcdir_google_re",
        required=True,
        type=str,
        help="Source directory. Should be Google_RE.",
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