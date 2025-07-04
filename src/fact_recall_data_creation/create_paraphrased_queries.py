import argparse
import pandas as pd
import json

PARAPHRASES = {"P19": ["[X] was born in [Y]",
                       "[X] is originally from [Y]",
                       "[X] was originally from [Y]",
                       "[X] originated from [Y]",
                       "[X] originates from [Y]"
                       ],
               "P20": ["[X] died in [Y]",
                       "[X] died at [Y]",
                       "[X] passed away in [Y]",
                       "[X] passed away at [Y]",
                       "[X] expired at [Y]",
                       "[X] lost their life at [Y]",
                       "[X]'s life ended in [Y]",
                       "[X] succumbed at [Y]"
                       ],
               "P27": ["[X] is a citizen of [Y]",
                       "[X], a citizen of [Y]",
                       "[X], who is a citizen of [Y]",
                       "[X] holds a citizenship of [Y]",
                       "[X] has a citizenship of [Y]",
                       "[X], who holds a citizenship of [Y]",
                       "[X], who has a citizenship of [Y]"
                       ],
               "P101": ["[X] works in the field of [Y]",
                        "[X] specializes in [Y]",
                        "The expertise of [X] is [Y]",
                        "The domain of activity of [X] is [Y]",
                        "The domain of work of [X] is [Y]",
                        "[X]'s area of work is [Y]",
                        "[X]'s domain of work is [Y]",
                        "[X]'s domain of activity is [Y]",
                        "[X]'s expertise is [Y]",
                        "[X] works in the area of [Y]"
                        ],
               "P495": ["[X] was created in [Y]",
                        "[X], that was created in [Y]",
                        "[X], created in [Y]",
                        "[X], that originated in [Y]",
                        "[X] originated in [Y]",
                        "[X] formed in [Y]",
                        "[X] was formed in [Y]",
                        "[X], that was formed in [Y]",
                        "[X] was formulated in [Y]",
                        "[X], formulated in [Y]",
                        "[X], that was formulated in [Y]",
                        "[X] was from [Y]",
                        #"[X], who was from [Y]" # subjects are not people
                        "[X], from [Y]",
                        "[X], that was developed in [Y]",
                        "[X] was developed in [Y]",
                        "[X], developed in [Y]"
                        ],
               "P740": ["[X] was founded in [Y]",
                        "[X], founded in [Y]",
                        "[X] that was founded in [Y]",
                        "[X], that was started in [Y]",
                        "[X] started in [Y]",
                        "[X] was started in [Y]",
                        "[X], that was created in [Y]",
                        "[X], created in [Y]",
                        "[X] was created in [Y]",
                        "[X], that originated in [Y]",
                        "[X] originated in [Y]",
                        "[X] formed in [Y]",
                        "[X] was formed in [Y]",
                        "[X], that was formed in [Y]"
                        ],
               "P1376": ["[X] is the capital of [Y]",
                         "[X] is the capital city of [Y]",
                         "[X], the capital of [Y]",
                         "[X], the capital city of [Y]",
                         "[X], that is the capital of [Y]",
                         "[X], that is the capital city of [Y]"
                         ]}
        
def parse_prompt(prompt, subject_label):
    SUBJ_SYMBOL = '[X]'
    OBJ_SYMBOL = '[Y]'
    # do not add an extra space before the object to be predicted
    prompt = prompt.replace(SUBJ_SYMBOL, subject_label)\
                   .replace(" "+OBJ_SYMBOL, "")
    return prompt

def generate_data(data, paraphrases, ouptut_data_path):
    f_true = open(ouptut_data_path, "w")
    for i, d in data.iterrows():
        for paraphrase in paraphrases[d.predicate_id]:
            dict_results = d.to_dict()
            dict_results["prompt"] = parse_prompt(paraphrase, d["sub_label"])
            dict_results["template"] = paraphrase
            f_true.write(json.dumps(dict_results))
            f_true.write("\n")
    f_true.close()
    
def main(args):
    data = pd.read_json(args.srcfile, lines=True)
    
    for rel, templates in PARAPHRASES.items():
        assert len(set(templates))==len(templates), "There should be no duplicates in the PARAPHRASES data"
        for template in templates:
            assert template[-3:]=="[Y]", "The template must suit ARMs"
    
    queries = generate_data(data, PARAPHRASES, args.outfile)
    print("Data saved!")
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--srcfile",
        required=True,
        type=str,
        help="Source jsonl file.",
    )
    argparser.add_argument(
        "--outfile",
        required=True,
        type=str,
        help="File to save filtered data to.",
    )
    args = argparser.parse_args()

    print(args)
    main(args)