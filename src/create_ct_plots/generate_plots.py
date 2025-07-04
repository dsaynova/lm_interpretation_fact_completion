import argparse
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import math

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

LABELS = [
    "First subject token",
    "Middle subject tokens",
    "Last subject token",
    "First subsequent token",
    "Further tokens",
    "Last token",
]

LINEPLOT_HIGH_SCORE = {"gpt2-xl": 0.42,
                       "llama2_7B": 0.34,
                       "llama2_13B": 0.34
                       }
LINEPLOT_LOW_SCORE = {"gpt2-xl": -0.02,
                      "llama2_7B": -0.02,
                      "llama2_13B": -0.02,
                      }

NORM_LINEPLOT_HIGH_SCORE = {"gpt2-xl": 0.4,
                            "llama2_7B": 0.7,
                            "llama2_13B": 0.7,
                            }
NORM_LINEPLOT_LOW_SCORE = {"gpt2-xl": -0.12,
                           "llama2_7B": -0.1,
                           "llama2_13B": -0.1,
                           }

class Avg:
    def __init__(self):
        self.d = []

    def add(self, v):
        self.d.append(v[None])

    def add_all(self, vv):
        self.d.append(vv)

    def avg(self):
        return np.concatenate(self.d).mean(axis=0)

    def std(self):
        return np.concatenate(self.d).std(axis=0)

    def size(self):
        return sum(datum.shape[0] for datum in self.d)

def read_knowledge(queries, savefolder, do_normalize=False, do_cap=False):
    (
        avg_fe,
        avg_ee,
        avg_le,
        avg_fa,
        avg_ea,
        avg_la,
        avg_hs,
        avg_ls,
        avg_fs,
        avg_fle,
        avg_fla,
    ) = [Avg() for _ in range(11)]
    num_neg_te = 0
    num_scores_above_one = 0
    num_scores_below_minus_one = 0
    num_capped_samples = 0
    
    for _, row in tqdm(queries.iterrows(), total=len(queries)):
        data = np.load(os.path.join(row.CT_results_dir, row.filename_template.format(row.known_id)))
        # old: Only consider cases where the model begins with the correct prediction
        if "correct_prediction" in data and not data["correct_prediction"]:
            print("Error: Sample is marked as 'not correct prediction' and skipped for the plots")
            continue
    
        scores = data["scores"].squeeze()
        if do_normalize:
            scores = (scores-data["low_score"].squeeze())/abs(data["high_score"].squeeze()-data["low_score"].squeeze())
        
        # get statistics for problematic scores
        sample_capped = False
        if (data["high_score"].squeeze()-data["low_score"].squeeze()) < 0:
            num_neg_te += 1
        if np.any(scores>1):
            num_scores_above_one +=1
            sample_capped = True
        if np.any(scores<-1):
            num_scores_below_minus_one +=1
            sample_capped = True
        if sample_capped:
            num_capped_samples += 1
            
        if do_cap:
            scores = np.clip(scores, -1, 1)
            
        first_e, first_a = data["subject_range"]
        last_e = first_a - 1
        last_a = len(scores) - 1
        # original prediction
        avg_hs.add(data["high_score"].squeeze())
        # prediction after subject is corrupted
        avg_ls.add(data["low_score"].squeeze())
        avg_fs.add(scores.max())
        # some maximum computations
        avg_fle.add(scores[last_e].max())
        avg_fla.add(scores[last_a].max())
        # First subject middle, last subjet.
        avg_fe.add(scores[first_e])
        avg_ee.add_all(scores[first_e + 1 : last_e])
        avg_le.add(scores[last_e])
        # First after, middle after, last after
        avg_fa.add(scores[first_a])
        avg_ea.add_all(scores[first_a + 1 : last_a])
        avg_la.add(scores[last_a])

    result = np.stack(
        [
            avg_fe.avg(),
            avg_ee.avg(),
            avg_le.avg(),
            avg_fa.avg(),
            avg_ea.avg(),
            avg_la.avg(),
        ]
    )
    result_std = np.stack(
        [
            avg_fe.std(),
            avg_ee.std(),
            avg_le.std(),
            avg_fa.std(),
            avg_ea.std(),
            avg_la.std(),
        ]
    )
    
    norm_extra = "_norm" if do_normalize else ""
    stats_filename = f"read_knowledge_stats{norm_extra}.txt"
    stats_filename = os.path.join(savefolder, stats_filename)
    with open(stats_filename, "w+") as f:
        f.write(f"Average Total Effect: {avg_hs.avg() - avg_ls.avg()}\n")
        f.write(
            f"Best average indirect effect on last subject: {avg_le.avg().max() - avg_ls.avg()}\n"
        )
        f.write(
            f"Best average indirect effect on last token: {avg_la.avg().max() - avg_ls.avg()}\n"
        )
        f.write(f"Average best-fixed score: {avg_fs.avg()}\n")
        f.write(f"Average best-fixed on last subject token score: {avg_fle.avg()}\n")
        f.write(f"Average best-fixed on last word score: {avg_fla.avg()}\n")
        f.write(f"Argmax at last subject token: {np.argmax(avg_le.avg())}\n")
        f.write(f"Max at last subject token: {np.max(avg_le.avg())}\n")
        f.write(f"Argmax at last prompt token: {np.argmax(avg_la.avg())}\n")
        f.write(f"Max at last prompt token: {np.max(avg_la.avg())}\n")
        f.write(f"Number of samples with negative TE: {num_neg_te} (of a total {len(queries)} samples)\n")
        f.write(f"Number of samples with restored p (or proportion of p) above 1: {num_scores_above_one} (of a total {len(queries)} samples)\n")
        f.write(f"Number of samples with restored p (or proportion of p) below -1: {num_scores_below_minus_one} (of a total {len(queries)} samples)\n")
        f.write(f"Number of samples that have been capped for capped plot: {num_capped_samples} (of a total {len(queries)} samples)\n")
        
    print(f"Data stats saved to '{stats_filename}'!\n")
    
    return dict(
        low_score=avg_ls.avg(), result=result, result_std=result_std, size=avg_fe.size()
    )
    
def create_plots(data, kind, count, arch, archname, savefolder):
    if os.path.exists(savefolder):
        print(f"{savefolder} already exists! Skipping... (remove it if you want to generate it anew)")
        return
    else:
        os.makedirs(savefolder, exist_ok=False)
    
    print("Generating plots for CT results...")
    d = read_knowledge(data, savefolder)

    # get 2D heatmap    
    result = np.clip(d["result"] - d["low_score"], 0, None)
    plot_array(
        np.squeeze(result),
        kind=kind,
        title=None,
        low_score=0.0,
        high_score=None,
        archname=archname,
        savepdf=os.path.join(savefolder, "all_ranks.pdf"),
    )
    print("Average CT results heatmap saved!")
        
    # get lineplot without axis limits
    result = d["result"] - d["low_score"]
    result_std = d["result_std"]
    make_line_plot(
        np.squeeze(result),
        np.squeeze(result_std),
        low_score=None,
        high_score=None,
        count=count,
        archname=archname,
        savepdf=os.path.join(savefolder, "line_plot_all_ranks_free_lim.pdf"),
    )
    # with axis limits
    make_line_plot(
        np.squeeze(result),
        np.squeeze(result_std),
        low_score=LINEPLOT_LOW_SCORE[arch],
        high_score=LINEPLOT_HIGH_SCORE[arch],
        count=count,
        archname=archname,
        savepdf=os.path.join(savefolder, "line_plot_all_ranks.pdf"),
    )
    
    # get lineplot that measures direct probability
    result = d["result"]
    result_std = d["result_std"]
    make_line_plot(
        np.squeeze(result),
        np.squeeze(result_std),
        low_score=LINEPLOT_LOW_SCORE[arch],
        high_score=LINEPLOT_HIGH_SCORE[arch],
        count=count,
        archname=archname,
        savepdf=os.path.join(savefolder, "line_plot_prob_all_ranks.pdf"),
    )
    print("Average CT results line plot saved!")
    print()
    
    # normalized results
    print("Generating plots for normalized CT results...")
    norm_d = read_knowledge(data, savefolder, do_normalize=True)
    
    # get 2D heatmap
    plot_array(
        np.squeeze(norm_d["result"]),
        kind=kind,
        title=None,
        low_score=0.0,
        high_score=None,
        archname=archname,
        savepdf=os.path.join(savefolder, "all_ranks_norm.pdf"),
        cbar_title="NAIE"
    )
    print("Normalized average CT results heatmap saved!")
    
    # get lineplot without axis limits
    result = norm_d["result"]
    result_std = norm_d["result_std"]
    make_line_plot(
        np.squeeze(result),
        np.squeeze(result_std),
        low_score=None,
        high_score=None,
        count=count,
        archname=archname,
        savepdf=os.path.join(savefolder, "norm_line_plot_all_ranks_free_lim.pdf"),
        ylabel="Normalized AIE"
    )
    # with axis limits
    make_line_plot(
        np.squeeze(result),
        np.squeeze(result_std),
        low_score=NORM_LINEPLOT_LOW_SCORE[arch],
        high_score=NORM_LINEPLOT_HIGH_SCORE[arch],
        count=count,
        archname=archname,
        savepdf=os.path.join(savefolder, "norm_line_plot_all_ranks.pdf"),
        ylabel="Normalized AIE"
    )
    print("Normalized average CT results line plot saved!")
    
    # normalized results - capped between [0,1]
    print("Generating plots for capped normalized CT results...")
    norm_d = read_knowledge(data, savefolder, do_normalize=True, do_cap=True)
    
    # get 2D heatmap
    plot_array(
        np.squeeze(norm_d["result"]),
        kind=kind,
        title=None,
        low_score=0.0,
        high_score=None,
        archname=archname,
        savepdf=os.path.join(savefolder, "all_ranks_norm_capped.pdf"),
        cbar_title="NAIE"
    )
    print("Normalized average CT results heatmap saved!")
    
    # get lineplot without axis limits
    result = norm_d["result"]
    result_std = norm_d["result_std"]
    make_line_plot(
        np.squeeze(result),
        np.squeeze(result_std),
        low_score=None,
        high_score=None,
        count=count,
        archname=archname,
        savepdf=os.path.join(savefolder, "norm_line_plot_all_ranks_free_lim_capped.pdf"),
        ylabel="Normalized AIE"
    )
    # with axis limits
    make_line_plot(
        np.squeeze(result),
        np.squeeze(result_std),
        low_score=NORM_LINEPLOT_LOW_SCORE[arch],
        high_score=NORM_LINEPLOT_HIGH_SCORE[arch],
        count=count,
        archname=archname,
        savepdf=os.path.join(savefolder, "norm_line_plot_all_ranks_capped.pdf"),
        ylabel="Normalized AIE"
    )
    print("Normalized average CT results line plot saved!")

def plot_array(
    differences,
    kind=None,
    savepdf=None,
    title=None,
    low_score=None,
    high_score=None,
    archname="GPT2-XL",
    cbar_title="AIE"
):
    if low_score is None:
        low_score = differences.min()
    if high_score is None:
        high_score = differences.max()

    fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
    h = ax.pcolor(
        differences,
        cmap={None: "Purples", "mlp": "Greens", "attn": "Reds"}[kind],
        vmin=low_score,
        vmax=high_score,
    )
    if title:
        ax.set_title(title)
    ax.invert_yaxis()
    ax.set_yticks([0.5 + i for i in range(len(differences))])
    ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
    ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
    ax.set_yticklabels(LABELS)
    if kind is None:
        ax.set_xlabel(f"single patched layer within {archname}")
    else:
        ax.set_xlabel(f"center of interval of 10 patched {kind} layers")
    cb = plt.colorbar(h)
    # The following should be cb.ax.set_xlabel(answer), but this is broken in matplotlib 3.5.1.
    if cbar_title:
        cb.ax.set_title(str(cbar_title).strip(), y=-0.16, fontsize=10)

    if savepdf:
        plt.savefig(savepdf, bbox_inches="tight")
    plt.close("all")

def export_legend(legend, filename):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def make_line_plot(result,
                   result_std,
                   low_score,
                   high_score,
                   count,
                   archname,
                   savepdf,
                   ylabel="AIE"
                  ):
    color_order = [0, 1, 2, 4, 5, 3]
    x = None

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(4, 1.8), dpi=200)
    for i, label in list(enumerate(LABELS)):
        y = result[i]
        if x is None:
            x = list(range(len(y)))
        std = result_std[i]
        error = std * 1.96 / math.sqrt(count)
        ax.fill_between(
            x, y - error, y + error, alpha=0.3, color=cmap.colors[color_order[i]]
        )
        ax.plot(x, y, label=label, color=cmap.colors[color_order[i]])

        ax.set_ylabel(ylabel)
        ax.set_xlabel("layer")
    legend = ax.legend(frameon=True, facecolor='white', framealpha=1)
    plt.ylim([low_score, high_score])
    plt.tight_layout()
    if savepdf:
        # save legend separately
        export_legend(legend, os.path.join(os.path.dirname(savepdf), "legend.pdf"))
        legend.remove()
        plt.savefig(savepdf, bbox_inches="tight")
    plt.close("all")

def main(args):
    kind = "mlp"
    data = pd.read_json(args.query_file, lines=args.query_file.endswith(".jsonl"))
    print(f"The data contains {len(data)} entries. Generating results for the first 1000...")
    print()
    data = data.iloc[:1000]
    count = len(data)
    
    data["CT_results_dir"] = args.CT_folder
    data["filename_template"] = args.filename_template
    
    create_plots(data, kind, count, args.arch, args.archname, args.savefolder)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--query_file",
        required=True,
        type=str,
        help="File with queries to process, corresponding to the CT results.",
    )
    argparser.add_argument(
        "--CT_folder",
        required=True,
        type=str,
        help="Folder with CT results to plot, corresponding to the queries.",
    )
    argparser.add_argument(
        "--savefolder",
        required=True,
        type=str,
        help="Folder to save plot results to.",
    )
    argparser.add_argument(
        "--filename_template",
        required=True,
        type=str,
        help="Template used for the files with CT results.",
    )
    argparser.add_argument(
        "--arch",
        required=True,
        type=str,
    )
    argparser.add_argument(
        "--archname",
        required=True,
        type=str,
    )
    args = argparser.parse_args()

    print(args)
    main(args)