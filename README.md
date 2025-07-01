# Interpretations of Language Models
Replication Code for Paper Fact Recall, Heuristics or Pure Guesswork? Precise Interpretations of Language Models for Fact Completion

## Mechanistic Interpretations of PrISM scenarios

### Information flow analysis

We perform the analysis proposed by [Geva et al. (2023)](https://arxiv.org/pdf/2304.14767) on the samples from the different prediction scenarios in the PrISM dataset for the GPT-2 XL model. Specifically, we reproduce their information flow analysis (Figure 2 in the paper by Geva et al.) and study of intermediate MHSA and MLP predictions (Figure 5 in the paper by Geva et al.).

We first leverage scripts to collect the information flow results and intermediate predictions for the samples in the PrISM dataset (since this requires the use of a GPU). For this, we run the scripts under scripts/information_flow_analysis. Before this, you may want to create subsets of the PrISM sets such that each set contains 1,000 samples and analyse these. We did this for our analysis.

> Make sure that the PrISM data file paths specified in the scripts are correct.

> For all scripts, note that you need to edit them to specify your GPU allocation, set up the virtual enviroment and potentially specify the Hugging Face cache folder for model weights.

 We then analyse and plot the results in the notebook [src/information_flow_analysis/analyze_information_flow_results.ipynb](src/information_flow_analysis/analyze_information_flow_results.ipynb). This notebook loads the results from the scripts described above and creates the corresponding plots.