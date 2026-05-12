import sys

sys.path.append("CMC_utils")

import os
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from CMC_utils.plots import load_missing_results, load_experiments_results

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=4)
from tqdm import tqdm
tqdm.pandas()

sns.set_style("whitegrid")
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Cambria']})
rc('text', usetex=True)


def compose_table(mean_results, std_results, metric: str, filename: str, output_path: str, col: str, test: bool = True):
    if test:
        mean_results, std_results = mean_results.copy().drop("train_percentage", axis=1).set_index([col, "test_percentage"]), std_results.copy().drop("train_percentage", axis=1).set_index([col, "test_percentage"])
    else:
        mean_results, std_results = mean_results.copy().drop("test_percentage", axis=1).set_index([col, "train_percentage"]), std_results.copy().drop("test_percentage", axis=1).set_index([col, "train_percentage"])
    mean_results = mean_results.loc[:, [metric]].round(2)  # mean_results.drop(["train_percentage", "accuracy", "recall", "precision", "f1_score", "mcc", "gmean"], axis=1).set_index(["missing_strategy", "test_percentage"])  # .unstack()
    std_results = std_results.loc[:, [metric]]  # std_results.drop(["train_percentage", "accuracy", "recall", "precision", "f1_score", "mcc", "gmean"], axis=1).set_index(["missing_strategy", "test_percentage"])  # .unstack()

    std_err_results = (std_results/np.sqrt(5)).round(2)

    results = (mean_results[metric].astype(str) + r'$\pm$' + std_err_results[metric].astype(str)).unstack()

    results.to_latex(os.path.join(output_path, filename + '.txt'), index=True)


def extract_info(sample):
    sample_name = sample['experiment']
    if sample_name == "joint_pretrained_mariacat_noimputation_separate":
        info = dict(model="Ours", missing_strategy="Masked Attention", fusion_strategy="Intermediate")
    elif sample_name == "joint_pretrained_model_selection_mlp_MLP_noimputation_separate":
        info = dict(model="MLP", missing_strategy="Model Selection", fusion_strategy="Intermediate")
    elif sample_name == "joint_pretrained_zeros_mlp_MLP_noimputation_separate":
        info = dict(model="MLP", missing_strategy="Zeros", fusion_strategy="Intermediate")
    elif sample_name == "joint_pretrained_add_pooling_MLP_noimputation_separate":
        info = dict(model="MLP", missing_strategy="Sum Pooling", fusion_strategy="Intermediate")
    elif sample_name == "joint_pretrained_max_pooling_MLP_noimputation_separate":
        info = dict(model="MLP", missing_strategy="Max Pooling", fusion_strategy="Intermediate")
    elif sample_name == "joint_pretrained_frozen_mariacat_noimputation_separate":
        info = dict(model="Ours", missing_strategy="Masked Attention", fusion_strategy="Early")
    elif sample_name == "joint_mariacat_noimputation_separate":
        info = dict(model="Ours Scratch", missing_strategy="Masked Attention", fusion_strategy="Intermediate")
    elif sample_name == "joint_pretrained_zeros_maria_mariacat_noimputation_separate":
        info = dict(model="Ours", missing_strategy="Zeros", fusion_strategy="Intermediate")
    elif sample_name == "joint_pretrained_model_selection_maria_mariacat_noimputation_separate":
        info = dict(model="Ours", missing_strategy="Model Selection", fusion_strategy="Intermediate")
    elif sample_name == "late_naimcat_resnet_noimputation_separatemean":
        info = dict(model="Ours", missing_strategy="Intrinsic", fusion_strategy="Late")
    elif sample_name == "multilabel_classification_resnet":
        info = dict(model="ResNet", missing_strategy="-", fusion_strategy="-")
    elif sample_name == "multilabel_naimcat_noimputation":
        info = dict(model="NAIM", missing_strategy="Intrinsic", fusion_strategy="-")
    else:
        print(sample_name)
        raise ValueError("Unknown sample name")

    info = pd.Series(info)
    sample.update(info)

    return sample


def plot_performance_by_missing_percentage(data: pd.DataFrame, metric: str, filename: str, output_path: str, order: list, extension: str = "svg", hue: str = None, style: str = None, title: str = None, y_lim: tuple = None, test: bool = True):
    unimodal_data = data.loc[data[hue] == "Unimodal"]  # .isin(("Unimodal Tabular", "Unimodal Imaging"))]
    data = data.loc[data[hue] != "Unimodal"]  # ~ bla .isin(("Unimodal Tabular", "Unimodal Imaging"))]
    if test:
        data_grouped = data.groupby('train_percentage')
        percentages = sorted(data.test_percentage.unique())
    else:
        data_grouped = data.groupby('test_percentage')
        percentages = sorted(data.train_percentage.unique())
    rowlength = np.ceil(data_grouped.ngroups / 2).astype(int)

    percentages_str = [f"{perc}%" for perc in percentages]

    palette = {
        "Intermediate": "#3B75AF",
        "Masked Attention": "#3B75AF",
        "Early": "#C53A32", #rosso
        "Late": "#519E3E", #verde
        "Zeros": "#C53A32", #giallo
        "Max Pooling": "#EF8636",
        "Model Selection": "#519E3E",
        "Unimodal": "black",
        # "Unimodal Imaging": "black",
        # "Unimodal Tabular": "black",
    }

    styles = {
        "Intermediate": "",
        "Masked Attention": "",
        "Early": "",
        "Late": "",
        "Zeros": "",
        "Max Pooling": "",
        "Model Selection": "",
        "Unimodal": (1.6, 0.8),
        #"Unimodal Imaging": (5, 3),
        #"Unimodal Tabular": (1, 2),
    }

    fig, axs = plt.subplots(figsize=(10, 6), nrows=1, ncols=rowlength, sharex='all', sharey='all', gridspec_kw=dict(hspace=0.5))

    targets = zip(data_grouped.groups.keys(), [axs])  # .flatten())
    for i, (key, ax) in enumerate(targets):
        if test:
            l1 = sns.lineplot(data=data_grouped.get_group(key), x="test_percentage", y=metric, hue=hue, style=hue, markers=False, dashes=styles, ax=ax, errorbar=None, linewidth=3, palette=palette)
            sns.lineplot(data=unimodal_data, x="test_percentage", y=metric, hue=hue, style=hue, markers=False, dashes=styles, ax=ax, errorbar=None, linewidth=2, palette={"Unimodal": "black", "Unimodal Imaging": "black", "Unimodal Tabular": "black"})
        else:
            l1 = sns.lineplot(data=data_grouped.get_group(key), x="train_percentage", y=metric, hue=hue, style=hue, markers=False, dashes=styles, ax=ax, errorbar=None, linewidth=3, palette=palette)
            sns.lineplot(data=unimodal_data, x="train_percentage", y=metric, hue=hue, style=hue, markers=False, dashes=styles, ax=ax, errorbar=None, linewidth=2, palette={"Unimodal": "black", "Unimodal Imaging": "black", "Unimodal Tabular": "black"})
        l1.set_xticks(percentages)
        l1.set_xticklabels(percentages_str)
        if test:
            ax.set(xlabel='Missing in testing (\%)', ylabel=f"{metric.upper()} (\%)")
        else:
            ax.set(xlabel='Missing in training (\%)', ylabel=f"{metric.upper()} (\%)")
        #ax.set(xlabel='', ylabel=f"{metric.upper()} (\%)")
        # ax.set_title(f"Missing in training: {key}\%", fontsize=24, pad=15)

        if y_lim:
            ax.set_ylim(*y_lim)
            ax.set_yticks(np.arange(*y_lim, 5).tolist() + [y_lim[1]])

        ax.xaxis.label.set_fontsize(20)
        ax.yaxis.label.set_fontsize(20)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        handles, labels = ax.get_legend_handles_labels()  # sorted(zip(*ax.get_legend_handles_labels()), key=lambda h_l: list(focus_map.keys()).index(h_l[1]))
        # handles, labels = [h for h, _ in handles_labels], [l for _, l in handles_labels]

        handles = [handles[labels.index(l)] for l in order]
        labels = order

        for handle in handles:
            handle.set_linewidth(3)
        # fig.legend(handles, list(map(lambda x: focus_map[x], labels)), loc='center left', bbox_to_anchor=(0.92, 0.5), fontsize=16)
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.9, 0.5), fontsize=10)
        #fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.3, 0.03), fontsize=10, ncol=len(labels))
        # fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.7, 0.25), fontsize=16)
        ax.get_legend().remove()

        ax.spines['bottom'].set_color("k")
        ax.spines['top'].set_color("k")
        ax.spines['right'].set_color("k")
        ax.spines['left'].set_color("k")
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelbottom=True)
        ax.xaxis.get_label().set_visible(True)
        ax.yaxis.get_label().set_visible(True)

    # if len(axs.flatten())-1 > i:
    #    for j in range(i+1, len(axs.flatten())):
    #        axs.flatten()[j].axis('off')
    if title:
        plt.suptitle(title.upper(), fontsize=24, fontweight='extra bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{filename}.{extension}'), format=extension)
    plt.close()

def plot_missing_results(result_paths, average, plots_output_path, extension="png"):
    if not os.path.exists(plots_output_path):
        os.makedirs(plots_output_path)

    result_paths = pd.Series([os.path.join(path, "multilabelstratifiedkfold_multilabelholdout") for path in result_paths])
    # unimodal_result_paths = pd.Series([os.path.join(path, "multilabelstratifiedkfold_multilabelholdout") for path in unimodal_results_paths])
    mean_results_path = os.path.join(plots_output_path, "mean_results.csv")
    std_results_path = os.path.join(plots_output_path, "std_results.csv")
    if not os.path.exists(mean_results_path):
        all_results = pd.concat( result_paths.progress_apply(lambda x: load_missing_results(x, result_type="averages", separate_experiments=False, verbose=False)).values, axis=0)

        all_results = all_results.loc[all_results["class"] == average]

        mean_cols = [col for col in all_results.columns if col.endswith("_mean")]
        std_cols = [col for col in all_results.columns if col.endswith("_std")]

        mean_results = all_results[mean_cols].rename(columns={col: col.replace("_mean", "") for col in mean_cols})
        std_results = all_results[std_cols].rename(columns={col: col.replace("_std", "") for col in std_cols})
        del mean_cols, all_results, std_cols

        mean_results = mean_results.reset_index()
        std_results = std_results.reset_index()

        mean_results.to_csv(mean_results_path, index=False)
        std_results.to_csv(std_results_path, index=False)
    else:
        mean_results = pd.read_csv(mean_results_path, header=0)
        std_results = pd.read_csv(std_results_path, header=0)
    del result_paths, mean_results_path

    # DL_models = {"mariacat": "MARIA", "MLP": "MLP", "naimcat_resnet": "NAIM & ResNet", "naimcat": "NAIM", "resnet": "ResNet"}

    filter_late_approaches = mean_results["fusion_strategy"].apply(lambda x: x not in ("latemajorityvoting", "latemin", "latemax"))
    mean_results = mean_results.loc[filter_late_approaches].reset_index(drop=True)
    filter_late_approaches_stds = std_results["fusion_strategy"].apply(lambda x: x not in ("latemajorityvoting", "latemin", "latemax"))
    std_results = std_results.loc[filter_late_approaches_stds].reset_index(drop=True)

    mean_results = mean_results.parallel_apply(extract_info, axis=1)
    std_results = std_results.parallel_apply(extract_info, axis=1)

    filter_pooling_approaches = mean_results["missing_strategy"].apply(lambda x: x not in ("Sum Pooling",))
    mean_results = mean_results.loc[filter_pooling_approaches].reset_index(drop=True)

    filter_pooling_approaches_std = std_results["missing_strategy"].apply(lambda x: x not in ("Sum Pooling",))
    std_results = std_results.loc[filter_pooling_approaches_std].reset_index(drop=True)

    filter_training_approaches = mean_results["model"].apply(lambda x: x not in ("Ours Scratch",))
    mean_results = mean_results.loc[filter_training_approaches].reset_index(drop=True)

    filter_training_approaches_std = std_results["model"].apply(lambda x: x not in ("Ours Scratch",))
    std_results = std_results.loc[filter_training_approaches_std].reset_index(drop=True)
    # fusion_strategies = {"mariacat": "MARIA", "MLP": "MLP", "naimcat_resnet": "NAIM & ResNet", "naimcat": "NAIM", "resnet": "ResNet"}
    # mean_results["model"] = mean_results.model.apply(lambda x: DL_models.get(x, x))

    mean_results.loc[:, "db"] = mean_results.db.str.replace("multimodal_MIMIC", "MIMIC_multimodal")
    mean_results = mean_results.drop(["experiment", "imputer"], axis=1)

    std_results.loc[:, "db"] = std_results.db.str.replace("multimodal_MIMIC", "MIMIC_multimodal")
    std_results = std_results.drop(["experiment", "imputer"], axis=1)

    std_results = std_results.loc[mean_results.index]
    # mean_results = mean_results.loc[mean_results.imputer.isin(imputers + ["noimputation"])].reset_index(drop=True)
    # del imputers

    # mean_results.loc[:, "fusion_strategy"] = mean_results.fusion_strategy.where(mean_results.fusion_strategy != "latemean", "late")
    # mean_results = mean_results.loc[mean_results.fusion_strategy.isin(["Early", "Intermediate", "Late"])].reset_index(drop=True)
    # mean_results = mean_results.loc[mean_results.missing_strategy.isin(["all", "modalities"])].reset_index(drop=True)

    unimodal_map = mean_results.fusion_strategy == "-"
    unimodal_results = mean_results.loc[unimodal_map]
    unimodal_results_std = std_results.loc[unimodal_map]
    multimodal_results = mean_results.loc[~unimodal_map]
    multimodal_results_std = std_results.loc[~unimodal_map]

    ours_map = multimodal_results.model.str.contains("Ours")
    all_ours_results = multimodal_results.loc[ours_map]
    all_ours_results_std = multimodal_results_std.loc[ours_map]
    competitors_results = multimodal_results.loc[~ours_map]
    competitors_results_std = multimodal_results_std.loc[~ours_map]

    # DL_competitors_results = competitors_results.loc[competitors_results.model.isin(DL_models.values())].reset_index(drop=True)
    # ML_competitors_results = competitors_results.loc[competitors_results.model.isin(ML_models.values())].reset_index(drop=True)
    our_map = (all_ours_results.missing_strategy == "Masked Attention") & (all_ours_results.fusion_strategy == "Intermediate")
    our_results = all_ours_results.loc[our_map].reset_index(drop=True)
    our_results_std = all_ours_results_std.loc[our_map].reset_index(drop=True)
    ours_competitors_results = all_ours_results.loc[~our_map].reset_index(drop=True)
    ours_competitors_results_std = all_ours_results_std.loc[~our_map].reset_index(drop=True)
    ours_competitors_intermediate_results = ours_competitors_results.loc[ours_competitors_results.fusion_strategy == "Intermediate"].reset_index(drop=True)
    ours_competitors_intermediate_results_std = ours_competitors_results_std.loc[ours_competitors_results_std.fusion_strategy == "Intermediate"].reset_index(drop=True)
    ours_competitors_early_late_results = ours_competitors_results.loc[ours_competitors_results.fusion_strategy != "Intermediate"].reset_index(drop=True)
    ours_competitors_early_late_results_std = ours_competitors_results_std.loc[ours_competitors_results_std.fusion_strategy != "Intermediate"].reset_index(drop=True)

    our_results_tab = our_results.loc[our_results.test_percentage.str.startswith('0-')].reset_index(drop=True)
    our_results_tab_std = our_results_std.loc[our_results_std.test_percentage.str.startswith('0-')].reset_index(drop=True)
    our_results_img = our_results.loc[our_results.test_percentage.str.endswith('-0')].reset_index(drop=True)
    our_results_img_std = our_results_std.loc[our_results_std.test_percentage.str.endswith('-0')].reset_index(drop=True)
    ours_competitors_intermediate_results_tab = ours_competitors_intermediate_results.loc[ours_competitors_intermediate_results.test_percentage.str.startswith('0-')].reset_index(drop=True)
    ours_competitors_intermediate_results_tab_std = ours_competitors_intermediate_results_std.loc[ours_competitors_intermediate_results_std.test_percentage.str.startswith('0-')].reset_index(drop=True)
    ours_competitors_intermediate_results_img = ours_competitors_intermediate_results.loc[ours_competitors_intermediate_results.test_percentage.str.endswith('-0')].reset_index(drop=True)
    ours_competitors_intermediate_results_img_std = ours_competitors_intermediate_results_std.loc[ours_competitors_intermediate_results_std.test_percentage.str.endswith('-0')].reset_index(drop=True)
    ours_competitors_early_late_results_tab = ours_competitors_early_late_results.loc[ours_competitors_early_late_results.test_percentage.str.startswith('0-')].reset_index(drop=True)
    ours_competitors_early_late_results_tab_std = ours_competitors_early_late_results_std.loc[ours_competitors_early_late_results_std.test_percentage.str.startswith('0-')].reset_index(drop=True)
    ours_competitors_early_late_results_img = ours_competitors_early_late_results.loc[ours_competitors_early_late_results.test_percentage.str.endswith('-0')].reset_index(drop=True)
    ours_competitors_early_late_results_img_std = ours_competitors_early_late_results_std.loc[ours_competitors_early_late_results_std.test_percentage.str.endswith('-0')].reset_index(drop=True)
    competitors_results_tab = competitors_results.loc[competitors_results.test_percentage.str.startswith('0-')].reset_index(drop=True)
    competitors_results_tab_std = competitors_results_std.loc[competitors_results_std.test_percentage.str.startswith('0-')].reset_index(drop=True)
    competitors_results_img = competitors_results.loc[competitors_results.test_percentage.str.endswith('-0')].reset_index(drop=True)
    competitors_results_img_std = competitors_results_std.loc[competitors_results_std.test_percentage.str.endswith('-0')].reset_index(drop=True)
    del ours_map, our_map, all_ours_results, our_results, mean_results, filter_pooling_approaches, filter_training_approaches, filter_late_approaches, unimodal_map, multimodal_results, ours_competitors_results, competitors_results
    del ours_competitors_intermediate_results, ours_competitors_early_late_results
    del ours_competitors_intermediate_results_std, ours_competitors_early_late_results_std

    #all_original_missing_percentages = {"ADNI_diagnosis_binary": 49, "ADNI_diagnosis_multiclass": 49,
    #                                    "ADNI_prognosis_m12": 36, "ADNI_prognosis_m24": 35, "AIforCOVID_death": 23,
    #                                    "AIforCOVID_prognosis": 23, "ADNI_prognosis_m36": 29, "ADNI_prognosis_m48": 37}
    # features_original_missing_percentages = {"ADNI_diagnosis_binary": 33, "ADNI_diagnosis_multiclass": 33,
    #                                         "ADNI_prognosis_m12": 27, "ADNI_prognosis_m24": 27, "AI4Covid_death": 0,
    #                                         "AI4Covid_prognosis": 0, "ADNI_prognosis_m36": 23,
    #                                         "ADNI_prognosis_m48": 28}

    #dbs_metrics = dict(ADNI_diagnosis_binary="auc", ADNI_diagnosis_multiclass="auc", ADNI_prognosis_m12="mcc", ADNI_prognosis_m24="mcc", AIforCOVID_death="mcc", AIforCOVID_prognosis="auc", ADNI_prognosis_m36="mcc", ADNI_prognosis_m48="mcc")

    #train_percentage_rule = lambda row, info: row.train_percentage > info[row.db] or row.train_percentage == 0
    #test_percentage_rule = lambda row, info: row.test_percentage > info[row.db] or row.test_percentage == 0

    #NAIM_results_all = NAIM_results_all.loc[NAIM_results_all.apply(lambda x: train_percentage_rule(x, all_original_missing_percentages), axis=1)].reset_index(drop=True)
    #NAIM_results_all = NAIM_results_all.loc[NAIM_results_all.apply(lambda x: test_percentage_rule(x, all_original_missing_percentages), axis=1)].reset_index(drop=True)

    # NAIM_results_features = NAIM_results_features.loc[NAIM_results_features.apply(lambda x: train_percentage_rule(x, features_original_missing_percentages), axis=1)].reset_index(drop=True)
    # NAIM_results_features = NAIM_results_features.loc[NAIM_results_features.apply(lambda x: test_percentage_rule(x, features_original_missing_percentages), axis=1)].reset_index(drop=True)
    ####################################################################################################################
    unimodal_results["missing_strategy"] = "Unimodal"
    unimodal_results_std["missing_strategy"] = "Unimodal"
    unimodal_results["fusion_strategy"] = "Unimodal"
    unimodal_results_std["fusion_strategy"] = "Unimodal"

    unimodal_results["train_percentage"] = unimodal_results.train_percentage.str.replace('-0', '').astype(int)
    unimodal_results_std["train_percentage"] = unimodal_results_std.train_percentage.str.replace('-0', '').astype(int)
    unimodal_results["test_percentage"] = unimodal_results.test_percentage.str.replace('-0', '').astype(int)
    unimodal_results_std["test_percentage"] = unimodal_results_std.test_percentage.str.replace('-0', '').astype(int)

    unimodal_results_train = unimodal_results.copy()
    unimodal_results = unimodal_results.loc[unimodal_results.train_percentage == 0]

    unimodal_results_std_train = unimodal_results_std.copy()
    unimodal_results_std = unimodal_results_std.loc[unimodal_results_std.train_percentage == 0]

    unimodal_results_img = unimodal_results.loc[unimodal_results.model == "ResNet"].drop(["db", "model"],axis=1).reset_index(drop=True)
    unimodal_results_img_std = unimodal_results_std.loc[unimodal_results_std.model == "ResNet"].drop(["db", "model"],axis=1).reset_index(drop=True)
    unimodal_results_tab = unimodal_results.loc[unimodal_results.model == "NAIM"].reset_index(drop=True)
    unimodal_results_tab_std = unimodal_results_std.loc[unimodal_results_std.model == "NAIM"].reset_index(drop=True)
    unimodal_results_img = pd.concat([unimodal_results_img, unimodal_results_img], axis=0).reset_index(drop=True)
    unimodal_results_img_std = pd.concat([unimodal_results_img_std, unimodal_results_img_std], axis=0).reset_index(drop=True)
    unimodal_results_tab = pd.concat([unimodal_results_tab, unimodal_results_tab], axis=0).reset_index(drop=True)
    unimodal_results_tab_std = pd.concat([unimodal_results_tab_std, unimodal_results_tab_std], axis=0).reset_index(drop=True)
    unimodal_results_img.loc[unimodal_results_img.shape[0]//2:, "test_percentage"] = 100
    unimodal_results_img_std.loc[unimodal_results_img_std.shape[0]//2:, "test_percentage"] = 100
    unimodal_results_tab.loc[unimodal_results_tab.shape[0]//2:, "test_percentage"] = 100
    unimodal_results_tab_std.loc[unimodal_results_tab_std.shape[0]//2:, "test_percentage"] = 100

    unimodal_results_img_train = unimodal_results_train.loc[unimodal_results_train.model == "ResNet"].drop(["db", "model"],axis=1).reset_index(drop=True)
    unimodal_results_img_std_train = unimodal_results_std_train.loc[unimodal_results_std_train.model == "ResNet"].drop(["db", "model"],axis=1).reset_index(drop=True)
    unimodal_results_tab_train = unimodal_results_train.loc[unimodal_results_train.model == "NAIM"].drop(["db", "model"],axis=1).reset_index(drop=True)
    unimodal_results_tab_std_train = unimodal_results_std_train.loc[unimodal_results_std_train.model == "NAIM"].drop(["db", "model"],axis=1).reset_index(drop=True)
    del unimodal_results, unimodal_results_std
    ##unimodal_results_tab.loc[:, ["missing_strategy", "fusion_strategy"]] = "Unimodal Tabular"
    ##unimodal_results_tab_std.loc[:, ["missing_strategy", "fusion_strategy"]] = "Unimodal Tabular"
    ##unimodal_results_tab_train.loc[:, ["missing_strategy", "fusion_strategy"]] = "Unimodal Tabular"
    ##unimodal_results_tab_std_train.loc[:, ["missing_strategy", "fusion_strategy"]] = "Unimodal Tabular"

    ##unimodal_results_img.loc[:, ["missing_strategy", "fusion_strategy"]] = "Unimodal Imaging"
    ##unimodal_results_img_std.loc[:, ["missing_strategy", "fusion_strategy"]] = "Unimodal Imaging"
    ##unimodal_results_img_train.loc[:, ["missing_strategy", "fusion_strategy"]] = "Unimodal Imaging"
    ##unimodal_results_img_std_train.loc[:, ["missing_strategy", "fusion_strategy"]] = "Unimodal Imaging"
    ####################################################################################################################

    competitors_results_img_all = pd.concat([our_results_img,competitors_results_img], axis=0).reset_index(drop=True)
    competitors_results_img_all_std = pd.concat([our_results_img_std,competitors_results_img_std], axis=0).reset_index(drop=True)

    competitors_results_img_all_train = competitors_results_img_all.loc[(competitors_results_img_all.test_percentage == "0-0") & competitors_results_img_all.train_percentage.str.endswith("-0")].reset_index(drop=True)
    competitors_results_img_all_std_train = competitors_results_img_all_std.loc[(competitors_results_img_all_std.test_percentage == "0-0") & competitors_results_img_all_std.train_percentage.str.endswith("-0")].reset_index(drop=True)
    competitors_results_img_all = competitors_results_img_all.loc[competitors_results_img_all.train_percentage == "0-0"].reset_index(drop=True)
    competitors_results_img_all_std = competitors_results_img_all_std.loc[competitors_results_img_all_std.train_percentage == "0-0"].reset_index(drop=True)

    competitors_results_img_all = competitors_results_img_all.drop(["db", "model", "fusion_strategy"], axis=1)
    competitors_results_img_all_std = competitors_results_img_all_std.drop(["db", "model", "fusion_strategy"], axis=1)
    competitors_results_img_all_train = competitors_results_img_all_train.drop(["db", "model", "fusion_strategy"], axis=1)
    competitors_results_img_all_std_train = competitors_results_img_all_std_train.drop(["db", "model", "fusion_strategy"], axis=1)
    competitors_results_img_all["train_percentage"] = competitors_results_img_all.train_percentage.str.replace('-0', '').astype(int)
    competitors_results_img_all_std["train_percentage"] = competitors_results_img_all_std.train_percentage.str.replace('-0', '').astype(int)
    competitors_results_img_all["test_percentage"] = competitors_results_img_all.test_percentage.str.replace('-0', '').astype(int)
    competitors_results_img_all_std["test_percentage"] = competitors_results_img_all_std.test_percentage.str.replace('-0', '').astype(int)
    competitors_results_img_all_train["train_percentage"] = competitors_results_img_all_train.train_percentage.str.replace('-0', '').astype(int)
    competitors_results_img_all_std_train["train_percentage"] = competitors_results_img_all_std_train.train_percentage.str.replace('-0', '').astype(int)
    competitors_results_img_all_train["test_percentage"] = competitors_results_img_all_train.test_percentage.str.replace('-0', '').astype(int)
    competitors_results_img_all_std_train["test_percentage"] = competitors_results_img_all_std_train.test_percentage.str.replace('-0', '').astype(int)
    competitors_results_img_all = pd.concat([competitors_results_img_all, unimodal_results_img.drop('fusion_strategy', axis=1)], axis=0).reset_index(drop=True)
    competitors_results_img_all_std = pd.concat([competitors_results_img_all_std, unimodal_results_img_std.drop('fusion_strategy', axis=1)], axis=0).reset_index(drop=True)
    competitors_results_img_all_train = pd.concat([competitors_results_img_all_train, unimodal_results_tab_train.drop('fusion_strategy', axis=1)], axis=0).reset_index(drop=True)
    competitors_results_img_all_std_train = pd.concat([competitors_results_img_all_std_train, unimodal_results_tab_std_train.drop('fusion_strategy', axis=1)], axis=0).reset_index(drop=True)

    competitors_results_tab_all = pd.concat([our_results_tab, competitors_results_tab], axis=0).reset_index(drop=True)
    competitors_results_tab_all_std = pd.concat([our_results_tab_std, competitors_results_tab_std], axis=0).reset_index(drop=True)

    competitors_results_tab_all_train = competitors_results_tab_all.loc[(competitors_results_tab_all.test_percentage == "0-0") & competitors_results_tab_all.train_percentage.str.startswith("0-")].reset_index(drop=True)
    competitors_results_tab_all_std_train = competitors_results_tab_all_std.loc[(competitors_results_tab_all_std.test_percentage == "0-0") & competitors_results_tab_all_std.train_percentage.str.startswith("0-")].reset_index(drop=True)
    competitors_results_tab_all = competitors_results_tab_all.loc[competitors_results_tab_all.train_percentage == "0-0"].reset_index(drop=True)
    competitors_results_tab_all_std = competitors_results_tab_all_std.loc[competitors_results_tab_all_std.train_percentage == "0-0"].reset_index(drop=True)

    competitors_results_tab_all = competitors_results_tab_all.drop(["db", "model", "fusion_strategy"], axis=1)
    competitors_results_tab_all_std = competitors_results_tab_all_std.drop(["db", "model", "fusion_strategy"], axis=1)
    competitors_results_tab_all_train = competitors_results_tab_all_train.drop(["db", "model", "fusion_strategy"], axis=1)
    competitors_results_tab_all_std_train = competitors_results_tab_all_std_train.drop(["db", "model", "fusion_strategy"], axis=1)
    competitors_results_tab_all["train_percentage"] = competitors_results_tab_all.train_percentage.str.replace('0-', '').astype(int)
    competitors_results_tab_all_std["train_percentage"] = competitors_results_tab_all_std.train_percentage.str.replace('0-', '').astype(int)
    competitors_results_tab_all["test_percentage"] = competitors_results_tab_all.test_percentage.str.replace('0-', '').astype(int)
    competitors_results_tab_all_std["test_percentage"] = competitors_results_tab_all_std.test_percentage.str.replace('0-', '').astype(int)
    competitors_results_tab_all_train["train_percentage"] = competitors_results_tab_all_train.train_percentage.str.replace('0-', '').astype(int)
    competitors_results_tab_all_std_train["train_percentage"] = competitors_results_tab_all_std_train.train_percentage.str.replace('0-', '').astype(int)
    competitors_results_tab_all_train["test_percentage"] = competitors_results_tab_all_train.test_percentage.str.replace('0-', '').astype(int)
    competitors_results_tab_all_std_train["test_percentage"] = competitors_results_tab_all_std_train.test_percentage.str.replace('0-', '').astype(int)
    competitors_results_tab_all = pd.concat([competitors_results_tab_all, unimodal_results_tab.drop('fusion_strategy', axis=1)], axis=0).reset_index(drop=True)
    competitors_results_tab_all_std = pd.concat([competitors_results_tab_all_std, unimodal_results_tab_std.drop('fusion_strategy', axis=1)], axis=0).reset_index(drop=True)
    competitors_results_tab_all_train = pd.concat([competitors_results_tab_all_train, unimodal_results_img_train.drop('fusion_strategy', axis=1)], axis=0).reset_index(drop=True)
    competitors_results_tab_all_std_train = pd.concat([competitors_results_tab_all_std_train, unimodal_results_img_std_train.drop('fusion_strategy', axis=1)], axis=0).reset_index(drop=True)
    del competitors_results_img, competitors_results_tab
    del competitors_results_img_std, competitors_results_tab_std
    ####################################################################################################################

    ours_competitors_intermediate_results_img_all = pd.concat([our_results_img, ours_competitors_intermediate_results_img], axis=0).reset_index(drop=True)
    ours_competitors_intermediate_results_img_all_std = pd.concat([our_results_img_std, ours_competitors_intermediate_results_img_std], axis=0).reset_index(drop=True)

    ours_competitors_intermediate_results_img_all_train = ours_competitors_intermediate_results_img_all.loc[(ours_competitors_intermediate_results_img_all.test_percentage == "0-0") & ours_competitors_intermediate_results_img_all.train_percentage.str.endswith("-0")].reset_index(drop=True)
    ours_competitors_intermediate_results_img_all_std_train = ours_competitors_intermediate_results_img_all_std.loc[(ours_competitors_intermediate_results_img_all_std.test_percentage == "0-0") & ours_competitors_intermediate_results_img_all_std.train_percentage.str.endswith("-0")].reset_index(drop=True)
    ours_competitors_intermediate_results_img_all = ours_competitors_intermediate_results_img_all.loc[ours_competitors_intermediate_results_img_all.train_percentage == "0-0"].reset_index(drop=True)
    ours_competitors_intermediate_results_img_all_std = ours_competitors_intermediate_results_img_all_std.loc[ours_competitors_intermediate_results_img_all_std.train_percentage == "0-0"].reset_index(drop=True)

    ours_competitors_intermediate_results_img_all = ours_competitors_intermediate_results_img_all.drop(["db", "model", "fusion_strategy"], axis=1)
    ours_competitors_intermediate_results_img_all_std = ours_competitors_intermediate_results_img_all_std.drop(["db", "model", "fusion_strategy"], axis=1)
    ours_competitors_intermediate_results_img_all["train_percentage"] = ours_competitors_intermediate_results_img_all.train_percentage.str.replace('-0', '').astype(int)
    ours_competitors_intermediate_results_img_all_std["train_percentage"] = ours_competitors_intermediate_results_img_all_std.train_percentage.str.replace('-0', '').astype(int)
    ours_competitors_intermediate_results_img_all["test_percentage"] = ours_competitors_intermediate_results_img_all.test_percentage.str.replace('-0', '').astype(int)
    ours_competitors_intermediate_results_img_all_std["test_percentage"] = ours_competitors_intermediate_results_img_all_std.test_percentage.str.replace('-0', '').astype(int)
    ours_competitors_intermediate_results_img_all_train["train_percentage"] = ours_competitors_intermediate_results_img_all_train.train_percentage.str.replace('-0', '').astype(int)
    ours_competitors_intermediate_results_img_all_std_train["train_percentage"] = ours_competitors_intermediate_results_img_all_std_train.train_percentage.str.replace('-0', '').astype(int)
    ours_competitors_intermediate_results_img_all_train["test_percentage"] = ours_competitors_intermediate_results_img_all_train.test_percentage.str.replace('-0', '').astype(int)
    ours_competitors_intermediate_results_img_all_std_train["test_percentage"] = ours_competitors_intermediate_results_img_all_std_train.test_percentage.str.replace('-0', '').astype(int)
    ours_competitors_intermediate_results_img_all = pd.concat([ours_competitors_intermediate_results_img_all, unimodal_results_img.drop('fusion_strategy', axis=1)], axis=0).reset_index(drop=True)
    ours_competitors_intermediate_results_img_all_std = pd.concat([ours_competitors_intermediate_results_img_all_std, unimodal_results_img_std.drop('fusion_strategy', axis=1)], axis=0).reset_index(drop=True)
    ours_competitors_intermediate_results_img_all_train = pd.concat([ours_competitors_intermediate_results_img_all_train, unimodal_results_tab_train.drop('fusion_strategy', axis=1)], axis=0).reset_index(drop=True)
    ours_competitors_intermediate_results_img_all_std_train = pd.concat([ours_competitors_intermediate_results_img_all_std_train, unimodal_results_tab_std_train.drop('fusion_strategy', axis=1)], axis=0).reset_index(drop=True)

    ours_competitors_intermediate_results_tab_all = pd.concat([our_results_tab, ours_competitors_intermediate_results_tab], axis=0).reset_index(drop=True)
    ours_competitors_intermediate_results_tab_all_std = pd.concat([our_results_tab_std, ours_competitors_intermediate_results_tab_std], axis=0).reset_index(drop=True)

    ours_competitors_intermediate_results_tab_all_train = ours_competitors_intermediate_results_tab_all.loc[(ours_competitors_intermediate_results_tab_all.test_percentage == "0-0") & ours_competitors_intermediate_results_tab_all.train_percentage.str.startswith("0-")].reset_index(drop=True)
    ours_competitors_intermediate_results_tab_all_std_train = ours_competitors_intermediate_results_tab_all_std.loc[(ours_competitors_intermediate_results_tab_all_std.test_percentage == "0-0") & ours_competitors_intermediate_results_tab_all_std.train_percentage.str.startswith("0-")].reset_index(drop=True)
    ours_competitors_intermediate_results_tab_all = ours_competitors_intermediate_results_tab_all.loc[ours_competitors_intermediate_results_tab_all.train_percentage == "0-0"].reset_index(drop=True)
    ours_competitors_intermediate_results_tab_all_std = ours_competitors_intermediate_results_tab_all_std.loc[ours_competitors_intermediate_results_tab_all_std.train_percentage == "0-0"].reset_index(drop=True)

    ours_competitors_intermediate_results_tab_all = ours_competitors_intermediate_results_tab_all.drop(["db", "model", "fusion_strategy"], axis=1)
    ours_competitors_intermediate_results_tab_all_std = ours_competitors_intermediate_results_tab_all_std.drop(["db", "model", "fusion_strategy"], axis=1)
    ours_competitors_intermediate_results_tab_all_train = ours_competitors_intermediate_results_tab_all_train.drop(["db", "model", "fusion_strategy"], axis=1)
    ours_competitors_intermediate_results_tab_all_std_train = ours_competitors_intermediate_results_tab_all_std_train.drop(["db", "model", "fusion_strategy"], axis=1)
    ours_competitors_intermediate_results_tab_all["train_percentage"] = ours_competitors_intermediate_results_tab_all.train_percentage.str.replace('0-', '').astype(int)
    ours_competitors_intermediate_results_tab_all_std["train_percentage"] = ours_competitors_intermediate_results_tab_all_std.train_percentage.str.replace('0-', '').astype(int)
    ours_competitors_intermediate_results_tab_all["test_percentage"] = ours_competitors_intermediate_results_tab_all.test_percentage.str.replace('0-', '').astype(int)
    ours_competitors_intermediate_results_tab_all_std["test_percentage"] = ours_competitors_intermediate_results_tab_all_std.test_percentage.str.replace('0-', '').astype(int)
    ours_competitors_intermediate_results_tab_all_train["train_percentage"] = ours_competitors_intermediate_results_tab_all_train.train_percentage.str.replace('0-', '').astype(int)
    ours_competitors_intermediate_results_tab_all_std_train["train_percentage"] = ours_competitors_intermediate_results_tab_all_std_train.train_percentage.str.replace('0-', '').astype(int)
    ours_competitors_intermediate_results_tab_all_train["test_percentage"] = ours_competitors_intermediate_results_tab_all_train.test_percentage.str.replace('0-', '').astype(int)
    ours_competitors_intermediate_results_tab_all_std_train["test_percentage"] = ours_competitors_intermediate_results_tab_all_std_train.test_percentage.str.replace('0-', '').astype(int)
    ours_competitors_intermediate_results_tab_all = pd.concat([ours_competitors_intermediate_results_tab_all, unimodal_results_tab.drop('fusion_strategy', axis=1)], axis=0).reset_index(drop=True)
    ours_competitors_intermediate_results_tab_all_std = pd.concat([ours_competitors_intermediate_results_tab_all_std, unimodal_results_tab_std.drop('fusion_strategy', axis=1)], axis=0).reset_index(drop=True)
    ours_competitors_intermediate_results_tab_all_train = pd.concat([ours_competitors_intermediate_results_tab_all_train, unimodal_results_img_train.drop('fusion_strategy', axis=1)], axis=0).reset_index(drop=True)
    ours_competitors_intermediate_results_tab_all_std_train = pd.concat([ours_competitors_intermediate_results_tab_all_std_train, unimodal_results_img_std_train.drop('fusion_strategy', axis=1)], axis=0).reset_index(drop=True)

    #######

    ours_competitors_early_late_results_img_all = pd.concat([our_results_img, ours_competitors_early_late_results_img], axis=0).reset_index(drop=True)
    ours_competitors_early_late_results_img_all_std = pd.concat([our_results_img_std, ours_competitors_early_late_results_img_std], axis=0).reset_index(drop=True)

    ours_competitors_early_late_results_img_all_train = ours_competitors_early_late_results_img_all.loc[(ours_competitors_early_late_results_img_all.test_percentage == "0-0") & ours_competitors_early_late_results_img_all.train_percentage.str.endswith("-0")].reset_index(drop=True)
    ours_competitors_early_late_results_img_all_std_train = ours_competitors_early_late_results_img_all_std.loc[(ours_competitors_early_late_results_img_all_std.test_percentage == "0-0") & ours_competitors_early_late_results_img_all_std.train_percentage.str.endswith("-0")].reset_index(drop=True)
    ours_competitors_early_late_results_img_all = ours_competitors_early_late_results_img_all.loc[ours_competitors_early_late_results_img_all.train_percentage == "0-0"].reset_index(drop=True)
    ours_competitors_early_late_results_img_all_std = ours_competitors_early_late_results_img_all_std.loc[ours_competitors_early_late_results_img_all_std.train_percentage == "0-0"].reset_index(drop=True)

    ours_competitors_early_late_results_img_all = ours_competitors_early_late_results_img_all.drop(["db", "model", "missing_strategy"], axis=1)
    ours_competitors_early_late_results_img_all_std = ours_competitors_early_late_results_img_all_std.drop(["db", "model", "missing_strategy"], axis=1)
    ours_competitors_early_late_results_img_all_train = ours_competitors_early_late_results_img_all_train.drop(["db", "model", "missing_strategy"], axis=1)
    ours_competitors_early_late_results_img_all_std_train = ours_competitors_early_late_results_img_all_std_train.drop(["db", "model", "missing_strategy"], axis=1)
    ours_competitors_early_late_results_img_all["train_percentage"] = ours_competitors_early_late_results_img_all.train_percentage.str.replace('-0', '').astype(int)
    ours_competitors_early_late_results_img_all_std["train_percentage"] = ours_competitors_early_late_results_img_all_std.train_percentage.str.replace('-0', '').astype(int)
    ours_competitors_early_late_results_img_all["test_percentage"] = ours_competitors_early_late_results_img_all.test_percentage.str.replace('-0', '').astype(int)
    ours_competitors_early_late_results_img_all_std["test_percentage"] = ours_competitors_early_late_results_img_all_std.test_percentage.str.replace('-0', '').astype(int)
    ours_competitors_early_late_results_img_all_train["train_percentage"] = ours_competitors_early_late_results_img_all_train.train_percentage.str.replace('-0', '').astype(int)
    ours_competitors_early_late_results_img_all_std_train["train_percentage"] = ours_competitors_early_late_results_img_all_std_train.train_percentage.str.replace('-0', '').astype(int)
    ours_competitors_early_late_results_img_all_train["test_percentage"] = ours_competitors_early_late_results_img_all_train.test_percentage.str.replace('-0', '').astype(int)
    ours_competitors_early_late_results_img_all_std_train["test_percentage"] = ours_competitors_early_late_results_img_all_std_train.test_percentage.str.replace('-0', '').astype(int)
    ours_competitors_early_late_results_img_all = pd.concat([ours_competitors_early_late_results_img_all, unimodal_results_img.drop('missing_strategy', axis=1)], axis=0).reset_index(drop=True)
    ours_competitors_early_late_results_img_all_std = pd.concat([ours_competitors_early_late_results_img_all_std, unimodal_results_img_std.drop('missing_strategy', axis=1)], axis=0).reset_index(drop=True)
    ours_competitors_early_late_results_img_all_train = pd.concat([ours_competitors_early_late_results_img_all_train, unimodal_results_tab_train.drop('missing_strategy', axis=1)], axis=0).reset_index(drop=True)
    ours_competitors_early_late_results_img_all_std_train = pd.concat([ours_competitors_early_late_results_img_all_std_train, unimodal_results_tab_std_train.drop('missing_strategy', axis=1)], axis=0).reset_index(drop=True)

    ours_competitors_early_late_results_tab_all = pd.concat([our_results_tab, ours_competitors_early_late_results_tab], axis=0).reset_index(drop=True)
    ours_competitors_early_late_results_tab_all_std = pd.concat([our_results_tab_std, ours_competitors_early_late_results_tab_std], axis=0).reset_index(drop=True)

    ours_competitors_early_late_results_tab_all_train = ours_competitors_early_late_results_tab_all.loc[(ours_competitors_early_late_results_tab_all.test_percentage == "0-0") & ours_competitors_early_late_results_tab_all.train_percentage.str.startswith("0-")].reset_index(drop=True)
    ours_competitors_early_late_results_tab_all_std_train = ours_competitors_early_late_results_tab_all_std.loc[(ours_competitors_early_late_results_tab_all_std.test_percentage == "0-0") & ours_competitors_early_late_results_tab_all_std.train_percentage.str.startswith("0-")].reset_index(drop=True)
    ours_competitors_early_late_results_tab_all = ours_competitors_early_late_results_tab_all.loc[ours_competitors_early_late_results_tab_all.train_percentage == "0-0"].reset_index(drop=True)
    ours_competitors_early_late_results_tab_all_std = ours_competitors_early_late_results_tab_all_std.loc[ours_competitors_early_late_results_tab_all_std.train_percentage == "0-0"].reset_index(drop=True)

    ours_competitors_early_late_results_tab_all = ours_competitors_early_late_results_tab_all.drop(["db", "model", "missing_strategy"], axis=1)
    ours_competitors_early_late_results_tab_all_std = ours_competitors_early_late_results_tab_all_std.drop(["db", "model", "missing_strategy"], axis=1)
    ours_competitors_early_late_results_tab_all_train = ours_competitors_early_late_results_tab_all_train.drop(["db", "model", "missing_strategy"], axis=1)
    ours_competitors_early_late_results_tab_all_std_train = ours_competitors_early_late_results_tab_all_std_train.drop(["db", "model", "missing_strategy"], axis=1)
    ours_competitors_early_late_results_tab_all["train_percentage"] = ours_competitors_early_late_results_tab_all.train_percentage.str.replace('0-', '').astype(int)
    ours_competitors_early_late_results_tab_all_std["train_percentage"] = ours_competitors_early_late_results_tab_all_std.train_percentage.str.replace('0-', '').astype(int)
    ours_competitors_early_late_results_tab_all["test_percentage"] = ours_competitors_early_late_results_tab_all.test_percentage.str.replace('0-', '').astype(int)
    ours_competitors_early_late_results_tab_all_std["test_percentage"] = ours_competitors_early_late_results_tab_all_std.test_percentage.str.replace('0-', '').astype(int)
    ours_competitors_early_late_results_tab_all_train["train_percentage"] = ours_competitors_early_late_results_tab_all_train.train_percentage.str.replace('0-', '').astype(int)
    ours_competitors_early_late_results_tab_all_std_train["train_percentage"] = ours_competitors_early_late_results_tab_all_std_train.train_percentage.str.replace('0-', '').astype(int)
    ours_competitors_early_late_results_tab_all_train["test_percentage"] = ours_competitors_early_late_results_tab_all_train.test_percentage.str.replace('0-', '').astype(int)
    ours_competitors_early_late_results_tab_all_std_train["test_percentage"] = ours_competitors_early_late_results_tab_all_std_train.test_percentage.str.replace('0-', '').astype(int)
    ours_competitors_early_late_results_tab_all = pd.concat([ours_competitors_early_late_results_tab_all, unimodal_results_tab.drop('missing_strategy', axis=1)], axis=0).reset_index(drop=True)
    ours_competitors_early_late_results_tab_all_std = pd.concat([ours_competitors_early_late_results_tab_all_std, unimodal_results_tab_std.drop('missing_strategy', axis=1)], axis=0).reset_index(drop=True)
    ours_competitors_early_late_results_tab_all_train = pd.concat([ours_competitors_early_late_results_tab_all_train, unimodal_results_img_train.drop('missing_strategy', axis=1)], axis=0).reset_index(drop=True)
    ours_competitors_early_late_results_tab_all_std_train = pd.concat([ours_competitors_early_late_results_tab_all_std_train, unimodal_results_img_std_train.drop('missing_strategy', axis=1)], axis=0).reset_index(drop=True)
    del ours_competitors_intermediate_results_img, ours_competitors_intermediate_results_tab, ours_competitors_early_late_results_img, ours_competitors_early_late_results_tab, our_results_img, our_results_tab
    del ours_competitors_intermediate_results_img_std, ours_competitors_intermediate_results_tab_std, ours_competitors_early_late_results_img_std, ours_competitors_early_late_results_tab_std, our_results_img_std, our_results_tab_std
    ####################################################################################################################

    compose_table(competitors_results_img_all, competitors_results_img_all_std, metric="auc", filename=f"remus_vs_competitors_img", output_path=plots_output_path, col="missing_strategy")
    compose_table(competitors_results_tab_all, competitors_results_tab_all_std, metric="auc", filename=f"remus_vs_competitors_tab", output_path=plots_output_path, col="missing_strategy")
    compose_table(competitors_results_img_all_train, competitors_results_img_all_std_train, metric="auc", filename=f"remus_vs_competitors_img_train", output_path=plots_output_path, col="missing_strategy", test=False)
    compose_table(competitors_results_tab_all_train, competitors_results_tab_all_std_train, metric="auc", filename=f"remus_vs_competitors_tab_train", output_path=plots_output_path, col="missing_strategy", test=False)
    compose_table(ours_competitors_intermediate_results_img_all, ours_competitors_intermediate_results_img_all_std, metric="auc", filename=f"remus_vs_ours_intermediate_competitors_img", output_path=plots_output_path, col="missing_strategy")
    compose_table(ours_competitors_intermediate_results_tab_all, ours_competitors_intermediate_results_tab_all_std, metric="auc", filename=f"remus_vs_ours_intermediate_competitors_tab", output_path=plots_output_path, col="missing_strategy")
    compose_table(ours_competitors_intermediate_results_img_all_train, ours_competitors_intermediate_results_img_all_std_train, metric="auc", filename=f"remus_vs_ours_intermediate_competitors_img_train", output_path=plots_output_path, col="missing_strategy", test=False)
    compose_table(ours_competitors_intermediate_results_tab_all_train, ours_competitors_intermediate_results_tab_all_std_train, metric="auc", filename=f"remus_vs_ours_intermediate_competitors_tab_train", output_path=plots_output_path, col="missing_strategy", test=False)
    compose_table(ours_competitors_early_late_results_img_all, ours_competitors_early_late_results_img_all_std, metric="auc", filename=f"remus_vs_ours_early_late_competitors_img", output_path=plots_output_path, col="fusion_strategy")
    compose_table(ours_competitors_early_late_results_tab_all, ours_competitors_early_late_results_tab_all_std, metric="auc", filename=f"remus_vs_ours_early_late_competitors_tab", output_path=plots_output_path, col="fusion_strategy")
    compose_table(ours_competitors_early_late_results_img_all_train, ours_competitors_early_late_results_img_all_std_train, metric="auc", filename=f"remus_vs_ours_early_late_competitors_img_train", output_path=plots_output_path, col="fusion_strategy", test=False)
    compose_table(ours_competitors_early_late_results_tab_all_train, ours_competitors_early_late_results_tab_all_std_train, metric="auc", filename=f"remus_vs_ours_early_late_competitors_tab_train", output_path=plots_output_path, col="fusion_strategy", test=False)
    plot_performance_by_missing_percentage(competitors_results_img_all, metric="auc", filename=f"remus_vs_competitors_img", output_path=plots_output_path, extension=extension, hue="missing_strategy", order=["Masked Attention", "Zeros", "Max Pooling", "Model Selection", "Unimodal"], title="Imaging Modality - Reducing Tabular", y_lim=(60,85))
    plot_performance_by_missing_percentage(competitors_results_tab_all, metric="auc", filename=f"remus_vs_competitors_tab", output_path=plots_output_path, extension=extension, hue="missing_strategy", order=["Masked Attention", "Zeros", "Max Pooling", "Model Selection", "Unimodal"], title="Tabular Modality - Reducing Imaging", y_lim=(60,85))
    plot_performance_by_missing_percentage(ours_competitors_intermediate_results_img_all, metric="auc", filename=f"remus_vs_ours_intermediate_competitors_img", output_path=plots_output_path, extension=extension, hue="missing_strategy", order=["Masked Attention", "Zeros", "Model Selection", "Unimodal"], title="Imaging Modality - Reducing Tabular", y_lim=(55,85))
    plot_performance_by_missing_percentage(ours_competitors_intermediate_results_tab_all, metric="auc", filename=f"remus_vs_ours_intermediate_competitors_tab", output_path=plots_output_path, extension=extension, hue="missing_strategy", order=["Masked Attention", "Zeros", "Model Selection", "Unimodal"], title="Tabular Modality - Reducing Imaging", y_lim=(55,85))
    plot_performance_by_missing_percentage(ours_competitors_early_late_results_img_all, metric="auc", filename=f"remus_vs_ours_early_late_competitors_img", output_path=plots_output_path, extension=extension, hue="fusion_strategy", order=["Intermediate", "Early", "Late", "Unimodal"], title="Imaging Modality - Reducing Tabular", y_lim=(70,85))
    plot_performance_by_missing_percentage(ours_competitors_early_late_results_tab_all, metric="auc", filename=f"remus_vs_ours_early_late_competitors_tab", output_path=plots_output_path, extension=extension, hue="fusion_strategy", order=["Intermediate", "Early", "Late", "Unimodal"], title="Tabular Modality - Reducing Imaging", y_lim=(70,85))

    unimodal_results_tab.loc[:, "test_percentage"] = 0
    unimodal_results_tab.loc[1, "train_percentage"] = 75
    unimodal_results_img.loc[:, "test_percentage"] = 0
    unimodal_results_img.loc[1, "train_percentage"] = 75
    #competitors_img_train = pd.concat([competitors_results_img_all_train, unimodal_results_img], axis=0, ignore_index=True)
    #competitors_tab_train = pd.concat([competitors_results_tab_all_train, unimodal_results_tab], axis=0, ignore_index=True)
    competitors_img_train = competitors_results_img_all_train
    competitors_tab_train = competitors_results_tab_all_train
    plot_performance_by_missing_percentage(competitors_img_train, metric="auc", filename=f"remus_vs_competitors_img_train", output_path=plots_output_path, extension=extension, hue="missing_strategy", order=["Masked Attention", "Zeros", "Max Pooling", "Model Selection", "Unimodal"], title="Imaging Modality - Reducing Tabular", y_lim=(55,85), test=False)
    plot_performance_by_missing_percentage(competitors_tab_train, metric="auc", filename=f"remus_vs_competitors_tab_train", output_path=plots_output_path, extension=extension, hue="missing_strategy", order=["Masked Attention", "Zeros", "Max Pooling", "Model Selection", "Unimodal"], title="Tabular Modality - Reducing Imaging", y_lim=(55,85), test=False)

    #ours_intermediate_img_train = pd.concat([ours_competitors_intermediate_results_img_all_train, unimodal_results_img], axis=0, ignore_index=True)
    #ours_intermediate_tab_train = pd.concat([ours_competitors_intermediate_results_tab_all_train, unimodal_results_tab], axis=0, ignore_index=True)
    ours_intermediate_img_train = ours_competitors_intermediate_results_img_all_train
    ours_intermediate_tab_train = ours_competitors_intermediate_results_tab_all_train
    plot_performance_by_missing_percentage(ours_intermediate_img_train, metric="auc", filename=f"remus_vs_ours_intermediate_competitors_img_train", output_path=plots_output_path, extension=extension, hue="missing_strategy", order=["Masked Attention", "Zeros", "Model Selection", "Unimodal"], title="Imaging Modality - Reducing Tabular", y_lim=(65,85), test=False)
    plot_performance_by_missing_percentage(ours_intermediate_tab_train, metric="auc", filename=f"remus_vs_ours_intermediate_competitors_tab_train", output_path=plots_output_path, extension=extension, hue="missing_strategy", order=["Masked Attention", "Zeros", "Model Selection", "Unimodal"], title="Tabular Modality - Reducing Imaging", y_lim=(65,85), test=False)

    #ours_early_late_img_train = pd.concat([ours_competitors_early_late_results_img_all_train, unimodal_results_img], axis=0, ignore_index=True)
    #ours_early_late_tab_train = pd.concat([ours_competitors_early_late_results_tab_all_train, unimodal_results_tab], axis=0, ignore_index=True)
    ours_early_late_img_train = ours_competitors_early_late_results_img_all_train
    ours_early_late_tab_train = ours_competitors_early_late_results_tab_all_train
    plot_performance_by_missing_percentage(ours_early_late_img_train, metric="auc", filename=f"remus_vs_ours_early_late_competitors_img_train", output_path=plots_output_path, extension=extension, hue="fusion_strategy", order=["Intermediate", "Early", "Late", "Unimodal"], title="Imaging Modality - Reducing Tabular", y_lim=(65,85), test=False)
    plot_performance_by_missing_percentage(ours_early_late_tab_train, metric="auc", filename=f"remus_vs_ours_early_late_competitors_tab_train", output_path=plots_output_path, extension=extension, hue="fusion_strategy", order=["Intermediate", "Early", "Late", "Unimodal"], title="Tabular Modality - Reducing Imaging", y_lim=(65,85), test=False)


    #plot_performance(competitors_results_img_all, metric="auc", filename=f"remus_vs_competitors_img", output_path=plots_output_path, extension=extension)
    #plot_performance(competitors_results_tab_all, metric="auc", filename=f"remus_vs_competitors_tab", output_path=plots_output_path, extension=extension)
    #plot_performance(NAIM_vs_ML_all, metrics=dbs_metrics, filename=f"MARIA_vs_ML_all", output_path=plots_output_path, extension=extension)
    ## plot_performance(NAIM_vs_ML_features, metrics=dbs_metrics, filename=f"NAIM_vs_ML_features", output_path=plots_output_path, extension=extension)
    #plot_performance(NAIM_vs_ML_modalities, metrics=dbs_metrics, filename=f"MARIA_vs_ML_modalities", output_path=plots_output_path, extension=extension)
    #plot_performance(NAIM_vs_NAIM_all, metrics=dbs_metrics, filename=f"MARIA_vs_NAIM_all", output_path=plots_output_path, extension=extension)
    ## plot_performance(NAIM_vs_NAIM_features, metrics=dbs_metrics, filename=f"NAIM_vs_NAIM_features", output_path=plots_output_path, extension=extension)
    #plot_performance(NAIM_vs_NAIM_modalities, metrics=dbs_metrics, filename=f"MARIA_vs_NAIM_modalities", output_path=plots_output_path, extension=extension)


if __name__ == "__main__":
    base_path = "/Users/camillocaruso/LocalDocuments/code_outputs/MIMIC_tr"

    multimodal_results_paths = [os.path.join(base_path, path) for path in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, path))]  # and path.startswith("multimodal")]
    # unimodal_results_paths = [os.path.join(base_path, path) for path in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, path)) and path.startswith("MIMIC")]

    average = "weighted_average"
    plots_output_path = "/Users/camillocaruso/Downloads/plots"

    plot_missing_results(multimodal_results_paths, average, plots_output_path, extension="svg")
