import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def load_json(fpath: str):
    """
        Load and parse a .json file.
    Args:
        fpath: The path to the .json file to load.
    Returns: The object resulting from parsing the file.
    """
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


TASK_CATEGORIES = load_json(CONFIGS_DIR / "task_categories.json")
THRESHOLDS_30BT = load_json(CONFIGS_DIR / "thresholds_30bt.json")
THRESHOLDS_100BT = load_json(CONFIGS_DIR / "thresholds_100bt.json")
RANDOM_BASELINES = load_json(CONFIGS_DIR / "random_baselines.json")

RES_COLUMNS = [
    "corpus",
    "category",
    "dataset",
    "task",
    "prompt",
    "model",
    "ckpt_num",
    "score",
]


def get_window(df: pd.DataFrame, lower_t: int, upper_t: int):
    """
        Select a subset of results within the specified pretraining window.
    Args:
        df: A dataframe with the results
        lower_t, upper_t: Thresholds that define the pretraining window.
    Returns: The results subset for the defined pretraining window.
    """
    return df.query("@lower_t <= ckpt_num <= @upper_t")


def aggregate_results(
    df: pd.DataFrame,
    groupby_cols: list = ["corpus", "category", "dataset", "model", "ckpt_num"],
):
    """
        Aggregate performance results for a given language.
    Args:
        df: A dataframe with the results
        groupby_cols: Ordered list of column names to group by for computing the statistics.
    Returns: Aggregated dataframe with arithmetic mean, standard deviation, min/max, median, MAD, and spread across a set of prompts for each task.
    """

    results = (
        df.groupby(groupby_cols)
        .score.agg(
            mean_score="mean",
            std_score="std",
            min_score="min",
            max_score="max",
            median_score="median",
            mad_score=lambda x: np.median(np.abs(x - np.median(x))),
        )
        .reset_index()
    )
    return results.sort_values(["corpus", "ckpt_num"])


def convert_ckpt_num(ckpt_num: int):
    """
        Convert a checkpoint identifier to a label.
    Args:
        ckpt: Checkpoint identifier (i.e., ``"<steps>"``)
    Returns: Token-count label in billions (e.g., ``"30B"``).
    """

    num_tokens = round(ckpt_num * 1024 * 2048 / 1000000000)
    return f"{num_tokens}B"


def read_results(fpath: str):
    """
        Load raw evaluation results and normalize the column names.
    Args:
        fpath: Path to a tab-separated results file.
    Returns: Normalized results formatted according to ``RES_COLUMNS``.
    """

    results = pd.read_csv(fpath, sep="\t")
    results["dataset"] = (
        results["task"].str.replace(r"_p\d+", "", regex=True).str.strip()
    )
    results["category"] = results["dataset"].apply(lambda x: TASK_CATEGORIES[x])
    results["ckpt_num"] = results["model"].apply(
        lambda x: int(x.replace("checkpoint_", ""))
    )
    results["model"] = results["ckpt_num"].apply(convert_ckpt_num)
    return results[RES_COLUMNS]


def compute_ordering_consistency(
    dataset_subset: pd.DataFrame, score_col: str, lower_t: int, upper_t: int
):
    """
        Measure how consistently model rankings are between successive checkpoints across the pretraining corpora.
    Args:
        dataset_subset: The aggregated results dataframe.
        score_col: Column name for the aggregated score (e.g., ``"max_score"``, ``"median_score"``, or ``"mean_score"``)
        lower_t, upper_t: Thresholds that define the pretraining window.
    Returns: Average Kendall's Tau correlation across the checkpoints.
    """
    window = get_window(df=dataset_subset, lower_t=lower_t, upper_t=upper_t)
    steps = window.model.unique().tolist()
    taus = []
    for i in range(len(steps) - 1):
        step1, step2 = steps[i], steps[i + 1]
        rank1 = (
            window.loc[window["model"] == step1, ["corpus", score_col]]
            .set_index("corpus")[score_col]
            .rank(ascending=False)
        )
        rank2 = (
            window.loc[window["model"] == step2, ["corpus", score_col]]
            .set_index("corpus")[score_col]
            .rank(ascending=False)
        )
        tau, _ = kendalltau(rank1, rank2)
        if np.isnan(tau):
            tau = 0.0
        taus.append(tau)
    return np.mean(taus)


def compute_coefficient_of_variation(
    dataset_subset: pd.DataFrame, score_col: str, lower_t: int, upper_t: int
):
    """
        Compute the coefficient of variation (CV) of performance trajectories.
    Args:
        dataset_subset: The aggregated results dataframe.
        score_col: Column name for the aggregated score (e.g., ``"max_score"``, ``"median_score"``, or ``"mean_score"``)
        lower_t, upper_t: Thresholds that define the pretraining window.
    Returns: Median CV (%).
    """

    results = []
    window = get_window(df=dataset_subset, lower_t=lower_t, upper_t=upper_t)
    for _, corpus_subset in window.groupby("corpus"):
        std_score = corpus_subset[score_col].std()
        mean_score = corpus_subset[score_col].mean()
        cv = np.nan if mean_score == 0 else std_score / mean_score * 100
        results.append(cv)
    return np.median(results)


def compute_signal_to_noise_ratio(
    dataset_subset: pd.DataFrame, score_col: str, lower_t: int, upper_t: int
):
    """
        Compute the signal-to-noise ratio (SNR) per corpus at the final checkpoint.
    Args:
        dataset_subset: The aggregated results dataframe.
        score_col: Column name for the aggregated score (e.g., ``"max_score"``, ``"median_score"``, or ``"mean_score"``)
        lower_t, upper_t: Thresholds that define the pretraining window.
    Returns: Median signal-to-noise ratio (SNR) across corpora.
    """

    results = []
    window = get_window(df=dataset_subset, lower_t=lower_t, upper_t=upper_t)
    for _, corpus_subset in window.groupby("corpus"):
        signal = corpus_subset[score_col].mean()
        if score_col == "median_score":
            noise = (1.4826 * corpus_subset["mad_score"]).mean()
        else:
            noise = corpus_subset["std_score"].mean()
        snr = signal / (noise + 1e-8)
        results.append(snr)
    return np.median(results)


def compute_monotonicity(
    dataset_subset: pd.DataFrame, score_col: str, lower_t: int, upper_t: int
):
    """
        Compute the  Spearman correlation between the checkpoint number and score.
    Args:
        dataset_subset: The aggregated results dataframe.
        score_col: Column name for the aggregated score (e.g., ``"max_score"``, ``"median_score"``, or ``"mean_score"``)
        lower_t, upper_t: Thresholds that define the pretraining window.
    Returns: Median Spearman correlation coefficient across corpora.
    """

    results = []
    window = get_window(df=dataset_subset, lower_t=lower_t, upper_t=upper_t)
    for _, corpus_subset in window.groupby("corpus"):
        monotonicity = corpus_subset["ckpt_num"].corr(
            corpus_subset[score_col], method="spearman"
        )
        results.append(monotonicity)
    return np.median(results)


def compute_randomness(
    dataset_subset: pd.DataFrame,
    score_col: str,
    random_baseline: float,
    lower_t: int,
    upper_t: int,
):
    """
        Compute the performance difference between the final checkpoint and a random baseline.
    Args:
        dataset_subset: The aggregated results dataframe.
        score_col: Column name for the aggregated score (e.g., ``"max_score"``, ``"median_score"``, or ``"mean_score"``)
        random_baseline: A random guessing baseline score for a given task.
        lower_t, upper_t: Thresholds that define the pretraining window.
    Returns: The absolute difference between the best checkpoint score across all corpora over the pretraining window and ``random_baseline``.
    """

    window = get_window(df=dataset_subset, lower_t=lower_t, upper_t=upper_t)
    return window[score_col].max() - random_baseline


def compute_mad(dataset_subset: pd.DataFrame, lower_t: int, upper_t: int):
    """
        Compute MAD and its 75th percentile (worst-case prompt sensitivity) across corpora.
    Args:
        dataset_subset: The aggregated results dataframe.
        lower_t, upper_t: Thresholds that define the pretraining window.
    Returns: Median MAD and 75th percentile MAD across corpora over the pretraining window.
    """

    results_mad, results_mad_q75 = [], []
    window = get_window(df=dataset_subset, lower_t=lower_t, upper_t=upper_t)
    for _, corpus_subset in window.groupby("corpus"):
        mad = corpus_subset.mad_score.median()
        mad_q75 = np.percentile(corpus_subset["mad_score"], 75)
        results_mad.append(mad)
        results_mad_q75.append(mad_q75)
    return np.median(results_mad), np.median(results_mad_q75)


def compute_criteria(
    raw_results: pd.DataFrame,
    aggregated_results: pd.DataFrame,
    score_col: str,
    thresholds: dict,
):
    """
        Summarize signal criteria and statistics for each task based on aggregated results.
    Args:
        raw_results: Raw results dataframe.
        aggregated_results: Aggregated results dataframe.
        score_col: Column name for the aggregated score (e.g., ``"max_score"``, ``"median_score"``, or ``"mean_score"``)
    Returns: Dataframe indexed by task with all computed criteria as columns.
    """

    criteria = {}
    for task, dataset_subset in aggregated_results.groupby("dataset"):
        criteria[task] = {}
        # Randomness
        randomness = compute_randomness(
            dataset_subset=dataset_subset,
            score_col=score_col,
            random_baseline=RANDOM_BASELINES[task],
            lower_t=thresholds["randomness"]["lower_t"],
            upper_t=thresholds["randomness"]["upper_t"],
        )
        criteria[task]["randomness"] = randomness
        # SNR
        snr = compute_signal_to_noise_ratio(
            dataset_subset=dataset_subset,
            score_col=score_col,
            lower_t=thresholds["snr"]["lower_t"],
            upper_t=thresholds["snr"]["upper_t"],
        )
        criteria[task]["snr"] = snr
        # Monotonicity
        monotonicity = compute_monotonicity(
            dataset_subset=dataset_subset,
            score_col=score_col,
            lower_t=thresholds["monotonicity"]["lower_t"],
            upper_t=thresholds["monotonicity"]["upper_t"],
        )
        criteria[task]["monotonicity"] = monotonicity
        # Consistency
        ordering_consistency = compute_ordering_consistency(
            dataset_subset=dataset_subset,
            score_col=score_col,
            lower_t=thresholds["consistency"]["lower_t"],
            upper_t=thresholds["consistency"]["upper_t"],
        )
        criteria[task]["consistency"] = ordering_consistency
        # Coefficient of variation
        cv = compute_coefficient_of_variation(
            dataset_subset=dataset_subset,
            score_col=score_col,
            lower_t=thresholds["cv"]["lower_t"],
            upper_t=thresholds["cv"]["upper_t"],
        )
        criteria[task]["cv"] = cv
        # MAD
        mad, mad_q75 = compute_mad(
            dataset_subset=dataset_subset,
            lower_t=thresholds["mad"]["lower_t"],
            upper_t=thresholds["mad"]["upper_t"],
        )
        criteria[task]["mad"] = mad
        criteria[task]["mad_q75"] = mad_q75
    criteria_results = pd.DataFrame(criteria).T.reset_index(names="dataset")
    prompt_switch_rate = compute_prompt_switch_rate(
        raw_results=raw_results,
        lower_t=thresholds["prompt_switch_rate"]["lower_t"],
        upper_t=thresholds["prompt_switch_rate"]["upper_t"],
    )
    return criteria_results.merge(prompt_switch_rate, on="dataset").round(3)


def compute_prompt_switch_rate(
    raw_results: pd.DataFrame,
    lower_t: int,
    upper_t: int,
    group_cols=["corpus", "dataset"],
    step_col="ckpt_num",
    task_col="task",
    score_col="score",
):
    """
        Compute the prompt-switch rate for each dataset.
    Args:
        results: Raw results dataframe.
        lower_t, upper_t: Thresholds that define the mid-late pretraining window.
        group_cols: Columns defining each evaluation run (e.g., corpus + dataset).
        step_col: Column representing pretraining steps or tokens.
        task_col: Column identifying the prompt variant as named in LM Evaluation Harness.
        score_col: Column containing the raw performance score.
    Returns: The prompt switch rate for each dataset.
    """
    results = []
    for (corpus, dataset), dataset_subset in raw_results.groupby(group_cols):
        window = get_window(df=dataset_subset, lower_t=lower_t, upper_t=upper_t)

        best_prompts = window.loc[
            window.groupby(step_col)[score_col].idxmax(), task_col
        ]
        if len(best_prompts) < 2:
            switch_rate = np.nan
        else:
            switches = (best_prompts != best_prompts.shift(1)).sum() - 1
            switch_rate = round(switches / (len(best_prompts) - 1) * 100, 3)
        results.append([corpus, dataset, switch_rate])

    switch_rates = (
        pd.DataFrame(results, columns=group_cols + ["prompt_switch_rate"])
        .groupby("dataset")
        .agg({"prompt_switch_rate": "median"})
        .reset_index()
    )
    return switch_rates


def filter_tasks(
    df: pd.DataFrame,
    monotonicity_threshold: float,
    snr_threshold: float,
    mad_threshold: float,
    cv_threshold: float,
):
    """
        Filter tasks that satisfy the criteria and statistics thresholds.
    Args:
        df: Dataframe with summarized signal criteria and statistics for each task based on aggregated results.
        monotonicity_threshold: Minimum required monotonicity value.
        snr_threshold: Minimum required SNR.
        mad_threshold: Maximum required MAD.
        cv_threshold: Maximum required CV.
    Returns: Filtered subset including only tasks that meet all thresholds.
    """

    return df[
        (df["monotonicity"] >= monotonicity_threshold)
        & (df["snr"] >= snr_threshold)
        & (df["mad"] <= mad_threshold)
        & (df["cv"] <= cv_threshold)
    ]


def normalize_within_range(score, lower_bound=0, higher_bound=100):
    """
        Normalize an aggregated performance score into the [0, 100] range (min-max normalization).

    Args:
        score: Aggregated performance score.
        lower_bound: Minimum expected score for the task (the random baseline performance).
        higher_bound: Maximum expected score for the task.
    Returns: Min-max normalized score.
    """

    return (np.clip(score - lower_bound, 0, None)) / (higher_bound - lower_bound) * 100


def get_normalized_results(
    raw_results: pd.DataFrame,
    score_col: str,
    monotonicity_threshold: float,
    snr_threshold: float,
    mad_threshold: float,
    cv_threshold: float,
    thresholds: dict,
    higher_bound: int = 100,
    keep_tasks: list = [],
):
    """
        Perform task filtering, score normalization, and result aggregation across task categories for a given language.

    Args:
        raw_results: Raw results dataframe.
        score_col: Column name for the aggregated score (e.g., ``"max_score"``, ``"median_score"``, or ``"mean_score"``)
        monotonicity_threshold: Minimum required monotonicity value.
        snr_threshold: Minimum required SNR.
        mad_threshold: Maximum required MAD.
        cv_threshold: Maximum required CV.
        higher_bound: Maximum expected score for the task.
        keep_tasks: Optional list of generative tasks to keep regardless of thresholds.
    Returns: Criteria dataframe, normalized score dataframe, and retained tasks.
    """
    aggregated_results = aggregate_results(raw_results)
    criteria_results = compute_criteria(
        raw_results, aggregated_results, score_col, thresholds
    )
    filtered_tasks = filter_tasks(
        criteria_results,
        monotonicity_threshold=monotonicity_threshold,
        snr_threshold=snr_threshold,
        mad_threshold=mad_threshold,
        cv_threshold=cv_threshold,
    )
    filtered_tasks = filtered_tasks["dataset"].tolist()
    if keep_tasks:
        filtered_tasks.extend(keep_tasks)
    if not filtered_tasks:
        raise ValueError("No retained tasks.")
    filtered_results = aggregated_results[
        aggregated_results["dataset"].isin(filtered_tasks)
    ]
    filtered_results["normalized_score"] = (
        (
            filtered_results[score_col]
            - filtered_results["dataset"].map(RANDOM_BASELINES)
        ).clip(lower=0)
        / (higher_bound - 0)
        * 100
    )
    task_category_means = filtered_results.groupby(
        ["corpus", "model", "category"], as_index=False
    )["normalized_score"].mean()
    normalized_results = task_category_means.groupby(
        ["corpus", "model"], as_index=False
    )["normalized_score"].mean()
    return criteria_results, normalized_results, filtered_tasks
