import math
import sys
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from loguru import logger
from tabulate import tabulate

from mdl_mhn import ModernHN, plot_prediction_and_gold


def load_data(results_dir: Path) -> pd.DataFrame:
    data = pd.read_csv(results_dir / "results.csv")

    # Augment the data with additional columns
    data["num_test_digits"] = data["test_digits"].apply(lambda x: len(eval(x)))
    data["mdl_score"] = data["g_score"] + data["d_given_g_score"]
    data["golden_mdl_score"] = (
        data["golden_mhn_g_score"] + data["golden_mhn_d_given_g_score"]
    )

    return data


def visualize_bar_plot(data: pd.DataFrame, title: str):
    palette = sns.color_palette("husl", len(data["noise_level"].unique()))

    # Determine the grid size
    num_unique_values = data["num_test_digits"].nunique()
    grid_size = math.ceil(math.sqrt(num_unique_values))

    # Create a larger figure and set of subplots with the adjusted layout
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(4 * grid_size, 4 * grid_size)
    )

    # Flatten axes for easy indexing
    axes = axes.ravel()

    # Plot the data for each len_test_digits
    for i, (len_digits, sub_data) in enumerate(data.groupby("num_test_digits")):
        ax = axes[i]
        sns.barplot(
            data=sub_data,
            x="num_examples_per_digit",
            y="num_memories",
            hue="noise_level",
            ax=ax,
            palette=palette,
            errorbar=None,
        )
        ax.set_ylim(0, 25)
        ax.axhline(y=len_digits, color="black", linestyle="--")
        ax.set_title(f"Num Test Digits = {len_digits}")

    # Remove axes for empty plots
    for i in range(num_unique_values, grid_size * grid_size):
        axes[i].axis("off")

    fig.suptitle(title, fontsize=30)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


def visualize_line_plot(data: pd.DataFrame, title: str, should_save: bool = False):
    """
    Visualize the results as a line plot with:
    - line type representing the level of noise
    - color representing the number of training examples
    - x-axis representing the number of digits tested
    - y-axis representing the number of memories
    """
    # Use a distinct color palette
    palette = sns.color_palette("husl", len(data["num_examples_per_digit"].unique()))

    plt.figure(figsize=(15, 10))

    # Change column names
    data = data.rename(
        columns={
            "num_test_digits": "Number of Digits Tested",
            "num_memories": "Average Number of Memories",
            "num_examples_per_digit": "Examples per digit",
            "noise_level": "Noise level",
        }
    )

    # Create the line plot with the updated color palette
    sns.lineplot(
        data=data,
        x="Number of Digits Tested",
        y="Average Number of Memories",
        hue="Examples per digit",
        style="Noise level",
        markers=True,
        dashes=True,
        palette=palette,
    )

    plt.xlabel("Number of target (/ golden) digits", fontsize=32)
    plt.ylabel("Average number of memories", fontsize=32)
    plt.xticks(fontsize=29)
    plt.yticks(fontsize=29)
    plt.legend(
        fontsize=31,
        framealpha=0.5,
    )
    plt.grid(True)
    plt.tight_layout()

    if should_save:
        plt.savefig(f"{title}.svg", transparent=True)

    plt.show()


def analyze_discrepancies(data: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    data["comparison"] = "same"
    data.loc[data["mdl_score"] < data["golden_mdl_score"], "comparison"] = "lower"
    data.loc[data["mdl_score"] > data["golden_mdl_score"], "comparison"] = "higher"
    comparison_counts = data["comparison"].value_counts()
    grouped_df = data.groupby(["num_test_digits", "num_examples_per_digit"])
    summary_df = grouped_df["num_memories"].agg(["mean", "var"]).reset_index()
    return comparison_counts, summary_df


def pretty_print_table(
    data: Union[pd.DataFrame, pd.Series], headers: list[str]
) -> None:
    if isinstance(data, pd.DataFrame):
        print(tabulate(data, headers=headers, tablefmt="grid", showindex=False))
    else:
        print(tabulate(data.items(), headers=headers, tablefmt="grid"))


def get_golden_and_trained_mhn_from_row(
    results_dir: Path, row: pd.Series
) -> tuple[ModernHN, ModernHN]:
    golden_mhn_path = results_dir / row["golden_mhn_path"]
    trained_mhn_path = results_dir / row["best_mhn_path"]

    return ModernHN(torch.load(golden_mhn_path)), ModernHN(torch.load(trained_mhn_path))


def get_distances_from_trained_mhn(
    source_images: torch.Tensor, trained_mhn: ModernHN
) -> pd.DataFrame:
    """
    For each image in source_images, get its distance from the closest memory slot in trained_mhn
    Return all distances in a DataFrame
    """
    trained_indices = trained_mhn.retrieve_idx(source_images)

    trained_images = trained_mhn.memorized_patterns[trained_indices]
    trained_distances = torch.dist(source_images, trained_images, p=2)

    return pd.DataFrame(
        {
            "distances": trained_distances.numpy(),
            "trained_indices": trained_indices.numpy(),
        }
    )


def get_distances_from_mhn_data(mhn_data: pd.DataFrame, mhn_dir: Path):
    golden_data_distances = pd.DataFrame()
    training_data_distances = pd.DataFrame()

    for _, row in mhn_data.iterrows():
        _, trained_mhn = get_golden_and_trained_mhn_from_row(mhn_dir, row)
        golden_data = torch.load(mhn_dir / row["golden_data_path"])
        train_data = torch.load(mhn_dir / row["train_data_path"])

        current_golden_data_distances = get_distances_from_trained_mhn(
            golden_data, trained_mhn
        )
        current_training_data_distances = get_distances_from_trained_mhn(
            train_data, trained_mhn
        )

        # Add noise level
        current_golden_data_distances["noise_level"] = row["noise_level"]
        current_training_data_distances["noise_level"] = row["noise_level"]

        golden_data_distances = pd.concat(
            [golden_data_distances, current_golden_data_distances]
        )
        training_data_distances = pd.concat(
            [training_data_distances, current_training_data_distances]
        )

    return golden_data_distances, training_data_distances


def show_sample_mhn_results(
    data: pd.DataFrame, results_dir: Path, should_save: bool = False
):
    # Sample two sibling experiments
    correct_num_memories_row = data.sample(n=1)
    row_before = data.loc[correct_num_memories_row.index[0] - 1]
    row_after = data.loc[correct_num_memories_row.index[0] + 1]
    # Take the one with the same seed as correct_num_memories_row
    if row_before["seed"].item() == correct_num_memories_row["seed"].item():
        incorrect_num_memories_row = row_before
    elif row_after["seed"].item() == correct_num_memories_row["seed"].item():
        incorrect_num_memories_row = row_after
    else:
        raise ValueError("Could not find sibling experiment")

    correct_num_memories_row = correct_num_memories_row.squeeze()
    if (
        correct_num_memories_row["num_memories"].item()
        != correct_num_memories_row["num_test_digits"].item()
    ):
        correct_num_memories_row, incorrect_num_memories_row = (
            incorrect_num_memories_row,
            correct_num_memories_row,
        )

    # Load train data
    train_data_path = results_dir / correct_num_memories_row["train_data_path"]
    train_data = torch.load(train_data_path)

    # Check that the train data is the same for both experiments
    incorrect_train_data_path = (
        results_dir / incorrect_num_memories_row["train_data_path"]
    )
    incorrect_train_data = torch.load(incorrect_train_data_path)
    assert torch.equal(train_data, incorrect_train_data)

    # Load MHNs
    (
        correct_num_memories_golden_mhn,
        correct_num_memories_trained_mhn,
    ) = get_golden_and_trained_mhn_from_row(results_dir, correct_num_memories_row)
    plot_prediction_and_gold(
        correct_num_memories_trained_mhn,
        train_data,
        should_show=True,
        should_save=should_save,
        golden_mhn=correct_num_memories_golden_mhn,
    )
    (
        incorrect_num_memories_golden_mhn,
        incorrect_num_memories_trained_mhn,
    ) = get_golden_and_trained_mhn_from_row(results_dir, incorrect_num_memories_row)
    plot_prediction_and_gold(
        incorrect_num_memories_trained_mhn,
        train_data,
        should_show=True,
        should_save=should_save,
        golden_mhn=incorrect_num_memories_golden_mhn,
    )


def get_mdl_mhn_sample_for_paper(data: pd.DataFrame, results_dir: Path):
    """
    Get an MDL MHN result for 4 digits with 5 examples per digit
    """
    # Take the first experiment with 4 digits and 5 examples per digit and medium noise
    mdl_mhn_data = data[
        (data["num_test_digits"] == 4)
        & (data["num_examples_per_digit"] == 5)
        & (data["noise_level"] == "medium")
    ]
    mdl_mhn_row = mdl_mhn_data.iloc[0]

    golden_mhn, trained_mhn = get_golden_and_trained_mhn_from_row(
        results_dir, mdl_mhn_row
    )
    train_data = torch.load(results_dir / mdl_mhn_row["train_data_path"])
    plot_prediction_and_gold(
        trained_mhn,
        train_data,
        should_show=True,
        should_save=True,
        golden_mhn=golden_mhn,
    )


def show_sample_mdl_mhn_results(
    data: pd.DataFrame, results_dir: Path, should_save=False
):
    # Sample 2 experiments
    experiments_results = data.sample(n=2)
    golden_mhn_paths = experiments_results["golden_mhn_path"].tolist()
    trained_mhn_paths = experiments_results["best_mhn_path"].tolist()
    golden_mhns = [
        ModernHN(torch.load(results_dir / golden_mhn_path))
        for golden_mhn_path in golden_mhn_paths
    ]
    trained_mhns = [
        ModernHN(torch.load(results_dir / trained_mhn_path))
        for trained_mhn_path in trained_mhn_paths
    ]
    train_data_paths = experiments_results["train_data_path"].tolist()
    train_datas = [
        torch.load(results_dir / train_data_path)
        for train_data_path in train_data_paths
    ]

    # Plot predictions and golds
    for golden_mhn, trained_mhn, train_data in zip(
        golden_mhns, trained_mhns, train_datas
    ):
        plot_prediction_and_gold(
            trained_mhn,
            train_data,
            should_show=True,
            should_save=should_save,
            golden_mhn=golden_mhn,
        )


def plot_distance_distribution(distances, title: str, is_golden: bool):
    if is_golden:
        title = f"Golden Digits Distances from Memories - {title}"
    else:
        title = f"Training Digits Distances from Memories - {title}"

    plt.figure(figsize=(15, 10))
    sns.histplot(data=distances, x="distances", hue="noise_level", multiple="dodge")
    plt.title(title, fontsize=20)
    plt.show()

    # Print mean and median distances
    print(f"=== Distance Statistics - {title} ===")
    for noise_level in distances["noise_level"].unique():
        noise_level_data = distances[distances["noise_level"] == noise_level]
        mean = noise_level_data["distances"].mean()
        median = noise_level_data["distances"].median()
        print(f"{title} Mean distance for noise level {noise_level}: {mean}")
        print(f"{title} Median distance for noise level {noise_level}: {median}")
    print("=" * 50)


def plot_distance_distributions(
    mhn_data: pd.DataFrame, mhn_dir: Path, title: str = None
):
    golden_data_distances, training_data_distances = get_distances_from_mhn_data(
        mhn_data, mhn_dir
    )

    plot_distance_distribution(
        golden_data_distances,
        title=title,
        is_golden=True,
    )
    plot_distance_distribution(
        training_data_distances,
        title=title,
        is_golden=False,
    )


def parse_mhn_results(results_dir: Path, show_samples: bool = True):
    """
    1. Show 2 sample experiments, one with correct number of memories and one with incorrect number of memories
    2. Generate distance distribution graph
    """
    mhn_dir = results_dir / "mhn"
    if not mhn_dir.exists():
        logger.warning(f"MHN results not found in {results_dir}")
        return
    mhn_data = load_data(mhn_dir)

    if show_samples:
        show_sample_mhn_results(mhn_data, mhn_dir)
    plot_distance_distributions(
        mhn_data, mhn_dir, f"Modern Hopfield Networks - {results_dir.name}"
    )


def get_accuracies(data: pd.DataFrame) -> pd.DataFrame:
    """
    Per noise level, get the accuracy of the trained MHN by comparing num_memories and num_test_digits
    """
    accuracies = pd.DataFrame()
    for noise_level in data["noise_level"].unique():
        noise_level_data = data[data["noise_level"] == noise_level]
        num_correct = 0
        num_soft_correct = 0
        for _, row in noise_level_data.iterrows():
            if row["num_memories"] == row["num_test_digits"]:
                num_correct += 1
            elif row["num_memories"] > row["num_test_digits"]:
                num_soft_correct += 1

        hard_accuracy = num_correct / len(noise_level_data)
        soft_accuracy = (num_correct + num_soft_correct) / len(noise_level_data)
        accuracies = pd.concat(
            [
                accuracies,
                pd.DataFrame(
                    [[noise_level, hard_accuracy, soft_accuracy]],
                    columns=["noise_level", "hard_accuracy", "soft_accuracy"],
                ),
            ]
        )

    return accuracies


def parse_mdl_mhn_results(results_dir: Path, show_samples: bool = False):
    """
    1. Show sample memory map of a trained MDL MHN
    2. Visualize bar blot
    3. Visualize line plot
    4. Generate distance distribution graph
    """
    mdl_mhn_dir = results_dir / "mdlmhn"
    if not mdl_mhn_dir.exists():
        logger.warning(f"MDL MHN results not found in {results_dir}")
        return
    mdl_mhn_data = load_data(mdl_mhn_dir)

    if show_samples:
        show_sample_mdl_mhn_results(mdl_mhn_data, mdl_mhn_dir)
    visualize_bar_plot(
        mdl_mhn_data, f"MDL Modern Hopfield Networks - {results_dir.name}"
    )
    visualize_line_plot(
        mdl_mhn_data,
        f"MDL-MHN Number of Memories vs. Number of Digits Tested - {results_dir.name}",
    )
    plot_distance_distributions(
        mdl_mhn_data, mdl_mhn_dir, f"MDL Modern Hopfield Networks - {results_dir.name}"
    )

    print(f"=== MDL MHN Accuracies - {results_dir.name} ===")
    accuracies = get_accuracies(mdl_mhn_data)
    pretty_print_table(accuracies, ["Noise Level", "Hard Accuracy", "Soft Accuracy"])
    print("=" * 50)


def parse_experiment(results_dir: Path):
    parse_mhn_results(results_dir)
    parse_mdl_mhn_results(results_dir)


def main():
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    # Take experiment path as first arg
    if len(sys.argv) != 2:
        raise ValueError("Experiment path must be specified")

    full_results_path = Path(sys.argv[1])
    experiments_paths = [
        experiment_path
        for experiment_path in full_results_path.iterdir()
        if experiment_path.is_dir()
    ]
    logger.info(f"Found results for the following experiments: {experiments_paths}")

    for experiment_path in experiments_paths:
        parse_experiment(experiment_path)


if __name__ == "__main__":
    main()
