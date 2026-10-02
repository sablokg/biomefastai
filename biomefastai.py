"""
Train a logistic-regression-style tabular model with fastai, driven by a Click CLI.

Usage examples:
    python train.py --data-path data.csv
    python train.py --data-path data.csv --epochs 50 --lr 0.005 --bs 128 --valid-pct 0.25
    python train.py --help
"""

import click
import pandas as pd
from fastai.tabular.all import (
    TabularDataLoaders,
    tabular_learner,
    RandomSplitter,
    range_of,
    CategoryBlock,
    CrossEntropyLossFlat,
    ClassificationInterpretation,
)


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the input CSV file.",
)
@click.option(
    "--target",
    "target_col",
    default="target",
    show_default=True,
    help="Name of the target column.",
)
@click.option(
    "--valid-pct",
    default=0.2,
    show_default=True,
    type=float,
    help="Fraction of rows held out for validation.",
)
@click.option("--seed", default=42, show_default=True, type=int, help="Random seed for the split.")
@click.option("--bs", default=64, show_default=True, type=int, help="Batch size.")
@click.option("--epochs", default=100, show_default=True, type=int, help="Number of training epochs.")
@click.option("--lr", default=0.01, show_default=True, type=float, help="Max learning rate for fit_one_cycle.")
@click.option(
    "--layers",
    default="",
    show_default=True,
    help="Comma-separated hidden layer sizes, e.g. '200,100'. Leave empty for plain logistic regression.",
)
@click.option(
    "--plot/--no-plot",
    default=True,
    show_default=True,
    help="Whether to display the confusion matrix plot.",
)
def main(data_path, target_col, valid_pct, seed, bs, epochs, lr, layers, plot):
    """Train a tabular classifier on DATA_PATH and report validation metrics."""

    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------
    df = pd.read_csv(data_path)

    if target_col not in df.columns:
        raise click.UsageError(
            f"Target column '{target_col}' not found in {data_path}. "
            f"Available columns: {list(df.columns)}"
        )

    df[target_col] = df[target_col].astype(int)

    click.echo("Dataset shape:")
    click.echo(df.shape)
    click.echo("\nTarget values:")
    click.echo(df[target_col].value_counts())

    # ------------------------------------------------------------------
    # 2. DEFINE FEATURES
    # ------------------------------------------------------------------
    cat_names = []  # no categorical input features
    cont_names = [c for c in df.columns if c != target_col]

    # ------------------------------------------------------------------
    # 3. TRAIN / VALIDATION SPLIT
    # ------------------------------------------------------------------
    splits = RandomSplitter(valid_pct=valid_pct, seed=seed)(range_of(df))
    click.echo(f"\nTraining rows: {len(splits[0])}")
    click.echo(f"Validation rows: {len(splits[1])}")

    # ------------------------------------------------------------------
    # 4. CREATE DATALOADERS
    # ------------------------------------------------------------------
    dls = TabularDataLoaders.from_df(
        df,
        path=".",
        y_names=target_col,
        cat_names=cat_names,
        cont_names=cont_names,
        splits=splits,
        bs=bs,
        y_block=CategoryBlock,
    )

    # ------------------------------------------------------------------
    # 5. CREATE MODEL
    # ------------------------------------------------------------------
    layer_sizes = [int(x) for x in layers.split(",") if x.strip()] if layers else []

    learn = tabular_learner(
        dls,
        layers=layer_sizes,
        loss_func=CrossEntropyLossFlat(),
        metrics="accuracy",
    )

    # ------------------------------------------------------------------
    # 6. CHECK MODEL CONFIGURATION
    # ------------------------------------------------------------------
    click.echo(f"\nNumber of classes: {dls.c}")
    click.echo(f"Loss function: {learn.loss_func}")
    click.echo(f"Layers: {layer_sizes if layer_sizes else '[] (plain logistic regression)'}")

    # ------------------------------------------------------------------
    # 7. TRAIN MODEL
    # ------------------------------------------------------------------
    learn.fit_one_cycle(epochs, lr)

    # ------------------------------------------------------------------
    # 8. VALIDATE MODEL
    # ------------------------------------------------------------------
    results = learn.validate()
    click.echo(f"\nValidation loss: {results[0]}")
    click.echo(f"Validation accuracy: {results[1]}")

    # ------------------------------------------------------------------
    # 9. CONFUSION MATRIX
    # ------------------------------------------------------------------
    interp = ClassificationInterpretation.from_learner(learn)
    click.echo("\nClassification report:")
    interp.print_classification_report()
    if plot:
        interp.plot_confusion_matrix()


if __name__ == "__main__":
    main()