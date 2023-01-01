#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using
    # this particular version of the artifact
    logger.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Drop outliers
    logger.info("Dropping outliers in the price column")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Restrict long and lat
    idx_coord = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx_coord].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Save the results to a CSV file
    logger.info("Save the results to CSV")
    df.to_csv(args.output_artifact, index=False)

    # Build artifact
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    logger.info("Log artifact: clean_sample.csv")
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the artifact to do preprocessing on",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the clean artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="clean_sample",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the preprocessed data",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Min price considered for the prediction column",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Max price considered for the prediction column",
        required=True
    )

    args = parser.parse_args()

    go(args)
