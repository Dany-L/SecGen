import argparse
import json
import os

from crnn.configuration import (CONFIG_FILE_ENV_VAR, ExperimentConfig,
                                ExperimentTemplate)


def main(directory: str) -> None:
    with open(os.path.expanduser(os.environ[CONFIG_FILE_ENV_VAR]), mode="r") as f:
        config_dict = json.load(f)

    config = ExperimentConfig.from_template(ExperimentTemplate(**config_dict))
    model_names = config.m_names
    with open(os.path.join(directory, "models.txt"), mode="w") as f:
        f.write("\n".join(model_names) + "\n")

    experiment_names = list(config.experiments.keys())
    with open(os.path.join(directory, "experiments.txt"), mode="w") as f:
        f.write("\n".join(experiment_names) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save experiment and model names.")
    parser.add_argument(
        "directory", type=str, help="Directory the text files will be stored in."
    )
    args = parser.parse_args()
    main(args.directory)
