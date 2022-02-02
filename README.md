# Feature Extractor for IAM Handwriting Database

## Setup

The IAM Handwriting Database itself is not included in the repository. To properly execute the feature preprocessing and
extraction, ensure `<repository root>/data/lines/lines.tgz` exists.

The easiest way to run this code on your own machine is by using [Pipenv](https://pypi.org/project/pipenv/).

If you have installed Pipenv on your machine, install the required packages by exeucuting the following command in the
root directory of this repository:

```shell
pipenv install
 ```

**This assumes that you have at least Python version 3.8 installed.**

Afterwards, every time you want to execute Python-related commands in this project, simply make sure to perform these in
the Pipenv shell. Active the Pipenv shell by using:

```shell
pipenv shell
```

Read the Pipenv documentation for information on how to add new requirements to this project.

## Usage

For everything mentioned below, make sure to first enter the pipenv shell (`pipenv shell`).

### Extracting and writing features

```shell
python -m src.writer
```

This will assume you want the feature files to be written to `./output/features`. If you instead want them to be written
to a different folder, add an argument:

```shell
python -m src.preparation.writer <output directory>
```

### Hyperparameter tuning

There is a number of variables that can be defined for hyperparameter tuning. These concern the ranges that are explored
by the tuner, namely:

- The total number of trials to run (i.e. number of different configurations to try) (`MAX_TRIALS`)
- The number of epochs to run for each trial (`EPOCHS_NUM`)
- The number of steps per epoch to run (`EPOCHS_STEPS_NUM`)
- Number of dense layers: minimum, maximum, step-size (`DENSE_LAYERS_(MIN | MAX | STEP)`)
- Number of nodes in the dense layers: minimum, maximum, step-size (`DENSE_LAYER_NODES_(MIN | MAX | STEP)`)

These variables are listed at the very top of the `src.training.tuner` file (found at `src/training/tuner.py`).

Run the automatic hyperparameter tuning by executing the following command in the pipenv shell:

```shell
python -m src.training.tuner
```

