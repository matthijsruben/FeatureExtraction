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

Enter the pipenv shell (`pipenv shell`) and execute:

```shell
python -m src.main
```

This will assume you want the feature files to be written to `./output/features`. If you instead want them to be written
to a different folder, add an argument:

```shell
python -m src.main <output directory>
```
