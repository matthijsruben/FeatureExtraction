# Feature Extractor for IAM Handwriting Database

## Usage
The IAM Handwriting Database itself is not included in the repository. To properly execute the feature preprocessing and
extraction, ensure `<repository root>/data/lines/lines.tgz` exists.

The easiest way to run this code on your own machine is by using [Pipenv](https://pypi.org/project/pipenv/).

If you have installed Pipenv on your machine, install the required packages by exeucuting the following command in the
root directory of this repository:

```shell
pipenv install
 ```

Afterwards, every time you want to execute Python-related commands in this project, simply make sure to perform these in
the Pipenv shell. Active the Pipenv shell by using:
```shell
pipenv shell
```

Read the Pipenv documentation for information on how to add new requirements to this project.
