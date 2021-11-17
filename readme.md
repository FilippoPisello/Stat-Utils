[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
# StatUtils
StatUtils builds mainly on NumPy and Pandas to offer a collection of functions and classes to perform tasks in the domain of statistics, data mining and machine learning.

The project is in expansion.

## Goal of the project
StatUtils is designed to contain shortcuts and interfaces for functionalities that are mostly already covered by some of the most famous statistical libraries like NumPy, SciPy and Sklearn.

This library is mainly a way for me to look into machine learning algorithms, coding them from scratch and improve my software design skills while working at a larger scale project.

# Getting Started
## Installation
At the moment this library has not been published to PyPI and it is thus not possible to pip install it.

To use it you should first download the files in the directory. You can do it manually or by cloning the repo:
```console
git clone https://github.com/FilippoPisello/Stat-Utils
```
## Usage
Since this is a multi-purpose package, there is no "one thing you should do". Look into its content to find what is useful for you, import it and give it a try.

# Requirements
The requirements are the following:
- Python:
  - Version >= 3.9
- Packages:
  - matplotlib >= 3.4.2
  - numpy >= 1.20.1
  - pandas >= 1.2.3
  - scikit_learn >= 1.0.1
  - scipy >= 1.7.1 *[Testing only]*

# Content
The following is the directory tree of the user-facing modules and brief description of their content.
```
├── classifiers\           --> Contains tools to perform classification
│   ├── __init__.py
│   ├── classifier.py          - General Classifier superclass
│   ├── kneighbors.py          - Classifier using kneighbors algorithm
│   └── mahalanobis.py         - Classifier using Mahalanobis distance
│
├── distances\             --> Contains various functs to calculate distance
│   ├── __init__.py
│   ├── euclidean.py           - Regular Euclidean distance
│   ├── hamming.py             - Hamming distance for non-numerical values
│   └── mahalanobis.py         - Mahalanobis distance
│
├── outliers\              --> Contains various functs to detect outliers
│   ├── __init__.py
│   ├── detectors.py           - Different criteria to define outliers
│   └── outliers_detection.py  - Tools that detect outliers using detectors
│
├── plot_tools\            --> Contains various functs to work with plots
│   ├── __init__.py
│   ├── basics.py              - Add/create/modify general plot features
│   └── extras.py              - Add/create/modify context-specific features
│
├── predictions\           --> Contains generic items to work with predictions
│   ├── prediction.py          - Algorithm-agnostic predictions representation
│   └── validation.py          - Algorithm-agnostic predictions validation
│
├── preprocessing\         --> Contains functions to clean data
│   ├── __init__.py
│   └── scaling.pyì            - Standardize arrays, series and dataframes
```
