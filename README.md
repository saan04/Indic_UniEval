# Indic_UniEval

Indic_UniEval extends the UniEval framework to evaluate machine translation outputs for Indic languages. This is a trial code that does not fine tune any aspect of the existing framework but mainly aims to look at how the framework works on the Indic-English side of the summarization for a given corpora.

## Overview

This project applies UniEval, originally designed for summarization tasks, to evaluate machine translations. It processes translation outputs for multiple Indian languages and provides multi-dimensional quality scores.

## Features

- Uses UniEval's multi-dimensional evaluation metrics
- Processes batch translation outputs
- Provides scores for coherence, consistency, fluency, and relevance

## Installation

```
git clone https://github.com/saan04/Indic_UniEval.git
cd Indic_UniEval
pip install -r requirements.txt
```

## Usage

Run the evaluation script:

```
python qual.py
```

## File Structure

- `qual.py`: Main evaluation script
- `utils.py`: Utility functions
- `metric/evaluator.py`: UniEval evaluator implementation

## Acknowledgments

Based on the [UniEval framework](https://github.com/maszhongming/UniEval).
