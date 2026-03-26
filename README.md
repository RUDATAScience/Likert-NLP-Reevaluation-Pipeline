# Text-Weighted Re-evaluation Pipeline for 5-Point Likert Scales

## Overview
This repository provides a Python pipeline to re-evaluate 5-point Likert scale survey data by incorporating Natural Language Processing (NLP) of open-ended text responses. 

It addresses the fundamental flaws of conventional 5-point evaluation aggregations—such as the "Illusion of the Law of Large Numbers" and "Macroscopic Degeneracy" (where human cognitive conflicts are overly dimensionally compressed)—by estimating the respondent's "confidence ($\beta$)" from their text inputs.

## Theoretical Background
In conventional surveys, an answer of "3 (Neutral)" can stem from two completely different cognitive processes:
1. **Cognitive Fatigue (Noise):** The respondent avoided cognitive load and chose the middle option without thought.
2. **Internal Conflict (Signal):** The respondent carefully weighed the pros and cons before arriving at a neutral stance.

Traditional simple aggregation treats these equally, causing statistical illusions. This pipeline solves this by calculating a **"Text Confidence Weight"** based on text length, noun count (specificity), and lazy response detection (e.g., "Nothing in particular"). It systematically down-weights lazy responses and protects minority signals (e.g., strong 1s or 5s) backed by enthusiastic text, revealing the true distribution of opinions.

## Features
- **Data Decomposition:** Separates 1D numerical scores from high-dimensional text data.
- **NLP Confidence Scoring:** Uses `Janome` (Japanese morphological analyzer) to detect lazy responses and measure cognitive effort.
- **Minority Signal Protection:** Applies baseline weight boosts to extreme scores (1 and 5) to prevent them from being crushed by peer pressure in large datasets.
- **Score Re-evaluation:** Merges and normalizes the weights to output a re-evaluated, bias-resistant score distribution.

## Requirements
```bash
pip install -r requirements.txt
