# Aggregate Summary of Key Comparisons

Below are 10 major comparisons using the aggregated summary data.

## Table of Contents
- [Comparison 1](#comparison-1)
- [Comparison 2](#comparison-2)
- [Comparison 3](#comparison-3)
- [Comparison 4](#comparison-4)
- [Comparison 5](#comparison-5)
- [Comparison 6](#comparison-6)
- [Comparison 7](#comparison-7)
- [Comparison 8](#comparison-8)
- [Comparison 9](#comparison-9)
- [Comparison 10](#comparison-10)

## Comparison 1

**Comparison 1: Accuracy by Model/Prompt**
This reveals how `prediction_accuracy` differs across `model_name` and `prompt_type`. Bar and strip plots illustrate the distribution and relative performance.

## Comparison 2

**Comparison 2: F1 by Model/Prompt**
Shows how different prompt types and models fare with respect to F1, balancing precision and recall.

## Comparison 3

**Comparison 3: Exec Time vs. Accuracy**
Visualizes trade-offs between speed and correctness. The second chart includes a regression line.

## Comparison 4

**Comparison 4: Precision vs. Recall**
Examines the trade-off between precision and recall. The second plot color-codes points by `f1_score`.

## Comparison 5

**Comparison 5: AUC-ROC by Model/Prompt**
Shows how well the model ranks positive vs. negative across different prompts.

## Comparison 6

**Comparison 6: Mean vs. Median Exec Time**
Points far from the y=x line indicate skew or outliers in runtime.

## Comparison 7

**Comparison 7: Execution Time SD**
Larger SD implies more variance in per-attempt runtimes.

## Comparison 8

**Comparison 8: total_attempts vs. Accuracy**
Verifies how the sample size per model relates to the final accuracy.

## Comparison 9

**Comparison 9: Load/Prompt/Eval Durations**
Shows how time is divided among loading, prompt evaluation, and final eval for each model & prompt.

## Comparison 10

**Comparison 10: txt_accuracy vs. numeric accuracy**
Checks whether textual accuracy from the report matches numeric calculations.

