# Post-Processed Fair Predictors

This repository contains code accompanying the paper:

Alan Mishler, Edward H. Kennedy, and Alexandra Chouldechova. **Fairness in Risk Assessment Instruments: Post-Processing to Achieve Counterfactual Equalized Odds**. 2021.  In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency (FAccT '21). Association for Computing Machinery, New York, NY, USA, 386–400. https://doi.org/10.1145/3442188.3445902

## Overview

This repository implements the methods proposed in the paper for achieving **Counterfactual Equalized Odds** (cEO) via **post-processing** of a predictor. The approach centers on fairness-aware linear programming and doubly robust estimation techniques to achieve group-conditional fairness in counterfactual outcomes.

The repo includes code to:

- Simulate data before and after the use of a risk assessment instrument designed to aid decision-making
- Estimate nuisance functions (risk, outcome, treatment)
- Post-process a binary predictor by solving a fairness-constrained linear program
- Evaluate predictive fairness and accuracy metrics for post-processed predictors
- Partially reproduce results from the paper. The Child Welfare dataset is not publicly available.

## Installation

Clone and install the repository:

```bash
git clone https://github.com/amishler/Post-Processed-Fair-Predictors
cd Post-Processed-Fair-Predictors
pip install .
```

## Repository Structure

```
counterfactualEO/
├── functions_estimation.py      # Estimation of risk/fairness LP coefficients
├── functions_simulation.py      # Data generation and simulation routines
├── functions_evaluation.py      # Metric computation for fairness/risk properties
├── functions_plotting.py        # Seaborn-based plotting utilities
├── utils.py                     # Helper functions
├── tests/                       # Unit and integration tests
├── notebooks/                   # Interactive examples
│   ├── simulations.ipynb        # Simulations demonstrating core methods
│   └── real_data_analysis.ipynb # Post-processed fair predictors using real datasets 
```

## Getting Started

For the simulations, "task 1" refers to estimating the optimal fair post-processed predictor, while "task 2" refers to estimating the performance and fairness characteristics of a fixed post-processed predictor and/or its corresponding input predictor. To run these simulations:

```python
from counterfactualEO.functions_simulation import simulate_task1, simulate_task1_metrics_to_df, simulate_task2
```

To run on real data:
```python
from counterfacturalEO.functions_estimation import fair_derived_crossfit
from counterfactualEO.functions_evaluation import metrics_post_crossfit
```

See the jupyter notebooks for example usage.

## Citation

If you use this code in your work, please cite:

```
@inproceedings{mishler2021counterfactual,
  title={Fairness in Risk Assessment Instruments: Post-Processing to Achieve Counterfactual Equalized Odds},
  author={Alan Mishler and Edward H. Kennedy and Alexandra Chouldechova},
  booktitle={Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency},
  pages={425--435},
  year={2021},
  organization={ACM},
  doi={10.1145/3442188.3445902}
}
```

## License
This project is licensed under the MIT License: https://mit-license.org.
