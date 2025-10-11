# HPOTreeBoosting

Repository containing code and experiments for the "HPOTreeBoosting" project used in the accompanying publication.

Contents
- data preparation scripts (download_data.py)
- hyperparameter optimization and evaluation (methods.py, SMAC_method.py)
- experiment runners (run_default.py, run_experiments.py, run_experiment_H_SMAC.py)
- model-specific utilities (model_gp_boosting.py, utils_modelling.py)
- plotting utilities and analysis (plot_maker.py, utils_plot_maker.py, utils_plots_comparison.py)
- general utilities (utils.py)

Quick start
1. Create a Python environment and install requirements used in the paper (gpboost, optuna, smac, scikit-learn, pandas, numpy, matplotlib, seaborn, pdpbox, shap, openml).
2. Prepare data: either run `download_data.py` or place preprocessed task folders under `data/` as used by the run scripts.
3. Run experiments using the runner scripts. Example (adjust arguments):

   python run_experiments.py --suite_id 335 --task_id 361102 --seed 27225 --result_folder Results

4. Create plots by running `plot_maker.py` after aggregating Results into the expected folders.

Notes for reproducibility
- Many scripts expect a specific folder layout (see `utils_modelling.py` and `plot_maker.py` for hard-coded paths). Update paths or environment variables as needed.
- Seeds used in the study are included in `utils_modelling.py` and other modules; changing those will change results.

License & citation
If you use this code for research, please cite the associated publication. (Add citation details here.)
