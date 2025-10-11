"""Main plotting entrypoints used to reproduce figures in the paper.

This module depends on `utils_plot_maker` and other helpers to load
experiment results and produce publication-ready figures.
"""

import json
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from functools import reduce
from matplotlib.gridspec import GridSpec
from utils_plot_maker import *
# --- Seeds and Experiment Setup ---
seeds = [
    27225, 34326, 92161, 99246, 108473, 117739, 235053, 257787, 
    89389, 443417, 572858, 620176, 671487, 710570, 773246, 936518, 32244, 147316, 777646, 778572
]
NUM_SEEDS = len(seeds)
FOLDS = [0, 1, 2, 3, 4]
NUM_FOLDS = len(FOLDS)
RANDOMNESS = ['both', 'seeds', 'tasks']

# --- Plot Appearance ---
TICKSIZE = 14
FONTSIZE = 14
TITLESIZE = 18
SUBTITLESIZE = 16
ALPHA = 0.6
ALPHA_FILL = 0.1
avg_linewidth = 2.25
linewidth_2 = 1.5
linewidth_grid = 0.5
alpha_grid = 0.3
MARKERS = ["o", "s", "^", "D", "*", "v", "P", "X"]

# --- Figure Size ---
WIDTH = 7.5
HEIGHT = 7.5

# --- Task and Metric Setup ---
NUM_TASKS = 59
NUM_REGR_TASKS = 36
NUM_CLASS_TASKS = 23
NUM_METRIC = 4
METRIC = [r"$\mathrm{R}^2$", "RMSE", "Accuracy", "Log Loss"]

# --- Iteration Setup ---
NUM_ITERS = 135
START = 45
NUM_ITERS_H = 11
START_HYPERBAND = int(START / NUM_ITERS * NUM_ITERS_H) + 1

# --- HPO Methods ---
HPO_METHODS = ['grid_search', 'random_search', 'tpe', 'gp_bo', 'hyperband', 'SMAC']
HPO_METHODS_NAMES = ['Deterministic Grid', 'Random Grid', 'TPE', 'GP-BO', 'Hyperband', 'SMAC']
NUM_METHODS = len(HPO_METHODS)

HPO_METHODS_SUBS = ['random_search', 'tpe', 'gp_bo', 'hyperband', 'SMAC']
HPO_METHODS_SUBS_NAMES = ['Random Grid', 'TPE', 'GP-BO', 'Hyperband', 'SMAC']
NUM_METHODS_SUBS = len(HPO_METHODS_SUBS)

HPO_METHODS_SUBS2 = ['random_search', 'tpe', 'gp_bo', 'SMAC']
HPO_METHODS_SUBS_NAMES2 = ['Random Grid Search', 'TPE', 'GP-BO', 'SMAC']

# --- Tuning Strategies ---
TUNING_STRAT = ['num_leaves', 'max_depth', 'joint', 'num_iter']
NAME_TUNING_STRAT = ['Num Leaves', 'Max Depth', 'Joint', 'Num Iter']
NUM_TUNING_STRAT = len(TUNING_STRAT)
NUM_TUNING_STRAT_GS = 2

TUNING_STRAT_SUBS = ['num_leaves', 'max_depth', 'joint']
NAME_TUNING_STRAT_SUBS = ['Num Leaves', 'Max Depth', 'Joint']

def compare_method(result_dict,percentiles, TUNING, randomness = None, normalize = True, relative_comparison = False, Test = None):
    """
    Create the main comparison figure that visualizes metrics for each HPO method.

    Parameters
    ----------
    result_dict : dict
        Dictionary mapping metric names to dictionaries mapping method names to result arrays and additional boolean keys to check if loss or already normalized.
    percentiles : tuple
        Percentiles used for normalization and/or confidence intervals.
    TUNING : str
        Tuning strategy to display (e.g. 'num_leaves').
    randomness : {'task','seeds','both' None}
        Which source of randomness to aggregate across.
    confidence_interval : bool
        Whether to plot percentile-based confidence intervals.
    normalize : bool
        Whether to normalize results before plotting.
    relative_comparison : bool
        Whether to plot results relative to the best observed extremum.
    Test : bool or None
        Internal flag for test output naming.
    """
    sns.set_style('white')
    res = copy.deepcopy(result_dict)
    if normalize:
        for metric, result in res.items():
            res[metric] = normalize_scores(result, percentiles)
        res_normalize = copy.deepcopy(res)
    index_tuning = TUNING_STRAT.index(TUNING)
    #define bool to check wether to consider grid_search in plot
    add_grid_search = not (index_tuning > TUNING_STRAT.index('max_depth'))
    add_hyperband = not (index_tuning > TUNING_STRAT.index('joint'))
    if not add_grid_search:
        for metric in res:
            del res[metric]['grid_search']
        hpo_methods = HPO_METHODS_SUBS
        hpo_method_names = HPO_METHODS_SUBS_NAMES
        if not add_hyperband:
            for metric in res:
                del res[metric]['hyperband']
            hpo_methods = HPO_METHODS_SUBS2
            hpo_method_names = HPO_METHODS_SUBS_NAMES2
    if add_grid_search and add_hyperband:
        hpo_methods = HPO_METHODS
        hpo_method_names = HPO_METHODS_NAMES
    palette = set_plot_theme(len(HPO_METHODS_NAMES)+1)
    

    subplot_size = WIDTH / 2  # 2 columns, so each subplot gets half the width
    fig_height = subplot_size * 2  # 2 rows, maintain square aspect ratio
    
    fig, axes = plt.subplots(2,2, figsize=(fig_height, fig_height))
    axes = axes.flatten()
    if add_hyperband:
        NUM_ITERS_H = res['R2']['hyperband'].shape[1]
    leg_default  = 0
    for metric in res:
        if metric == 'R2':
            k=0  
        elif metric == 'Accuracy':
            k=2
        elif metric == 'RMSE':
            k=1
        else:
            k=3
        ax = axes[k]  # Select the appropriate subplot
        ax.set_box_aspect(1)
        if relative_comparison:
            extremum = {}
            if normalize:
                extremum = get_extremum(res_normalize, hpo_methods,index_tuning)
            else:
                extremum = get_extremum(result_dict, hpo_methods,index_tuning)
        for method in hpo_methods:
            idx = HPO_METHODS.index(method)
            if  method != 'hyperband':
                iterations = np.arange(NUM_ITERS)
                marker_interval = 12  
                marker_offset = 5*idx  
                marker_every = (marker_offset,marker_interval)
                start = START
            else:
                iterations = np.linspace(0,NUM_ITERS-1,NUM_ITERS_H)
                marker_interval = 1  
                marker_offset = 0  
                marker_every = (marker_offset,marker_interval)
                start = START_HYPERBAND
            if relative_comparison:
                m, res[metric][f'avg_std_{method}'] = compute_avg_std(res, metric, method, index_tuning,randomness, relative_comparison, extremum)
            else:
                m, res[metric][f'avg_std_{method}'] = compute_avg_std(res, metric, method, index_tuning,randomness)
            res[metric][method] = np.mean(res[metric][method],axis=(2,3))[index_tuning,:]
            if relative_comparison:
                res[metric][method] = res[metric][method]/extremum[metric]
            ax.plot(
                iterations,
                res[metric][method],
                color=palette[idx],
                label=hpo_method_names[hpo_methods.index(method)],
                linewidth=avg_linewidth,
                marker=MARKERS[idx],
                markersize=8,  # Slightly larger for better visibility
                markevery=marker_every,   # Staggered markers with offset
                markerfacecolor=palette[idx],
            )
            if randomness is not None:
                plot_confidence_interval(ax,iterations,res[metric][method],res[metric][f'avg_std_{method}'],m,palette[idx])
        
        y_min, y_max = get_limit(res, metric, hpo_methods, start,iterations, m,randomness, debug = False)
        # Set y-limits with proper bounds
        ax.set_ylim(bottom=y_min, top=y_max)
        if relative_comparison:
            leg_default = print_default(np.mean(res[metric]['Default'],-1)[0]/extremum[metric], metric, y_min, y_max, palette, ax, k,leg_default)
        else:
            leg_default = print_default(np.mean(res[metric]['Default'],-1)[0],metric, y_min, y_max, palette, ax, k,leg_default)
        # Set x-limits to start from the specified iteration
        ax.set_xlim(left=START, right=NUM_ITERS-1)
        
        # Adjust x-tick labels to correspond to actual iterations
        num_x_ticks = 7  # Increased from 6 for better granularity
        # Position ticks within the visible x-axis range
        tick_positions = np.linspace(START, NUM_ITERS - 1, num_x_ticks)
        tick_labels = np.linspace(START, NUM_ITERS, num_x_ticks, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(label) for label in tick_labels])
        
        for spine in ax.spines.values():
            spine.set_edgecolor('dimgray')
            spine.set_linewidth(1)
        ax.grid(True, color='lightgray', linestyle='--', linewidth=linewidth_grid, alpha=alpha_grid)
        ax.tick_params(axis='both', which='major', labelsize=TICKSIZE)
        title = METRIC[k]
        ax.set_title(title, fontsize=SUBTITLESIZE)
        
    lines, labels = axes[leg_default].get_legend_handles_labels()
    labels, lines = zip(*sorted(zip(labels, lines), key=lambda t: t[0]))
    # Determine number of columns for legend (max 3 per row for better readability)
    max_cols_per_row = 3
    if len(labels) == 3 or len(labels) == 4:
        ncol = len(labels)
    else:
        ncol = max_cols_per_row
    
    # Reorder labels for column-first filling when we have multiple rows
    if len(labels) > ncol:
        # Calculate number of rows needed
        nrows = (len(labels) + ncol - 1) // ncol
        
        # Reorder labels and lines for column-first arrangement
        reordered_lines = []
        reordered_labels = []
        
        for col in range(ncol):
            for row in range(nrows):
                idx = row * ncol + col
                if idx < len(labels):
                    reordered_lines.append(lines[idx])
                    reordered_labels.append(labels[idx])
        
        lines = reordered_lines
        labels = reordered_labels
    
    fig.legend(
        lines, labels,  # Use the actual labels from the plots
        loc='lower center',
        ncol=ncol,
        fontsize=FONTSIZE,
        bbox_to_anchor=(0.5, -0.05),  # Place below the plots
        frameon=True,
        fancybox=True,
        shadow=False
    )
    
    # Adjust layout to accommodate legend below
    plt.tight_layout()
    if len(labels) <= 4:
        plt.subplots_adjust(bottom=0.06, left=0.05)  # Reduced left margin for smaller gap
    elif len(labels) <=6:
        plt.subplots_adjust(bottom=0.10, left=0.05)  # Reduced left margin for smaller gap
    else:
        plt.subplots_adjust(bottom=0.13, left=0.05)  # Reduced left margin for smaller gap
    
    
    # Add row titles on the left-hand side with smaller margin
    # Get the actual subplot positions for better alignment
    top_row_center = (axes[0].get_position().y0 + axes[0].get_position().y1) / 2
    bottom_row_center = (axes[2].get_position().y0 + axes[2].get_position().y1) / 2
    
    fig.text(0.00, top_row_center, 'Regression', rotation=90, fontsize=SUBTITLESIZE, 
             ha='center', va='center', weight='bold', color='black')
    fig.text(0.00, bottom_row_center, 'Classification', rotation=90, fontsize=SUBTITLESIZE, 
             ha='center', va='center', weight='bold', color='black')
    if Test:
        plt.title(f"Quantiles {percentiles[0]},{percentiles[1]}")
    for i, ax in enumerate(axes):
        bbox = ax.get_position()
        width = bbox.width
        height = bbox.height
        print(f"Subplot {i}: width={width:.3f}, height={height:.3f}, ratio={width/height:.3f}")
    
    save_plot(plt, "compare_method", randomness,relative_comparison,
              normalize, TUNING=TUNING, test = Test, percentiles=percentiles)    

def compare_tuning(result_dict,percentiles, randomness = None, normalize = True):
    """
    Plot comparison across different tuning settings for each HPO method for all tasks.

    Parameters
    ----------
    result_dict : dict
        Dictionary mapping metric names to dictionaries mapping method names 
        to result arrays and additional boolean keys to check if loss or already normalized.
    percentiles : tuple
        Percentiles used for normalization.
    randomness : {'task','seeds', None}
        Which source of randomness to aggregate across.
    confidence_interval : bool
        Whether to display percentile-based confidence intervals.
    normalize : bool
        Whether to normalize the input results before plotting.
    """
    sns.set_style('white')
    res = copy.deepcopy(result_dict)
    if normalize:
        for metric, result in res.items():
            res[metric] = normalize_scores(result, percentiles)
    for metric in res:
        del res[metric]['grid_search']
    #define bool to check wether to consider grid_search in plot
    
    
    metrics = ['R2','RMSE','Accuracy','Logloss']
    metrics_names = [r"$\mathrm{R}^2$",'RMSE','Accuracy','Log Loss']
    
    palette = set_plot_theme(NUM_TUNING_STRAT+1)
    fig, axes = plt.subplots(5,4, figsize=(2*WIDTH,2.5*HEIGHT))
    axes = axes.flatten()
    NUM_ITERS_H = res['R2']['hyperband'].shape[1]

    k=0
    leg_default = 0
    for METHOD in HPO_METHODS_SUBS:
        if  METHOD != 'hyperband':
            iterations = np.arange(NUM_ITERS)
            tuning_strat = TUNING_STRAT
            name_tuning_strat = NAME_TUNING_STRAT
            start = START
        else:
            iterations = np.linspace(0,NUM_ITERS-1,NUM_ITERS_H)
            tuning_strat = TUNING_STRAT_SUBS
            name_tuning_strat = NAME_TUNING_STRAT_SUBS
            start = START_HYPERBAND
        for metric in metrics:
            ax = axes[k]  # Select the appropriate subplot              
            ax.set_box_aspect(1)
            
            # Collect all y-data for dynamic limit calculation
            all_y_min = []
            all_y_max = []
            all_x = []
            
            for j,tuning in enumerate(tuning_strat):
                #make a case distinction because there are only NUM_ITERS_H datapoints for Hyperband
                m, res[metric][f'avg_std_{tuning}'] = compute_avg_std(res, metric, METHOD, index_tuning=j, randomness = randomness)
                print(f'randomnomess {randomness} and m {m}')
                # Store data for limit calculation
                all_x.append(iterations)
                # Staggered markers: each method gets markers at different x-positions
                if METHOD == 'hyperband':
                    marker_interval = 1
                    marker_offset = 0  
                    marker_every = (marker_offset,marker_interval)
                else:
                    marker_interval = 12 
                    marker_offset = 5*j  
                    marker_every = (marker_offset,marker_interval)
                ax.plot(
                    iterations,
                    np.mean(res[metric][METHOD][j,:,:,:],axis=(1,2)),
                    color=palette[j],
                    label=name_tuning_strat[j],
                    linewidth= avg_linewidth,
                    marker=MARKERS[j],
                    markersize=8,  
                    markevery=marker_every,   
                    markerfacecolor=palette[j]
                )
                if randomness is not None:
                    plot_confidence_interval(ax,iterations,np.mean(res[metric][METHOD][j,:,:,:],axis=(1,2)),res[metric][f'avg_std_{tuning}'],m,color=palette[j])
                all_y_max.append(np.mean(res[metric][METHOD][j,:,:,:],axis=(1,2))[start:] + (2*res[metric][f'avg_std_{tuning}'][start:])/(np.sqrt(m)))# ,-1 if (res[metric]['rmse'] or res[metric]['logloss']) else 0,1))
                all_y_min.append(np.mean(res[metric][METHOD][j,:,:,:],axis=(1,2))[start:] - (2*res[metric][f'avg_std_{tuning}'][start:])/(np.sqrt(m)))# ,-1 if (res[metric]['rmse'] or res[metric]['logloss']) else 0,1))
                
            for spine in ax.spines.values():
                spine.set_edgecolor('dimgray')  # Set the desired color here
                spine.set_linewidth(1)  # Optionally, adjust the thickness

            # Customize grid
            ax.grid(True, color='lightgray', linewidth=linewidth_grid, alpha = alpha_grid)

            
            
            
            # Enhanced y-axis limits: focus on most important parts of the plots
            all_y_flat_min = np.concatenate(all_y_min)
            all_y_flat_max = np.concatenate(all_y_max)
            # Set limits based on metric type
            if metric in ['R2', 'Accuracy']:
                y_min = np.min(all_y_flat_min)
                y_max = np.max(all_y_flat_max)
                #print(f'We use the Method {METHOD} with TUNING {tuning} and metric {metric}, the bounds are {y_min} and {y_max}.')
            else:  # 'RMSE', 'Logloss'
                y_min = np.min(all_y_flat_min)
                y_max = np.max(all_y_flat_max)
                #print(f'We use the Method {METHOD} with TUNING {tuning} and metric {metric}, the bounds are {y_min} and {y_max}.')
                
            
            ax.set_ylim(y_min, y_max)  # Use calculated dynamic limits
            
            leg_default = print_default(np.mean(res[metric]['Default'],-1)[0],metric, y_min, y_max, palette, ax, k,leg_default)
            # Set x-limits to start from the specified iteration
            ax.set_xlim(left=START, right=NUM_ITERS-1)
            
            # Adjust x-tick labels to correspond to actual iterations
            num_x_ticks = 7  # Increased from 6 for better granularity
            # Position ticks within the visible x-axis range
            tick_positions = np.linspace(START, NUM_ITERS - 1, num_x_ticks)
            tick_labels = np.linspace(START, NUM_ITERS, num_x_ticks, dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(label) for label in tick_labels])
            ax.tick_params(axis='both', which='major', labelsize=TICKSIZE)
            if k==0 or k==1 or k==2 or k==3:
                ax.set_title(f'{metrics_names[k]}', fontsize=SUBTITLESIZE)
            if k%4 ==0:            
                title = HPO_METHODS_SUBS_NAMES[HPO_METHODS_SUBS.index(METHOD)]
                ax.set_ylabel(f'{title}', fontsize=SUBTITLESIZE)
            k += 1

    lines, labels = axes[leg_default].get_legend_handles_labels()
    lines2, labels2 = axes[0].get_legend_handles_labels()
    lines += lines2
    labels += labels2
    # Remove duplicates from labels and lines while preserving order
    # Remove duplicates from labels and lines while preserving order
    unique = {}
    for lbl, ln in zip(labels, lines):
        if lbl not in unique:
            unique[lbl] = ln
    labels = list(unique.keys())
    lines = [unique[lbl] for lbl in labels]
    labels, lines = zip(*sorted(zip(labels, lines), key=lambda t: t[0]))
    # Determine number of columns for legend (max 3 per row for better readability)
    max_cols_per_row = 4
    if len(labels) <= max_cols_per_row:
        ncol = len(labels)
    else:
        ncol = max_cols_per_row-1
    
    # Reorder labels for column-first filling when we have multiple rows
    if len(labels) > ncol:
        # Calculate number of rows needed
        nrows = (len(labels) + ncol - 1) // ncol
        
        # Reorder labels and lines for column-first arrangement
        reordered_lines = []
        reordered_labels = []
        
        for col in range(ncol):
            for row in range(nrows):
                idx = row * ncol + col
                if idx < len(labels):
                    reordered_lines.append(lines[idx])
                    reordered_labels.append(labels[idx])
        
        lines = reordered_lines
        labels = reordered_labels
    
    fig.legend(
        lines, labels,
        loc='lower center',
        ncol=ncol,
        fontsize=FONTSIZE,
        bbox_to_anchor=(0.5, 0.065),  
        frameon=True,
        fancybox=True,
        shadow=False
    )
    for i, ax in enumerate(axes):
        bbox = ax.get_position()
        width = bbox.width
        height = bbox.height
        print(f"Subplot {i}: width={width:.3f}, height={height:.3f}, ratio={width/height:.3f}")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    save_plot(plt,"compare_tuning",randomness, False, normalize, False)
def compare_by_task(result_dict,names, TUNING,metric = 'R2', randomness = None, normalize = False, percentiles = None):
    """
    Generate per-task comparison figures showing performance trajectories per task.

    Parameters
    ----------
    result_dict : dict
        Dictionary mapping metric names to dictionaries mapping method names to result arrays and additional boolean keys to check if loss or already normalized.
    names : list[str]
        List of task names corresponding to the task axis.
    TUNING : str
        Tuning strategy to visualize.
    percentiles: tuple
        Percentiles used for possible normalization
    metric : str
        Metric to plot (default 'R2').
    metric_name : str
        Human-readable metric label for titles.
    randomness : {'task','seeds', None}
        Which randomness mode to aggregate across.
    normalize : bool
        Whether to normalize input results prior to plotting.
    """

    num_task = len(names)
    
    # Check if we need to split into multiple plots
    max_plots_per_figure = 24  # 4 columns x 6 rows
    need_multiple_plots = num_task > max_plots_per_figure
    
    if need_multiple_plots:
        # Split tasks into chunks of 24
        task_chunks = []
        name_chunks = []
        for i in range(0, num_task, max_plots_per_figure):
            end_idx = min(i + max_plots_per_figure, num_task)
            task_chunks.append((i, end_idx))
            name_chunks.append(names[i:end_idx])
        
        # Create plots for each chunk
        for chunk_idx, (start_idx, end_idx) in enumerate(task_chunks):
            chunk_names = name_chunks[chunk_idx]
            chunk_suffix = f"_{chunk_idx + 1}"
            
            # Create a modified result_dict for this chunk
            chunk_result_dict = copy.deepcopy(result_dict)
            for metric_key in chunk_result_dict.keys():
                if isinstance(chunk_result_dict[metric_key], dict):
                    for method_key in chunk_result_dict[metric_key].keys():
                        if isinstance(chunk_result_dict[metric_key][method_key], np.ndarray):
                            # Select only the tasks for this chunk
                            if method_key != 'Default':
                                chunk_result_dict[metric_key][method_key] = chunk_result_dict[metric_key][method_key][:, :, start_idx:end_idx, :]
                            else:
                                chunk_result_dict[metric_key][method_key] = chunk_result_dict[metric_key][method_key][:,start_idx:end_idx]
            # Call the plotting function for this chunk
            _plot_single_figure(chunk_result_dict, chunk_names, TUNING, metric, 
                              randomness, normalize, chunk_suffix,percentiles)
    else:
        # Use original logic for single plot
        _plot_single_figure(result_dict, names, TUNING, metric, 
                          randomness, normalize, "",percentiles)

def _plot_single_figure(result_dict, names, TUNING, metric, randomness, normalize, suffix, percentiles):
    """
    Helper to draw a single multi-panel figure (up to 24 tasks) for `compare_by_task`.

    Parameters
    ----------
    result_dict, percentiles, names, TUNING, metric, randomness,
    confidence_interval, normalize, suffix : see `compare_by_task` for meanings.
    """
    
    num_task = len(names)
    sns.set_style('white')
    res = copy.deepcopy(result_dict)
    if normalize:
        for metrics, result in res.items():
            res[metrics] = normalize_scores(result, percentiles)
    index_tuning = TUNING_STRAT.index(TUNING)
    #define bool to check wether to consider grid_search in plot
    add_grid_search = not (index_tuning > TUNING_STRAT.index('max_depth'))
    add_hyperband = not (index_tuning > TUNING_STRAT.index('joint'))
    if not add_grid_search:
        for metrics in res:
            del res[metrics]['grid_search']
        hpo_methods = HPO_METHODS_SUBS
        hpo_method_names = HPO_METHODS_SUBS_NAMES
        if not add_hyperband:
            for metrics in res:
                del res[metrics]['hyperband']
            hpo_methods = HPO_METHODS_SUBS2
            hpo_method_names = HPO_METHODS_SUBS_NAMES2
    if add_grid_search and add_hyperband:
        hpo_methods = HPO_METHODS
        hpo_method_names = HPO_METHODS_NAMES
    palette = set_plot_theme(len(HPO_METHODS_NAMES)+1)
    
    # Always use 4 columns, calculate rows needed
    num_cols = 4
    if num_task % num_cols > 0:
        num_rows = num_task // num_cols + 1
    else:
        num_rows = num_task // num_cols
    
    # Consistent figure sizing with other functions
    fig_width = WIDTH * 2  # Consistent with original sizing approach
    fig_height = (num_rows / 2) * HEIGHT  # Proportional to number of rows
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()
    
    # Consistent layout adjustments with other functions
    plt.tight_layout()
    plt.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.16)  # Leave space for legend
    if add_hyperband:
        NUM_ITERS_H = res['R2']['hyperband'].shape[1]        
    
    # Process data for all methods first
    all_data_for_tasks = [[] for _ in range(num_task)]  # Store all data points for each task
    num_iters = NUM_ITERS -START
    print(hpo_methods)
    for method in hpo_methods:
        m, res[metric][f'avg_std_{method}'] = compute_avg_std(res,metric, method, index_tuning,
                                                              randomness, per_task = True)
        print(f"Shape of avg_std for method {method}", res[metric][f'avg_std_{method}'].shape)
        res[metric][method] = np.mean(res[metric][method],axis=3)[index_tuning,:,:]
        
        # Collect all data points for y-axis calculation
        for k in range(num_task):
            # Main line data
            #print(k, len(all_data_for_tasks))
            #print(res[metric][method].shape, metric, method)
            all_data_for_tasks[k].extend(res[metric][method][(START if method != 'hyperband' else START_HYPERBAND):,k])

    
    # Calculate y-axis limits for each task
    y_limits = []
    for k in range(num_task):
        if all_data_for_tasks[k]:
            data_array = np.array(all_data_for_tasks[k])
            data_array = data_array[np.isfinite(data_array)]  # Remove NaN/inf values
            
            if len(data_array) > 0:
                y_min = np.min(data_array)
                y_max = np.max(data_array)
                
                # Add some padding (5% of the range)
                y_range = y_max - y_min
                padding = y_range * 0.05 if y_range > 0 else 0.1
                
                y_min_padded = y_min - padding
                y_max_padded = y_max + padding
                y_limits.append((y_min_padded, y_max_padded))
            else:
                # Fallback to default limits
                if metric == 'RMSE' or metric == 'Logloss':
                    y_limits.append((-1, 1))
                else:
                    y_limits.append((0, 1))
        else:
            # Fallback to default limits
            if metric == 'RMSE' or metric == 'Logloss':
                y_limits.append((-1, 1))
            else:
                y_limits.append((0, 1))
    
    # Now plot the data
    leg_default = 0
    method_count = -1
    for method in hpo_methods:
        if  method != 'hyperband':
            iterations = np.arange(num_iters)
        else:
            iterations = np.linspace(0,num_iters-1,NUM_ITERS_H-START_HYPERBAND)
        for k in range(num_task):    
            ax = axes[k]  # Select the appropriate subplot
            ax.set_box_aspect(1)
            
            # Use task-specific y-limits for clipping
            y_min_task, y_max_task = y_limits[k]
            ax.plot(
                iterations,
                np.clip(res[metric][method][(START if method != 'hyperband' else START_HYPERBAND):,k], y_min_task, y_max_task),
                color=palette[HPO_METHODS.index(method)],
                label=hpo_method_names[hpo_methods.index(method)],
                linewidth=avg_linewidth
                
            )
            if randomness is not None:
                plot_confidence_interval(ax,iterations, res[metric][method][(START if method != 'hyperband' else START_HYPERBAND):,k],
                                          res[metric][f'avg_std_{method}'][(START if method != 'hyperband' else START_HYPERBAND):,k],m, color=palette[HPO_METHODS.index(method)])
            for spine in ax.spines.values():
                spine.set_edgecolor('dimgray')  # Set the desired color here
                spine.set_linewidth(1)  # Optionally, adjust the thickness

            # Customize grid
            ax.grid(True, color='lightgray', linewidth=linewidth_grid, alpha = alpha_grid)

            # Set the y-axis to have at most 5 ticks
            ax.tick_params(axis='both', which='major', labelsize=TICKSIZE)
            
            # Set task-specific y-limits
            ax.set_ylim(y_limits[k])
            #print('Default', res[metric]['Default'][0,k])
            if method_count == -1:
                leg_default = print_default(res[metric]['Default'][0,k],metric, y_min_task, y_max_task, palette, ax, k, leg_default)
            title = names[k]
            ax.set_title(title, fontsize=SUBTITLESIZE)  # Consistent with other functions
                        
            # Adjust x-tick labels to correspond to actual iterations
            num_x_ticks = 7 
            # Position ticks within the visible x-axis range
            tick_positions = np.linspace(0, NUM_ITERS - 1 - START, num_x_ticks)
            tick_labels = np.linspace(START, NUM_ITERS, num_x_ticks, dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_xlim(iterations[0], iterations[-1])
            ax.set_xticklabels([str(label) for label in tick_labels])
        method_count = 0
    
    lines, labels = axes[leg_default].get_legend_handles_labels()
    labels, lines = zip(*sorted(zip(labels, lines), key=lambda t: t[0]))
    print(labels, leg_default)
    # Hide unused subplots
    for k in range(num_task, len(axes)):
        axes[k].set_visible(False)
    
    # Determine number of columns for legend (max 3 per row for better readability)
    max_cols_per_row = 4
    if len(labels) <= max_cols_per_row:
        ncol = len(labels)
    else:
        ncol = max_cols_per_row-1
    
    # Reorder labels for column-first filling when we have multiple rows
    if len(labels) > ncol:
        # Calculate number of rows needed
        nrows = (len(labels) + ncol - 1) // ncol
        
        # Reorder labels and lines for column-first arrangement
        reordered_lines = []
        reordered_labels = []
        
        for col in range(ncol):
            for row in range(nrows):
                idx = row * ncol + col
                if idx < len(labels):
                    reordered_lines.append(lines[idx])
                    reordered_labels.append(labels[idx])
        
        lines = reordered_lines
        labels = reordered_labels
    
    if num_rows <= 3:  
        bbox_y = 0.01  
        bottom_margin = 0.13
    elif num_rows <= 6:
        bbox_y = 0.01  
        bottom_margin = 0.07
    else:  # 7+ rows: tall figures need more space
        bbox_y = 0.005  # Lower position for more rows
        bottom_margin = 0.14
    
    # Adjust layout first, before adding legend
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(bottom=bottom_margin)  # Responsive bottom margin based on number of rows
    
    # Alternative legend positioning approach for better control
    fig.legend(
        lines, labels,
        loc='lower center',
        ncol=ncol,
        fontsize=FONTSIZE,
        bbox_to_anchor=(0.5, bbox_y),  # Responsive positioning based on number of rows
        frameon=True,
        fancybox=True,
        shadow=False,
        bbox_transform=fig.transFigure  # Use figure coordinates for more reliable positioning
    )
    
    for i, ax in enumerate(axes):
        bbox = ax.get_position()
        width = bbox.width
        height = bbox.height
        print(f"Subplot {i}: width={width:.3f}, height={height:.3f}, ratio={width/height:.3f}")

    save_plot(plt, "by_task", randomness, relative_comparison=False, normalize=False, test=False, TUNING=TUNING,
              metric = metric, suffix = suffix)


def test_percentiles(result_dict, TUNING, randomness, normalize, relative_comparison):
    upper_percentile = np.array([60,70,80,90])
    lower_percentile = np.array([40,30,20,10])
    for upper in upper_percentile:
        for lower in lower_percentile:
            percentile = np.array([lower, upper])
            compare_method(result_dict,percentile, TUNING, randomness, normalize, relative_comparison, Test = True)

def create_table(result_dict,percentiles,TUNING,normalize = True):
    res = copy.deepcopy(result_dict)
    if normalize:
        for metric, result in res.items():
            res[metric] = normalize_scores(result, percentiles)
        res_normalize = copy.deepcopy(res)
    index_tuning = TUNING_STRAT.index(TUNING)
    #define bool to check wether to consider grid_search in plot
    add_grid_search = not (index_tuning > TUNING_STRAT.index('max_depth'))
    add_hyperband = not (index_tuning > TUNING_STRAT.index('joint'))
    if not add_grid_search:
        for metric in res:
            del res[metric]['grid_search']
        hpo_methods = HPO_METHODS_SUBS
        hpo_method_names = HPO_METHODS_SUBS_NAMES
    if not add_hyperband:
        for metric in res:
            del res[metric]['hyperband']
        hpo_methods = HPO_METHODS_SUBS2
        hpo_method_names = HPO_METHODS_SUBS_NAMES2
    if add_grid_search and add_hyperband:
        hpo_methods = HPO_METHODS
        hpo_method_names = HPO_METHODS_NAMES
    extremum = {}
    if normalize:
        extremum = get_extremum(res_normalize, hpo_methods,index_tuning)
    else:
        extremum = get_extremum(result_dict, hpo_methods,index_tuning)
    # Prepare table data
    metrics = ['R2', 'RMSE', 'Accuracy', 'Logloss']
    stats = ['norm', 'rel', 'rank']
    table = {}
    for stat in stats:
        for metric in METRIC:
            table[f"{metric} ({stat})"] = []
    # Alphabetically order hpo_methods and hpo_method_names
    sorted_methods_with_names = sorted(zip(hpo_method_names, hpo_methods), key=lambda x: x[0])
    sorted_hpo_methods = [n  for _,n in sorted_methods_with_names]
    sorted_hpo_method_names = [n for n, _ in sorted_methods_with_names]
    tracker = True
    for method in sorted_hpo_methods:
        # For each metric, get last normalized score
        if tracker:
            norm_scores_default = []
            rel_diffs_default = []
            ranks_default = []
        norm_scores = []
        rel_diffs = []
        ranks = []
        for metric in metrics:
            if tracker:
                last_score = np.mean(res[metric]['Default'], axis=1)[-1]
                norm_scores_default.append(last_score)
            scores = np.mean(res[metric][method], axis=(2,3))[index_tuning,:]
            last_score = scores[-1]
            norm_scores.append(last_score)
        # Compute extremum for each metric
        for i, metric in enumerate(metrics):
            best_score = extremum[metric]
            if tracker:
                rel_diff = (norm_scores_default[i] - best_score) / best_score if best_score != 0 else 0
                rel_diffs_default.append(rel_diff)
            rel_diff = (norm_scores[i] - best_score) / best_score if best_score != 0 else 0
            rel_diffs.append(rel_diff)
        # Compute ranks for each metric
        for i, metric in enumerate(metrics):
            ranks_task = []
            ranks_default_task = []
            for k in range(res[metric]['Default'].shape[1]):
                all_last_scores_per_task = [np.mean(res[metric][m], axis=(3))[index_tuning,-1,k] for m in sorted_hpo_methods]
                all_last_scores_per_task.append(res[metric]['Default'][-1,k])
                if tracker:
                    if metric in ['R2', 'Accuracy']:
                        rank = len(sorted_hpo_methods)+1 - np.argsort(np.argsort(all_last_scores_per_task))[-1]
                    else:
                        rank = np.argsort(np.argsort(all_last_scores_per_task))[-1] + 1
                    ranks_default_task.append(rank)
                if metric in ['R2', 'Accuracy']:
                    rank = (len(sorted_hpo_methods)+1) - np.argsort(np.argsort(all_last_scores_per_task))[sorted_hpo_methods.index(method)]
                else:
                    rank = np.argsort(np.argsort(all_last_scores_per_task))[sorted_hpo_methods.index(method)] + 1
                ranks_task.append(rank)
            if i == 0 and  method == 'grid_search':
                ranks_temp = ranks_task
            if i == 0 and method == 'grid_search':
                print([a - b for a, b in zip(ranks_temp, ranks_task)])
            ranks.append(np.mean(ranks_task))
            ranks_default.append(np.mean(ranks_default_task))
            
        # Fill table columns
        for i, metric in enumerate(METRIC):
            if tracker:
                table[f"{metric} (norm)"].append(norm_scores_default[i])
                table[f"{metric} (rel)"].append(rel_diffs_default[i])
                table[f"{metric} (rank)"].append(ranks_default[i])
            table[f"{metric} (norm)"].append(norm_scores[i])
            table[f"{metric} (rel)"].append(rel_diffs[i])
            table[f"{metric} (rank)"].append(ranks[i])
        tracker = False
    # Build DataFrame with metrics/statistics as rows and methods as columns
    # Add 'Default' to sorted_hpo_method_names in alphabetical order
    idx = sorted(sorted_hpo_method_names + ['Default'])

    df = pd.DataFrame(table, index=idx)
    latex_string = style_hyperparameter_results(df)
    with open("hyperparameter_results.tex", "w") as f:
        f.write(latex_string)

