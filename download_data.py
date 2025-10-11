"""download_data

Utility to download OpenML benchmark suites and save task indices and names.

Author: Ioana Iacobici (https://github.com/iiacobici)
Modified by: Floris Koster (https://github.com/fl0risk)

Usage:
    Run this module to fetch the OpenML suites listed in `suites` and store
    task index and name files under the `task_indices/` directory.
"""
import json
import os
import numpy as np
import openml

suites = [334, 335, 336, 337]

def main():
    for suite_id in suites:
        benchmark_suite = openml.study.get_suite(suite_id)
        tasks = benchmark_suite.tasks

        os.makedirs("task_indices", exist_ok=True)
        np.save(f"task_indices/{suite_id}_task_indices.npy", tasks)

        names = []

        for task_id in benchmark_suite.tasks:
            task = openml.tasks.get_task(task_id)   # download the OpenML task
            dataset = task.get_dataset()    #get dataset from task
            name = dataset.name     #get name from dateset
            names.append(name)  #append names               

        with open(f"task_indices/{suite_id}_task_names.json", "w") as f:
            json.dump(names, f)
    
    #for financial task
    # tasks = [361055, 361066, 317599]

    # os.makedirs("task_indices", exist_ok=True)
    # np.save(f"task_indices/financial_task_indices.npy", tasks)

    # names = []

    # for task_id in tasks:
    #     task = openml.tasks.get_task(task_id)   # download the OpenML task
    #     dataset = task.get_dataset()    #get dataset from task
    #     name = dataset.name     #get name from dateset
    #     names.append(name)  #append names               

    # with open(f"task_indices/financial_task_names.json", "w") as f:
    #     json.dump(names, f)



if __name__ == '__main__':
    main()