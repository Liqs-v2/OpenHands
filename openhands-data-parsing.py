# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: openhands-ai-PaNfh9YJ-py3.12
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Key requirements
#
# Follows data structure
# {
#     "experiment": {
#         'instance_cost': <instance_cost>,
#         'solve_rate': <solve_rate>,
#         'instance_summary_cost' <instance_summary_cost>
#     }
# }

# %%
import os
import json
import pandas as pd

base_dir = 'evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent'
experiment_dirs = [
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-llm_summary_N_21_M_10-summarizer_for_eval-run_1',
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-raw_agent-run_1',
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-observation_masking-observation_masking_for_eval-run_1'
]

# Extract data for each experiment according to the specified requirements
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def _aggregate_instance(experiment_path, instance_id, row_map, lock):
    """Aggregate costs for a single instance within an experiment.

    This function reads all completion files for the given instance and updates
    the shared row_map with the computed cost and summary_cost in a thread-safe
    manner.
    """
    llm_instance_dir = os.path.join(experiment_path, 'llm_completions', instance_id)
    total_cost = 0.0
    summary_cost = 0.0

    if os.path.isdir(llm_instance_dir):
        for subdir, _, files in os.walk(llm_instance_dir):
            for file in files:
                if not file.endswith('.json'):
                    continue
                json_path = os.path.join(subdir, file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    cost = data.get('cost', 0.0)
                    if cost is None:
                        cost = 0.0
                    filename_lower = file.lower()
                    if 'summary' in filename_lower:
                        summary_cost += cost
                    total_cost += cost
                except (OSError, json.JSONDecodeError) as e:
                    print(f"Error processing {json_path}: {e}")

    with lock:
        row = row_map[instance_id]
        row['cost'] = round(total_cost, 4)
        row['summary_cost'] = round(summary_cost, 4)


def process_experiment(experiment_name, results_rows):
    """Process a single experiment directory by aggregating per-instance data.

    Initializes rows for each instance using report.json (single read), then
    concurrently aggregates instance costs from llm_completions.
    """
    experiment_path = os.path.join(base_dir, experiment_name)

    # Load outcomes once
    try:
        report_path = os.path.join(experiment_path, 'report.json')
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        resolved_ids = report.get('resolved_ids', [])
        unresolved_ids = report.get('unresolved_ids', [])
        total_instances = report.get('total_instances')

        if isinstance(total_instances, int) and (len(resolved_ids) + len(unresolved_ids) != total_instances):
            print(
                f"Sanity check failed for {experiment_name}: "
                f"resolved({len(resolved_ids)}) + unresolved({len(unresolved_ids)}) != total_instances({total_instances})"
            )
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error processing report.json for {experiment_name}: {e}")
        return

    # Instance list (preserve order, deduplicate if needed)
    instance_ids = set(dict.fromkeys(resolved_ids + unresolved_ids))

    # Prepare rows and a map for quick updates
    row_map = {}
    for instance_id in instance_ids:
        outcome = 1 if instance_id in resolved_ids else 0
        row = {
            'experiment_name': experiment_name,
            'instance_id': instance_id,
            'cost': 0.0,
            'summary_cost': 0.0,
            'outcome': outcome,
        }
        results_rows.append(row)
        row_map[instance_id] = row

    # Concurrency across instances; thread-safe updates to row_map
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(_aggregate_instance, experiment_path, instance_id, row_map, lock)
            for instance_id in instance_ids
        ]
        for _ in as_completed(futures):
            pass

# Process all experiments (aggregate per-instance rows)
all_rows = []
for experiment_dir in experiment_dirs:
    print(f"\nProcessing experiment: {experiment_dir}")
    process_experiment(experiment_dir, all_rows)

# Build DataFrame and write to CSV
df = pd.DataFrame(all_rows, columns=['experiment_name', 'instance_id', 'cost', 'summary_cost', 'outcome'])
csv_filename = 'experiment_instance_costs.csv'
df.to_csv(csv_filename, index=False)
print(f"\nWrote {len(df)} rows to {csv_filename}")

# %% [markdown]
# # Disable reasoning robustness analysis

# %%
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading

base_dir = 'evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent'
llm_completions_dir = os.path.join(
    base_dir,
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-observation_masking-observation_masking_for_eval-run_1',
    'llm_completions'
)

# Thread-safe aggregation using a dict and a lock
agg = defaultdict(int)
agg_lock = threading.Lock()

def check_completion_tokens_details(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        # Traverse to the field, handle missing keys gracefully
        details = None
        try:
            details = data['response']['usage']['completion_tokens_details']
        except (KeyError, TypeError):
            details = None
        with agg_lock:
            if details is not None:
                agg['not_null'] += 1
            else:
                agg['null'] += 1
    except Exception as e:
        print(f"Error processing {json_path}: {e}")

# Gather all JSON files in all subdirectories of llm_completions
json_files = []
for subdir, dirs, files in os.walk(llm_completions_dir):
    for file in files:
        if file.endswith('.json'):
            json_files.append(os.path.join(subdir, file))

print(f"Found {len(json_files)} JSON files")

with ThreadPoolExecutor(max_workers=40) as executor:
    futures = [executor.submit(check_completion_tokens_details, path) for path in json_files]
    for future in as_completed(futures):
        pass  # All aggregation is handled in the function

print(dict(agg))

# %%
37774 / (37774+1496)

# %%
87079 / (87079+1739)
