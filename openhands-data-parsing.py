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
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-llm_summary_N_21_M_10-verified_50-summarizer_for_eval-run_1',
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-raw_agent-run_1',
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-raw_agent-verified_50-run_1',
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-observation_masking-observation_masking_for_eval-run_1',
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-observation_masking-M_10-verified_50-observation_masking_for_eval-run_1',
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-observation_masking-M_20-verified_50-observation_masking_for_eval-run_1',
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-observation_masking-M_29-verified_50-observation_masking_for_eval-run_1',
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-observation_masking-M_43-verified_50-observation_masking_for_eval-run_1',
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-observation_masking-M_58-verified_50-observation_masking_for_eval-run_1',
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-observation_masking-M_54-verified_50-observation_masking_for_eval-run_1',
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-observation_masking-M_50-verified_50-observation_masking_for_eval-run_1',
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-observation_masking-M_72-observation_masking_for_eval-run_1',
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-observation_masking-M_86-verified_50-observation_masking_for_eval-run_1'
]

csv_filename = 'experiment_instance_costs_openhands.csv'

# Load existing dataframe if it exists, otherwise create empty
if os.path.exists(csv_filename):
    df_existing = pd.read_csv(csv_filename)
    if 'experiment' in df_existing.columns:
        processed_experiments = set(df_existing['experiment'].unique())
    else:
        processed_experiments = set()
else:
    df_existing = pd.DataFrame()
    processed_experiments = set()

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
    has_summary = False
    summary_count = 0
    turn_count = 0
    sum_reasoning_tokens = 0
    sum_completion_tokens = 0
    sum_prompt_tokens = 0

    if os.path.isdir(llm_instance_dir):
        for subdir, _, files in os.walk(llm_instance_dir):
            for file in files:
                if not file.endswith('.json'):
                    continue
                turn_count += 1
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
                        summary_count += 1
                        has_summary = True
                    total_cost += cost

                    # Aggregate usage fields for mean calculations
                    response = data.get('response', {}) if isinstance(data, dict) else {}
                    usage = response.get('usage', {}) if isinstance(response, dict) else {}
                    completion_tokens = usage.get('completion_tokens', 0) or 0
                    prompt_tokens = usage.get('prompt_tokens', 0) or 0
                    details = usage.get('completion_tokens_details')
                    reasoning_tokens = 0
                    if isinstance(details, dict):
                        reasoning_tokens = details.get('reasoning_tokens', 0) or 0

                    sum_completion_tokens += int(completion_tokens)
                    sum_prompt_tokens += int(prompt_tokens)
                    sum_reasoning_tokens += int(reasoning_tokens)
                except (OSError, json.JSONDecodeError) as e:
                    print(f"Error processing {json_path}: {e}")

    with lock:
        row = row_map[instance_id]
        row['cost'] = round(total_cost, 4)
        row['summary_cost'] = round(summary_cost, 4)
        row['has_summary'] = has_summary
        row['summary_count'] = summary_count
        row['turn_count'] = turn_count
        if turn_count > 0:
            row['mean_reasoning_tokens'] = int(round(sum_reasoning_tokens / turn_count))
            row['mean_completion_tokens'] = int(round(sum_completion_tokens / turn_count))
            row['mean_prompt_tokens'] = int(round(sum_prompt_tokens / turn_count))
        else:
            row['mean_reasoning_tokens'] = 0
            row['mean_completion_tokens'] = 0
            row['mean_prompt_tokens'] = 0


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
        submitted_ids = report.get('submitted_ids', [])
        unresolved_ids = [instance_id for instance_id in submitted_ids if instance_id not in resolved_ids]

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
            'experiment': experiment_name,
            'instance_id': instance_id,
            'cost': 0.0,
            'summary_cost': 0.0,
            'has_summary': False,
            'summary_count': 0,
            'turn_count': 0,
            'mean_reasoning_tokens': 0,
            'mean_completion_tokens': 0,
            'mean_prompt_tokens': 0,
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
    if experiment_dir in processed_experiments:
        print(f"\nSkipping already processed experiment: {experiment_dir}")
        continue
    print(f"\nProcessing experiment: {experiment_dir}")
    process_experiment(experiment_dir, all_rows)

# If there are new rows, append to existing dataframe and write to CSV
if all_rows:
    df_new = pd.DataFrame(
        all_rows,
        columns=[
            'experiment',
            'instance_id',
            'cost',
            'summary_cost',
            'has_summary',
            'summary_count',
            'turn_count',
            'mean_reasoning_tokens',
            'mean_completion_tokens',
            'mean_prompt_tokens',
            'outcome',
        ]
    )
    if not df_existing.empty:
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new
    df_final.to_csv(csv_filename, index=False)
    print(f"\nWrote {len(df_new)} new rows to {csv_filename}")
else:
    print("\nNo new experiments processed. Existing file is up to date.")

# %%
df_final[df_final['experiment'].str.contains('summary')].describe()

# %%
df_final[df_final['experiment'] == 'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-observation_masking-M_54-verified_50-observation_masking_for_eval-run_1'].describe()

# %%
df_final[df_final['experiment'] == 'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-llm_summary_N_21_M_10-summarizer_for_eval-run_1'].describe()

# %%
df[df['experiment'] == 'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-observation_masking-M_43-verified_50-observation_masking_for_eval-run_1'].describe()

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

# %% [markdown]
# # Retry within instance occurrence analysis

# %%
import os

base_dir = 'evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent'
experiment_dirs = [
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-llm_summary_N_21_M_10-summarizer_for_eval-run_1',
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-raw_agent-run_1',
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-observation_masking-observation_masking_for_eval-run_1'
]

for experiment_dir in experiment_dirs:
    retry_count = 0
    instance_ids = []
    print('-'*3 + f'{experiment_dir}' + '-'*3)
    log_dir = os.path.join(base_dir, experiment_dir, 'infer_logs')
    for file in os.listdir(log_dir):
        text = open(os.path.join(log_dir, file), 'r').read()
        if '----------[The above error occurred. Retrying... (attempt 1 of 2)]----------' in text:
            instance_id = '_'.join(file.split('.')[0].split('_')[1:])
            instance_ids.append(instance_id)
            print(instance_id)
            retry_count += 1
    print(f"{experiment_dir}: {retry_count}")
    print(instance_ids)

# %% [markdown]
# # Re-merge error instances

# %%
import os, json
from pathlib import Path

to_fix = Path('evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-observation_masking-observation_masking_for_eval-run_1/output.jsonl')
replacements = Path('evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-masking_M_10-retries-observation_masking_for_eval-run_1/output.jsonl')

replacements_dict = {}

with open(replacements, 'r') as f:
    for line in f:
        data = json.loads(line)
        instance_id = data['instance_id']
        replacements_dict[instance_id] = data

# Write updated lines to a temporary file, then replace the original
from tempfile import NamedTemporaryFile

with NamedTemporaryFile('w', delete=False, dir=to_fix.parent, encoding='utf-8') as tmpfile:
    with open(to_fix, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            instance_id = data['instance_id']
            if instance_id in replacements_dict:
                out_data = replacements_dict[instance_id]
            else:
                out_data = data
            tmpfile.write(json.dumps(out_data) + '\n')

# Replace the original file with the updated file
os.replace(tmpfile.name, to_fix)


# %%
