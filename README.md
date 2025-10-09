# Overview
This repository contains our fork of Openhands with which we ran the generalization probing experiments reported our the work. This file summarizes the changes we made in our fork and how to use this project. For the original README see `README-OpenHands.md`.

## Contributions
- Added a docker disk space manager script to cope with the intense disk space requirements of OpenHands resulting from docker images and containers (`docker-disk-space-manager.py`) that is configurable though `docker_disk_space_manager.json`. We recommend running this in the background through `screen` when evaluating on the entire benchmark.
- Updated the logging logic in to log LLM summaries directly with the agent model completions for easier analysis (see commit 0491ac773174faaa9574dfda3cc1baf16d01daa6).
- Made the retry mechanism in case of hanging docker containers more robust (see commit 752310ac5760932e75eca1e8c4ea7f984bddbe9a).
- Made highly experimental changes to the LLM-Summary implementation to allow for splitting the turns to summarize (N) and tail turns to retain (M) into arbitrary sizes. This allowed us to evaluate this scaffold with the same LLM-Summary hyperparameters as we used in SWE-agent.

## Setup
To set up this project, simply follow the official OpenHands instructions. We used the setup for [developing without root access](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

For convenience, we present the minimal setup steps for reproducibility here:
1. Install `conda`

2. Load `conda` into the terminal session
```bash
eval "$(/path/to/conda/miniforge3/bin/conda shell.<YOUR_TERMINAL> hook)"
```

3. Create a conda environment with Python 3.12 and install high-level project dependencies with `conda`
```bash
conda create -n openhands python=3.12
conda activate openhands
conda install conda-forge::nodejs
conda install conda-forge::poetry
```

4. Set up the project dependencies:
```bash
make build
```
5. Install `jupytext` to deserialize our notebook to a Jupyter notebook:
`poetry run pip install jupytext`

6. Install auxiliary dependency groups needed for dev and eval:
`poetry install --with dev,evaluation`

7. Make sure your system has `docker` installed and that the daemon is running.

## Running the agent on SWE-bench
To run the agent, simply follow the official OpenHands documentation [here](https://github.com/All-Hands-AI/OpenHands/blob/main/evaluation/benchmarks/swe_bench/README.md).

### Configuration
We provide our configuration in `config.toml`. All you need to do is configure your API key/model to use.

Note that a single turn consists of 2 messages in Openhands. These 2 messages refer to input (reasoning and action) and output (environment observation). Thus, in the configuration all window sizes etc reported in our work are multiplied by 2.

To run SWE-bench Verified-50, the subset on which we report our hyperparameter tuning ablation in our work, we use `evaluation/benchmarks/swe_bench/config_v50.toml`. To run on this subset, rename the file to `config.toml`.

## Unpacking and using our notebook
For easier versioning, we serialize our Jupyter notebook to Python with `jupytext`. To deserialize it use:
```bash
jupytext --set-formats ipynb,py openhands-data-parsing.py
```

## 🪪 License <a name="license"></a>
MIT. Check `LICENSE`.
