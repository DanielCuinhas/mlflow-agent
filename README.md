# MLflow Experiment Analysis Agent

A local example data product for recording tabular model experiments in MLflow and asking questions about those runs through a conversational agent. Training and tracking are handled by MLflow; the agent uses a selectable Anthropic, OpenAI, or Hugging Face model through LangChain/LangGraph and accesses tracking data through a read-only MCP server.

## What You Can Do

- Run a reproducible baseline across five scikit-learn classifiers using the bundled breast cancer dataset.
- Review cross-validation means and standard deviations, train-validation gaps, parameters, tags, and run artifacts in MLflow.
- Ask the agent to summarize, filter, and compare runs, then inspect selected JSON diagnostics.
- Open the MLflow UI to browse runs and logged models.

The included dataset and training script are a demonstration, not a production training pipeline. The code uses a fixed seed and five-fold stratified cross-validation. It reserves 20% of the dataset in `train_test_split`, but the training script does not evaluate or log that final test split. Do not interpret the logged cross-validation metrics as final test performance.

## Requirements

- Python 3.10 or later
- An API key for the selected model provider (Anthropic, OpenAI, or Hugging Face)
- Internet access when calling the selected provider

## Install

From the repository directory, create and activate a virtual environment, then install the dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env   # add keys
```


For OpenAI, set `MODEL_PROVIDER=openai` and `OPENAI_API_KEY`. For Hugging Face, set `MODEL_PROVIDER=huggingface` and `HUGGINGFACE_API_KEY`; it uses the Hugging Face Inference Providers router by default. The agent loads `.env` automatically. Optional settings:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MODEL_PROVIDER` | Chat model provider | `anthropic` |
| `MODEL_NAME` / `ANTHROPIC_MODEL` | Anthropic model name (`ANTHROPIC_MODEL` remains supported) | `claude-sonnet-5-5` |
| `OPENAI_MODEL_NAME` | OpenAI model name | `gpt-4o-mini` |
| `HUGGINGFACE_MODEL_NAME` | Hugging Face model and provider | `Qwen/Qwen2.5-72B-Instruct:deepinfra` |
| `MODEL_BASE_URL` | Optional OpenAI-compatible endpoint override; for Hugging Face, defaults to its Inference Providers router | Provider default |
| `MLFLOW_TRACKING_URI` | MLflow tracking store used by the agent and MCP server | Absolute path to `mlflow.db` in the current directory |
| `LANGSMITH_TRACING` and `LANGSMITH_API_KEY` | Optional LangSmith tracing | Disabled |

Run commands from the repository directory so training and the agent use the same default `mlflow.db`.

## Quick Start

1. Train the example models and log five runs:

   ```sh
   python train_tabular.py
   ```

   Each run logs model parameters and tags, cross-validation metrics, diagnostics JSON artifacts, and a fitted model. Re-running the script creates additional runs in the same `tabular-classification-baseline` experiment; it does not replace earlier runs.

2. Start the MLflow UI in a separate terminal:

   ```sh
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```

   Open the local URL printed by MLflow (normally `http://127.0.0.1:5000`). Stop the UI with Ctrl+C.

3. Start a chat session in another terminal, with the virtual environment activated:

   ```sh
   python agent.py
   ```

   Ask questions and enter `exit` or `quit` to finish. For a single question, run:

   ```sh
   python agent.py "Summarize the runs and compare their cross-validation F1 scores"
   ```

   Choose a provider and model for a run with `--provider` and `--model`:

   ```sh
   python agent.py --provider openai --model gpt-4o "Summarize the runs"
   python agent.py --provider huggingface --model "Qwen/Qwen2.5-72B-Instruct:deepinfra"
   ```

   `--provider` accepts `anthropic`, `openai`, or `huggingface`. Both options are optional; without them, the agent uses the configured provider and model from the environment or `.env` file. API keys and endpoint settings still come from the environment or `.env` file.

The agent also works with an existing MLflow store: set `MLFLOW_TRACKING_URI` before starting it. The example trainer currently writes to `sqlite:///mlflow.db` in the current directory and does not read that environment variable; configure the trainer separately if you change stores.

## Working With Your Own Experiments

The agent discovers experiments and runs from MLflow, so it can analyze runs created by other training code as long as they are logged to the selected tracking store. For useful comparisons, log consistent parameter names, metric names, and tags across runs. This example's agent instructions expect metrics such as `cv_f1_mean`, `cv_f1_std`, and `gap_f1`; adapt those instructions or your logging schema when using a different task or metric.

To provide task-specific guidance to the agent, add a `problem_context.md` file alongside `agent.py`. It is optional; when absent, the agent starts without additional problem context. Keep context factual and include the objective, constraints, and candidate search space. Do not include secrets or private data that should not be sent to the configured model provider.

## What the Agent Can Read

The MCP server exposes five read-only tools:

| Tool | Use |
| --- | --- |
| `list_experiments` | List experiment names, IDs, and run counts. |
| `search_runs` | Filter and rank runs; returns parameters, user tags, and selected `cv_*`, `gap_*`, and `fit_*` metrics. Results are capped at 50. |
| `get_run_summary` | Retrieve one run's parameters, all metrics, user tags, and artifact paths. |
| `compare_runs` | Compare 2 to 10 runs, showing differing parameters and metrics. |
| `get_artifact_json` | Read a JSON artifact for a run; only `.json` files are accepted and output is capped/truncated at 6,000 characters. |

The server does not provide tools to start runs, change experiment data, or delete tracking records. Its read-only design is an application-level restriction, not a substitute for protecting the tracking store and its credentials.

## Interpreting Results

- `cv_<metric>_mean` is the mean validation score across folds; `cv_<metric>_std` describes fold-to-fold variation.
- `gap_<metric>` is training mean minus validation mean. A larger positive gap can indicate overfitting; it is a diagnostic, not proof by itself.
- `train_<metric>_mean` is included in full run summaries and comparisons, but omitted from compact search results.
- Small score differences relative to the fold variation should not be treated as a reliable ranking. The agent is instructed to identify such differences as inconclusive.
- `diagnostics/classification_report.json` and `diagnostics/confusion_matrix.json` are generated from out-of-fold predictions. `diagnostics/feature_importances.json` is logged when the fitted estimator provides coefficients or feature importances.

The agent's responses are model-generated interpretations of the values it retrieves. Verify important conclusions in MLflow and validate proposed experiments independently.

## Next Steps

- Test the agent on messier data, including inconsistent metric names, missing values, and runs with incomplete logging.
- Close the loop manually first: review the agent's findings, choose and run the next experiment yourself, then compare the new results.
- **Update `problem_context.md` after each round** to keep the objective, constraints, and candidate search space current.
- **Add the `get_metric_history` tool** so the agent can inspect how a metric changes over the course of a run.

## Troubleshooting

- **Agent reports a missing API key:** set `ANTHROPIC_API_KEY` in the environment or in the repository's `.env` file, then restart the agent.
- **No experiments or runs appear:** check that commands are running from the repository directory or that `MLFLOW_TRACKING_URI` points to the store containing the runs.
- **Experiment not found:** the example trainer creates `tabular-classification-baseline`; check the exact experiment name and tracking store.
- **Agent has no task-specific guidance:** create `problem_context.md` next to `agent.py`; it is optional and is not included in this repository by default.