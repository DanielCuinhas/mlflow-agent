"""
Read-only MLflow MCP server with a small, compact tool surface.

Tools: list_experiments, search_runs, get_run_summary, compare_runs, get_artifact_json
Env:   MLFLOW_TRACKING_URI (default: sqlite:///mlflow.db)
"""
import json
import os

import mlflow
from mcp.server.fastmcp import FastMCP
from mlflow import MlflowClient

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
client = MlflowClient()
mcp = FastMCP("mlflow-readonly")

MAX_ARTIFACT_CHARS = 6000


def _r(x):
    return round(x, 4) if isinstance(x, float) else x


def _experiment_id(name: str) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is None:
        available = [e.name for e in client.search_experiments()]
        raise ValueError(f"Experiment '{name}' not found. Available: {available}")
    return exp.experiment_id


def _user_tags(run) -> dict:
    return {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}


def _list_artifacts(run_id: str, path: str | None = None) -> list[str]:
    out = []
    for a in client.list_artifacts(run_id, path):
        if a.is_dir:
            out.extend(_list_artifacts(run_id, a.path))
        else:
            out.append(a.path)
    return out


@mcp.tool()
def list_experiments() -> list[dict]:
    """List all MLflow experiments with their IDs and run counts."""
    result = []
    for e in client.search_experiments():
        n = len(client.search_runs([e.experiment_id], max_results=1000))
        result.append({"name": e.name, "experiment_id": e.experiment_id, "n_runs": n})
    return result


@mcp.tool()
def search_runs(
    experiment_name: str,
    filter_string: str = "",
    order_by: str = "metrics.cv_f1_mean DESC",
    max_results: int = 10,
) -> list[dict]:
    """Search runs in an experiment and return a compact summary per run
    (tags, params, and the cv_*/gap_*/fit_* metrics; train_* metrics omitted).

    filter_string uses MLflow syntax, e.g.
      "metrics.cv_f1_mean > 0.97 and tags.model_family = 'gbm'"
    Note: params are stored as strings, e.g. "params.max_depth = '6'".
    order_by examples: "metrics.cv_f1_mean DESC", "metrics.gap_f1 ASC".
    max_results is capped at 50.
    """
    runs = client.search_runs(
        [_experiment_id(experiment_name)],
        filter_string=filter_string,
        order_by=[order_by],
        max_results=min(max_results, 50),
    )
    return [
        {
            "run_id": r.info.run_id,
            "run_name": r.info.run_name,
            "status": r.info.status,
            "tags": _user_tags(r),
            "params": r.data.params,
            "metrics": {
                k: _r(v)
                for k, v in r.data.metrics.items()
                if k.startswith(("cv_", "gap_", "fit_"))
            },
        }
        for r in runs
    ]


@mcp.tool()
def get_run_summary(run_id: str) -> dict:
    """Full details for one run: all params, all metrics (including train_*),
    user tags, and the list of logged artifact paths."""
    run = client.get_run(run_id)
    return {
        "run_id": run_id,
        "run_name": run.info.run_name,
        "status": run.info.status,
        "tags": _user_tags(run),
        "params": run.data.params,
        "metrics": {k: _r(v) for k, v in run.data.metrics.items()},
        "artifacts": _list_artifacts(run_id),
    }


@mcp.tool()
def compare_runs(run_ids: list[str]) -> dict:
    """Compare 2-10 runs side by side. Returns only the params that DIFFER
    between runs, plus all metrics, keyed by run name."""
    if not 2 <= len(run_ids) <= 10:
        raise ValueError("Provide between 2 and 10 run_ids.")
    runs = [client.get_run(i) for i in run_ids]
    names = [r.info.run_name or r.info.run_id[:8] for r in runs]

    param_keys = sorted({k for r in runs for k in r.data.params})
    differing = {}
    for k in param_keys:
        vals = [r.data.params.get(k) for r in runs]
        if len(set(vals)) > 1:
            differing[k] = dict(zip(names, vals))

    metric_keys = sorted({k for r in runs for k in r.data.metrics})
    metrics = {
        k: {n: _r(r.data.metrics.get(k)) for n, r in zip(names, runs)}
        for k in metric_keys
    }
    return {
        "runs": [{"run_id": r.info.run_id, "run_name": n} for r, n in zip(runs, names)],
        "differing_params": differing,
        "metrics": metrics,
    }


@mcp.tool()
def get_artifact_json(run_id: str, artifact_path: str) -> str:
    """Read a JSON artifact from a run, e.g. 'diagnostics/confusion_matrix.json',
    'diagnostics/classification_report.json', 'diagnostics/feature_importances.json'.
    Large outputs are truncated."""
    if not artifact_path.endswith(".json"):
        raise ValueError("Only .json artifacts can be read.")
    data = mlflow.artifacts.load_dict(f"runs:/{run_id}/{artifact_path}")
    text = json.dumps(data)
    if len(text) > MAX_ARTIFACT_CHARS:
        return text[:MAX_ARTIFACT_CHARS] + f"... [truncated, {len(text)} chars total]"
    return text


if __name__ == "__main__":
    mcp.run()  # stdio transport