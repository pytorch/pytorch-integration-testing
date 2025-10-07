import json
import os
import glob
import lm_eval
import yaml
from logging import warning, info
from argparse import Action, ArgumentParser, Namespace
import torch
from typing import Dict, Any, List, Optional


# See lm-eval docs for the list of acceptable values
LM_EVAL_MODEL_SOURCE = os.environ.get("LM_EVAL_MODEL_SOURCE", "vllm")


class ValidateDir(Action):
    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        if os.path.isdir(values):
            setattr(namespace, self.dest, values)
            return

        parser.error(f"{values} is not a valid directory")


def parse_args() -> Any:
    parser = ArgumentParser("Run vLLM lm-eval harness")

    parser.add_argument(
        "--configs-dir",
        type=str,
        action=ValidateDir,
        help="the directory contains vLLM lm-eval harness configs",
        required=True,
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="the comma-separated list of models to evaluate (optional)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="the comma-separated list of tasks to evaluate (optional)",
    )

    return parser.parse_args()


def convert_to_pytorch_benchmark_format(
    model_name: str, tp_size: int, results: Dict[str, Any]
) -> List[Any]:
    records = []
    configs = results.get("configs", {})

    for task_name, metrics in results.get("results", {}).items():
        for metric_name, metric_value in metrics.items():
            if type(metric_value) is str:
                continue

            record = {
                "benchmark": {
                    "name": "vLLM lm-eval harness",
                    "extra_info": {
                        "args": {
                            "tensor_parallel_size": tp_size,
                        },
                        "configs": configs.get(task_name, {}),
                    },
                },
                "model": {
                    "name": model_name,
                },
                "metric": {
                    "name": metric_name,
                    "benchmark_values": [metric_value],
                },
            }
            records.append(record)

    return records


def run(
    model_name: str, tasks: List[str], tp_size: int, config: Dict[str, Any]
) -> Dict[str, Any]:
    trust_remote_code = config.get("trust_remote_code", False)
    max_model_len = config.get("max_model_len", 8192)

    model_args = (
        f"pretrained={model_name},"
        f"tensor_parallel_size={tp_size},"
        f"add_bos_token=true,"
        f"trust_remote_code={trust_remote_code},"
        f"max_model_len={max_model_len}"
    )
    info(f"Evaluating {model_name} with {model_args}")
    return lm_eval.simple_evaluate(
        model=LM_EVAL_MODEL_SOURCE,
        model_args=model_args,
        tasks=tasks,
        num_fewshot=config["num_fewshot"],
        limit=config["limit"],
        batch_size="auto",
    )


def run_lm_eval(configs_dir: str, models: List[str], tasks: List[str]) -> None:
    device_name = torch.cuda.get_device_name().lower()
    device_count = torch.cuda.device_count()

    results_dir = os.path.join(configs_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    for file in glob.glob(f"{configs_dir}/**/*.yml", recursive=True):
        with open(file) as f:
            config = yaml.safe_load(f)
        # Check the model name
        model_name = config.get("model_name", "").lower()
        if models and model_name not in models:
            info(f"Skip {model_name} from {file}")
            continue

        tp_size = 0
        selected_tasks = []

        # Check the lm-eval tasks, the selected device, and tp
        for t in config.get("tasks", []):
            task_name = t["name"]
            if not task_name:
                warning(f"{model_name} from {file}: skip missing task")
                continue

            if tasks and task_name not in tasks:
                info(f"{model_name} from {file}: {task_name} not selected")

            selected_device = t["device"].lower()
            if selected_device not in device_name:
                continue

            tp = t["tp"]
            if device_count < tp:
                warning(
                    f"{model_name} from {file}: device count {device_count} < tp {tp} in {task_name}"
                )
                continue

            selected_tasks.append(task_name)
            if not tp_size:
                tp_size = tp
            assert tp_size == tp

        if not selected_tasks:
            info(f"Skip {model_name} from {file}: no task")
            continue

        results = run(model_name, selected_tasks, tp_size, config)
        results_pytorch_format = convert_to_pytorch_benchmark_format(
            model_name, tp_size, results
        )

        results_file = os.path.splitext(os.path.basename(file))[0]
        # Dump the results from lm-eval
        with open(os.path.join(results_dir, f"{results_file}_lm_eval.json"), "w") as f:
            json.dump(results, f, indent=2)
        # Dump the results that can be uploaded to PyTorch OSS benchmark infra
        with open(os.path.join(results_dir, f"{results_file}_pytorch.json"), "w") as f:
            json.dump(results_pytorch_format, f, indent=2)


def main() -> None:
    args = parse_args()
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    tasks = [m.strip().lower() for m in args.tasks.split(",") if m.strip()]
    run_lm_eval(args.configs_dir, models, tasks)


if __name__ == "__main__":
    main()
