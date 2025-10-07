import os
import glob
import lm_eval
import yaml
from logging import warning, info
from argparse import Action, ArgumentParser, Namespace
import torch
from typing import Dict, Any, List, Optional


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


def run(
    model_name: str, tasks: List[str], tp_size: int, config: Dict[str, Any]
) -> None:
    trust_remote_code = config.get("trust_remote_code", False)
    max_model_len = config.get("max_model_len", 8192)

    model_args = (
        f"pretrained={model_name},"
        f"tensor_parallel_size={tp_size},"
        f"add_bos_token=true,"
        f"trust_remote_code={trust_remote_code},"
        f"max_model_len={max_model_len}"
    )
    print(model_args)
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=config["num_fewshot"],
        limit=config["limit"],
        batch_size="auto",
    )
    print(results)


def run_lm_eval(configs_dir: str, models: List[str], tasks: List[str]) -> None:
    device_name = torch.cuda.get_device_name().lower()
    device_count = torch.cuda.device_count()

    for file in glob.glob(f"{configs_dir}/**/*.yml", recursive=True):
        config = yaml.safe_load(file)
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

            selected_tasks.push(task_name)
            if not tp_size:
                tp_size = tp
            assert tp_size == tp

        if not selected_tasks:
            info(f"Skip {model_name} from {file}: no task")
            continue

        run(model_name, selected_tasks, tp_size, config)


def main() -> None:
    args = parse_args()
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    tasks = [m.strip().lower() for m in args.runners.split(",") if m.strip()]
    run_lm_eval(args.configs_dir, models, tasks)


if __name__ == "__main__":
    main()
