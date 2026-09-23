"""Inspect a model response cache without initializing an API client.

Usage:
    python scripts/check_cache.py --model gpt-6-astra
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_responses.cache_utils import is_valid_cached_response


DEFAULT_CONFIG = REPO_ROOT / "configs" / "default_config.yaml"

# These aliases mirror initialize_model() in scripts/responses_runner.py. Models
# whose deployment/API name is also their config name do not need an entry.
API_MODEL_NAMES = {
    "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
    "gpt-4o": "gpt-4o-2024-05-13",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-5": "gpt-5-2025-08-07",
    "gpt-5-mini": "gpt-5-mini-2025-08-07",
    "o3": "o3-2025-04-16",
    "o3-mini": "o3-mini-2025-01-31",
    "gpt-5.1": "gpt-5.1-2025-11-13",
    "qwen-3.8-max": "qwen/qwen3.8-max-0902",
    "kimi-k3": "moonshotai/kimi-k3",
    "glm-5.3": "z-ai/glm-5.3",
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro-0813",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
    "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
    "claude-4.0-sonnet": "claude-sonnet-4-20250514",
    "claude-4.1-opus": "claude-opus-4-1-20250805",
    "claude-4.5-opus": "claude-opus-4-5-20251101",
    "claude-opus-5.5": "claude-opus-5-5",
    "claude-fable-5.1": "claude-fable-5-1",
    "perplexity-sonar-huge": "llama-3.1-sonar-huge-128k-online",
    "gemma-2-27b-it": "google/gemma-2-27b-it",
    "llama-3.1-70b-it": "meta-llama/Meta-Llama-3.1-70B-Instruct",
}

CLAUDE_EFFORT_MODELS = {
    "claude-opus-5",
    "claude-opus-5.5",
    "claude-fable-5.1",
}


@dataclass(frozen=True)
class CacheIdentity:
    provider: str
    api_model_name: str
    cache_filename: str
    cache_key_prefix: str
    effort: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the response cache and response CSV for one model."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name as it appears in the benchmark config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Config file (default: {DEFAULT_CONFIG.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=10,
        help="Maximum number of failed/missing row identifiers to display.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    try:
        with path.expanduser().resolve().open() as config_file:
            config = yaml.safe_load(config_file)
    except (OSError, yaml.YAMLError) as error:
        raise SystemExit(f"Could not load config '{path}': {error}") from error

    if not isinstance(config, dict):
        raise SystemExit(f"Config '{path}' is not a YAML object.")
    return config


def find_model_provider(config: dict, model_name: str) -> str:
    matches = [
        model
        for model in config.get("models", [])
        if model.get("name") == model_name
    ]
    if not matches:
        available = ", ".join(
            model.get("name", "") for model in config.get("models", [])
        )
        raise SystemExit(
            f"Model '{model_name}' was not found in the config. "
            f"Available models: {available}"
        )
    return matches[0].get("type", "")


def get_cache_identity(model_name: str, provider: str) -> CacheIdentity:
    api_model_name = API_MODEL_NAMES.get(model_name, model_name)

    if provider == "azure_openai":
        return CacheIdentity(
            provider=provider,
            api_model_name=api_model_name,
            cache_filename=f"azure_{api_model_name}_cache.json",
            cache_key_prefix=f"azure_{api_model_name}",
            effort="low",
        )

    if provider == "openrouter":
        cache_name = api_model_name.replace("/", "__")
        return CacheIdentity(
            provider=provider,
            api_model_name=api_model_name,
            cache_filename=f"openrouter_{cache_name}_cache.json",
            cache_key_prefix=f"openrouter_{api_model_name}",
            effort="low",
        )

    cache_name = (
        api_model_name.rsplit("/", 1)[-1]
        if provider == "huggingface"
        else api_model_name
    )
    effort = "low" if model_name in CLAUDE_EFFORT_MODELS else None
    return CacheIdentity(
        provider=provider,
        api_model_name=api_model_name,
        cache_filename=f"{cache_name}_cache.json",
        cache_key_prefix=api_model_name,
        effort=effort,
    )


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def get_dataset_path(config: dict) -> Path:
    data_dir = resolve_repo_path(
        config.get("paths", {}).get("dataset_directory", "./data/")
    )
    split = config.get("dataset", {}).get("split", "test")
    return data_dir / f"CARDBiomedBench_{split}.csv"


def make_cache_key(
    identity: CacheIdentity,
    system_prompt: str,
    question: str,
) -> str:
    effort_suffix = (
        f"_effort={identity.effort}" if identity.effort is not None else ""
    )
    return (
        f"{identity.cache_key_prefix}{effort_suffix}_"
        f"{system_prompt}_{question}"
    )


def row_identifiers(data: pd.DataFrame, indices: list[int]) -> list:
    identifier_column = next(
        (
            column
            for column in ("uuid", "question_uuid", "id", "template_uuid")
            if column in data.columns
        ),
        None,
    )
    if identifier_column is None:
        return indices
    return data.iloc[indices][identifier_column].tolist()


def inspect_cache(
    cache_path: Path,
    data: pd.DataFrame,
    identity: CacheIdentity,
    system_prompt: str,
) -> tuple[dict, list[int], list[int], int]:
    try:
        with cache_path.open() as cache_file:
            cache = json.load(cache_file)
    except OSError as error:
        raise SystemExit(f"Could not read cache '{cache_path}': {error}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"Cache is not valid JSON: {cache_path}: {error}") from error

    if not isinstance(cache, dict):
        raise SystemExit(f"Cache is not a JSON object: {cache_path}")

    missing_indices = []
    invalid_indices = []
    valid_rows = 0

    for row_index, question in enumerate(data["question"]):
        cache_key = make_cache_key(identity, system_prompt, str(question))
        if cache_key not in cache:
            missing_indices.append(row_index)
        elif is_valid_cached_response(cache[cache_key]):
            valid_rows += 1
        else:
            invalid_indices.append(row_index)

    return cache, missing_indices, invalid_indices, valid_rows


def inspect_response_csv(
    response_path: Path,
    model_name: str,
) -> tuple[int, int, list] | None:
    if not response_path.exists():
        return None

    responses = pd.read_csv(response_path)
    response_column = f"{model_name}_response"
    if response_column not in responses.columns:
        raise SystemExit(
            f"Response CSV does not contain '{response_column}': {response_path}"
        )

    failed_mask = responses[response_column].apply(
        lambda value: pd.isna(value) or not is_valid_cached_response(value)
    )
    failed_indices = [
        index for index, failed in enumerate(failed_mask.tolist()) if failed
    ]
    return len(responses), len(failed_indices), row_identifiers(responses, failed_indices)


def main() -> None:
    args = parse_args()
    if args.show < 0:
        raise SystemExit("--show must be zero or greater.")

    config = load_config(args.config)
    provider = find_model_provider(config, args.model)
    identity = get_cache_identity(args.model, provider)
    system_prompt = config.get("prompts", {}).get("system_prompt", "").rstrip()

    dataset_path = get_dataset_path(config)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    data = pd.read_csv(dataset_path)
    if "question" not in data.columns:
        raise SystemExit(f"Dataset does not contain a 'question' column: {dataset_path}")

    cache_dir = resolve_repo_path(
        config.get("paths", {}).get("cache_directory", "./.cache/")
    ) / "model_responses_cache"
    cache_path = cache_dir / identity.cache_filename
    if not cache_path.exists():
        raise SystemExit(f"Cache not found: {cache_path}")

    cache, missing, invalid, valid_rows = inspect_cache(
        cache_path,
        data,
        identity,
        system_prompt,
    )
    invalid_entry_count = sum(
        not is_valid_cached_response(value) for value in cache.values()
    )
    query_count = len(missing) + len(invalid)

    print("MODEL")
    print(f"  Config name: {args.model}")
    print(f"  Provider: {identity.provider}")
    print(f"  API/deployment name: {identity.api_model_name}")
    print(f"  Cache-key effort: {identity.effort or 'not encoded'}")
    print()
    print("CACHE")
    print(f"  Path: {cache_path}")
    print(f"  Size: {cache_path.stat().st_size / (1024 * 1024):.1f} MB")
    print(f"  Total entries: {len(cache)}")
    print(f"  Invalid/empty entries: {invalid_entry_count}")
    print(f"  Dataset rows: {len(data)}")
    print(f"  Valid cache hits for current run: {valid_rows}")
    print(f"  Invalid cache hits for current run: {len(invalid)}")
    print(f"  Missing cache keys for current run: {len(missing)}")
    print(f"  Rows that will be queried: {query_count}")
    if args.show and invalid:
        print(
            "  First invalid row identifiers: "
            f"{row_identifiers(data, invalid)[:args.show]}"
        )
    if args.show and missing:
        print(
            "  First missing row identifiers: "
            f"{row_identifiers(data, missing)[:args.show]}"
        )

    output_dir = resolve_repo_path(
        config.get("paths", {}).get("output_directory", "./results/")
    )
    response_path = output_dir / "by_model" / f"{args.model}_responses.csv"
    response_summary = inspect_response_csv(response_path, args.model)
    print()
    print("RESPONSE CSV")
    print(f"  Path: {response_path}")
    if response_summary is None:
        print("  Status: not found")
        return

    total_rows, failed_count, failed_identifiers = response_summary
    print(f"  Total rows: {total_rows}")
    print(f"  Valid responses: {total_rows - failed_count}")
    print(f"  Failed responses: {failed_count}")
    if args.show and failed_identifiers:
        print(f"  First failed row identifiers: {failed_identifiers[:args.show]}")


if __name__ == "__main__":
    main()
