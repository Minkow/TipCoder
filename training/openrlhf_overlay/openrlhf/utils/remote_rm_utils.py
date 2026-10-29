import os
import time
import ray
import requests
import torch

from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.rm_serialization import decode_from_transport

logger = init_logger(__name__)
DEBUG_REMOTE_RM = os.environ.get("REMOTE_RM_DEBUG", "0") == "1"


def _unwrap_remote_payload(payload):
    """Handle optional job wrappers from newer RM servers."""
    if isinstance(payload, dict) and "result" in payload:
        result = payload["result"]
        if isinstance(result, dict) and "data" in result:
            return result["data"]
        return result
    return payload


def request_api_wrapper(url, data, try_max_times=5):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=180)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            payload = response.json()
            payload = _unwrap_remote_payload(payload)
            payload = decode_from_transport(payload)
            if DEBUG_REMOTE_RM:
                if isinstance(payload, dict):
                    rewards_preview = payload.get("rewards")
                    if isinstance(rewards_preview, (list, tuple)):
                        preview = rewards_preview[: min(len(rewards_preview), 3)]
                        preview_type = (
                            type(rewards_preview[0]).__name__
                            if rewards_preview
                            else "empty"
                        )
                        preview_msg = f"len={len(rewards_preview)}, sample_type={preview_type}, sample={preview}"
                    else:
                        preview_msg = f"type={type(rewards_preview).__name__}"
                else:
                    preview_msg = f"non-dict payload type={type(payload).__name__}"
                logger.info(
                    "[RemoteRM][response] keys=%s | rewards=%s",
                    list(payload.keys()) if isinstance(payload, dict) else "-",
                    preview_msg,
                )
            return payload
        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(
        f"Request error for {try_max_times} times, returning None. Please check the API server."
    )


@ray.remote
def remote_rm_fn_ray(api_url, queries, prompts, labels):
    return request_api_wrapper(
        api_url, {"query": queries, "prompts": prompts, "labels": labels}
    )


@ray.remote
class RemoteRewardModel:
    def __init__(self, args, remote_rm_url):
        self.args = args
        self.remote_rm_url = (
            [remote_rm_url] if isinstance(remote_rm_url, str) else remote_rm_url
        )
        self.custom_reward_func = None

        if self.remote_rm_url and self.remote_rm_url[0].endswith(".py"):
            print(
                f"Loading custom `reward_func(queries, prompts, labels)` from {self.remote_rm_url[0]}"
            )
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "reward_func", self.remote_rm_url[0]
            )
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = ray.remote(reward_module.reward_func)

    @staticmethod
    def _ensure_list(value):
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @staticmethod
    def _merge_result_chunks(chunks):
        if not isinstance(chunks, list):
            return chunks

        if DEBUG_REMOTE_RM:
            logger.info(
                "[RemoteRM][merge] received %d chunks; types=%s",
                len(chunks),
                [type(chunk).__name__ for chunk in chunks],
            )

        merged = {
            "rewards": [],
        }

        have_scores = all(
            isinstance(chunk, dict) and "scores" in chunk for chunk in chunks if chunk
        )
        if have_scores:
            merged["scores"] = []

        extra_logs_acc = None

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue

            rewards = RemoteRewardModel._ensure_list(chunk.get("rewards"))
            merged["rewards"].extend(rewards)

            if have_scores:
                merged["scores"].extend(
                    RemoteRewardModel._ensure_list(chunk.get("scores"))
                )

            if "extra_logs" in chunk and isinstance(chunk["extra_logs"], dict):
                if extra_logs_acc is None:
                    extra_logs_acc = {key: [] for key in chunk["extra_logs"].keys()}
                for key, value in chunk["extra_logs"].items():
                    extra_logs_acc.setdefault(key, []).append(value)

        if extra_logs_acc:
            merged["extra_logs"] = {}
            for key, tensors in extra_logs_acc.items():
                try:
                    cat_ready = [
                        t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
                        for t in tensors
                    ]
                    merged["extra_logs"][key] = torch.cat(cat_ready, dim=0)
                except Exception:
                    merged["extra_logs"][key] = tensors

        if DEBUG_REMOTE_RM:
            logger.info(
                "[RemoteRM][merge] merged rewards=%d, scores=%d, extra_logs=%s",
                len(merged.get("rewards", [])),
                len(merged.get("scores", [])) if "scores" in merged else 0,
                list(merged.get("extra_logs", {}).keys())
                if "extra_logs" in merged
                else [],
            )

        return merged

    def get_rewards(self, queries_list, prompts_list, labels_list):
        if self.custom_reward_func:
            # Let Ray automatically distribute the workload across available resources
            batch_size = self.args.micro_rollout_batch_size
            num_chunks = (len(queries_list) + batch_size - 1) // batch_size
            r_refs = []
            for i in range(num_chunks):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(queries_list))
                r = self.custom_reward_func.remote(
                    queries_list[start_idx:end_idx],
                    prompts_list[start_idx:end_idx],
                    labels_list[start_idx:end_idx],
                )
                r_refs.append(r)
        else:
            # Distribute data across different remote reward function servers
            num_servers = len(self.remote_rm_url)
            batch_size = (len(queries_list) + num_servers - 1) // num_servers
            r_refs = []
            for i in range(num_servers):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(queries_list))
                rm = self.remote_rm_url[i]
                r = remote_rm_fn_ray.remote(
                    rm,
                    queries=queries_list[start_idx:end_idx],
                    prompts=prompts_list[start_idx:end_idx],
                    labels=labels_list[start_idx:end_idx],
                )
                r_refs.append(r)

        results = ray.get(r_refs)
        merged = self._merge_result_chunks(results)
        return merged
