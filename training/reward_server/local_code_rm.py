import os
import time
import logging
import math
import uuid
import queue
import threading
import torch
import datasets
from typing import List, Dict, Any, Optional
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm


class LocalRMActor:
    def __init__(self):
        # 在 Actor 内部初始化 logger，避免序列化问题
        self.logger = logging.getLogger("local_code_rm")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(asctime)s] [INFO] [%(name)s] %(message)s")
            )
            self.logger.addHandler(handler)

        # 详细日志开关：设置环境变量 RM_VERBOSE_LOG=1 启用详细日志
        self.verbose_log = os.environ.get("RM_VERBOSE_LOG", "0") == "1"

        self.gen_instances = []  # vLLM实例列表
        self.gen_tk = None
        self.dataset_tk = None
        self.gen_max_new_tokens = None
        self.num_gen_instances = 0
        self.initialized = False

        # 异步流水线配置
        self.pipeline_enabled = os.environ.get("RM_ASYNC_PIPELINE", "1") == "1"
        self.pipeline_queue_size = int(os.environ.get("RM_PIPELINE_QUEUE_SIZE", "4"))
        self.pipeline_queue_timeout = float(
            os.environ.get("RM_PIPELINE_QUEUE_TIMEOUT", "30")
        )
        self._request_queue: Optional[queue.Queue] = None
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._pending_futures: Dict[str, Future] = {}
        self._pending_lock = threading.Lock()

        # Test case模式相关
        self.dataset = None
        self.id_map = None  # id -> dataset_item
        self.prompt_prefix_to_id = None  # prompt前缀 -> id
        self.question_to_id = (
            None  # 原始question文本 -> id（用于与actor传入的原始ins0匹配）
        )
        self.baseline_map = None  # id -> baseline_pass_rate
        self.test_case_binary = False
        self.test_case_n_workers = 8

        # Reward计算相关配置
        self.n_samples_per_prompt = int(os.environ.get("N_SAMPLES_PER_PROMPT", "8"))
        self.use_baseline_comparison = (
            os.environ.get("RM_USE_BASELINE", "1") == "1"
        )  # 是否使用baseline比较
        self.length_penalty_weight = float(
            os.environ.get("RM_LENGTH_PENALTY", "0.005")
        )  # 长度惩罚权重
        # Actor指导模式：tips（默认）或 rewrite（重写ins0，不再拼接Tips段）
        self.guidance_mode = os.environ.get(
            "ACTOR_GUIDANCE_MODE", "tips"
        )  # tips | rewrite

        # 评分进度跟踪（按prompt计数，batch里样本属于同一问题，可按batch_size汇总）
        self.total_scores_completed = 0
        self.expected_total_scores = int(
            os.environ.get("RM_EXPECTED_TOTAL_SCORES", "0")
        )
        if self.expected_total_scores:
            self.logger.info("评分进度目标: %d 条样本", self.expected_total_scores)

        self.logger.info("LocalRMActor 已创建（Test-case模式），首次调用时加载模型")
        if self.verbose_log:
            self.logger.info("详细日志模式已启用 (RM_VERBOSE_LOG=1)")

    def _init_test_case_dataset(self):
        """初始化test case模式的数据集和hash映射"""
        dataset_name = os.environ.get("RM_DATASET")
        if not dataset_name:
            raise ValueError("Test case模式需要设置 RM_DATASET 环境变量")

        input_key = os.environ.get("RM_INPUT_KEY", "context_messages")
        gt_key = os.environ.get("RM_GT_KEY", "tests")
        baseline_model = os.environ.get("RM_BASELINE_MODEL", "qwen_coder_2.5")

        self.logger.info(f"加载数据集: {dataset_name}")
        # 支持JSONL格式（每行一个JSON对象）
        self.dataset = datasets.load_dataset(
            "json", data_files=dataset_name, split="train"
        )

        # 构建映射：使用数据集中的id字段进行匹配
        # 使用前200个字符作为key，避免格式细节问题
        self.id_map = {}  # id -> dataset_item
        self.prompt_prefix_to_id = {}  # prompt前200字符 -> id
        self.question_to_id = {}  # 原始question文本 -> id

        for item in tqdm(self.dataset, desc="构建映射"):
            item_id = item.get("id")
            conversation = item.get(input_key, [])

            if item_id:
                self.id_map[item_id] = item
                # 记录原始question文本映射
                question_text = item.get("question") or item.get("prompt") or ""
                if isinstance(question_text, str) and question_text:
                    self.question_to_id[question_text] = item_id
                # RM数据集已经是纯净instruction，不需要apply_chat_template
                # 直接使用question字段作为key
                if item.get("question"):
                    question_text = item["question"]
                    # 使用前200个字符作为key（虽然question通常很短）
                    question_prefix = question_text[:200]
                    self.prompt_prefix_to_id[question_prefix] = item_id

                # 同时也索引 input_key (context_messages) 中的内容，以防 question 字段与 context_messages 不一致
                input_data = item.get(input_key)
                if isinstance(input_data, list):
                    for msg in reversed(input_data):
                        if msg.get("role") == "user":
                            content = msg.get("content", "")
                            if content:
                                self.prompt_prefix_to_id[content[:200]] = item_id
                                self.question_to_id[content] = item_id
                                # 同时也尝试 strip 后的版本
                                self.question_to_id[content.strip()] = item_id
                            break
                elif isinstance(input_data, str) and input_data:
                    self.prompt_prefix_to_id[input_data[:200]] = item_id
                    self.question_to_id[input_data] = item_id
                    self.question_to_id[input_data.strip()] = item_id

        # 构建baseline映射：从数据集的inferences中提取指定模型的平均pass_rate
        self.baseline_map = {}  # id -> baseline_pass_rate
        if "inferences" in self.dataset.column_names:
            self.logger.info(f"计算baseline (模型: {baseline_model})...")
            for item in tqdm(self.dataset, desc="计算baseline"):
                item_id = item.get("id")
                inferences = item.get("inferences", [])

                if item_id:
                    # 筛选指定模型的inferences
                    model_inferences = [
                        inf
                        for inf in inferences
                        if inf.get("model_name") == baseline_model
                    ]

                    if model_inferences:
                        # 计算该问题在baseline模型上的平均pass_rate
                        avg_pass_rate = sum(
                            inf.get("pass_rate", 0.0) for inf in model_inferences
                        ) / len(model_inferences)
                        self.baseline_map[item_id] = avg_pass_rate
                    else:
                        # 如果没有该模型的数据，使用0作为baseline
                        self.baseline_map[item_id] = 0.0

            avg_baseline = (
                sum(self.baseline_map.values()) / len(self.baseline_map)
                if self.baseline_map
                else 0.0
            )
            self.logger.info(f"Baseline加载完成，平均pass_rate: {avg_baseline:.4f}")
        else:
            self.logger.warning("数据集中未找到'inferences'字段，无法计算baseline")

        # 读取配置
        self.test_case_binary = os.environ.get("RM_BINARY", "0") == "1"
        self.test_case_n_workers = int(os.environ.get("RM_N_WORKERS", "8"))

        self.logger.info(f"数据集加载完成，共 {len(self.id_map)} 个样本")
        self.logger.info(f"数据集路径: {dataset_name}")
        self.logger.info(f"数据集前5个ID: {list(self.id_map.keys())[:5]}")
        self.logger.info(
            f"Test case配置: binary={self.test_case_binary}, n_workers={self.test_case_n_workers}"
        )
        self.logger.info(
            f"Reward配置: use_baseline={self.use_baseline_comparison}, length_penalty={self.length_penalty_weight}, n_samples_per_prompt={self.n_samples_per_prompt}"
        )

        if not self.expected_total_scores:
            dataset_total = len(self.id_map)
            self.expected_total_scores = dataset_total
            self.logger.info(
                "评分进度目标自动设置为 %d 条样本", self.expected_total_scores
            )

    # ===== 异步流水线相关 =====

    def _init_pipeline_workers(self):
        if self._request_queue is not None:
            return
        self._request_queue = queue.Queue(maxsize=self.pipeline_queue_size)
        self._dispatcher_thread = threading.Thread(
            target=self._pipeline_worker_loop,
            name="rm_pipeline_dispatcher",
            daemon=True,
        )
        self._dispatcher_thread.start()
        self.logger.info("异步流水线已启用: queue_size=%d", self.pipeline_queue_size)

    def _pipeline_worker_loop(self):
        while not self._shutdown_event.is_set():
            try:
                request_id, payload, future = self._request_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                result = self._process_batch(
                    payload["queries"], payload["prompts"], payload["labels"]
                )
                future.set_result(result)
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.error("流水线处理失败: %s", exc, exc_info=True)
                future.set_exception(exc)
            finally:
                self._request_queue.task_done()

    def _submit_pipeline_request(
        self,
        queries: List[str],
        prompts: List[str],
        labels: List[str],
        request_id: Optional[str] = None,
    ):
        if self._request_queue is None:
            raise RuntimeError("异步流水线未初始化")

        req_id = request_id or str(uuid.uuid4())
        future = Future()
        payload = {
            "queries": list(queries),
            "prompts": list(prompts),
            "labels": list(labels),
        }

        try:
            self._request_queue.put(
                (req_id, payload, future), timeout=self.pipeline_queue_timeout
            )
        except queue.Full as exc:
            raise RuntimeError(
                f"异步流水线队列已满 (size={self.pipeline_queue_size})"
            ) from exc

        return req_id, future

    def _register_async_future(self, request_id: str, future: Future):
        with self._pending_lock:
            self._pending_futures[request_id] = future

    def fetch_async_result(self, request_id: str, timeout: Optional[float] = None):
        with self._pending_lock:
            future = self._pending_futures.get(request_id)

        if future is None:
            return {"status": "not_found", "request_id": request_id}

        try:
            result = future.result(timeout=timeout)
        except FutureTimeoutError:
            return {"status": "pending", "request_id": request_id}

        with self._pending_lock:
            self._pending_futures.pop(request_id, None)

        return {"status": "done", "request_id": request_id, "result": result}

    def _init_once(self):
        if self.initialized:
            return

        self.logger.info("开始加载本地RM（Test-case模式）...")
        gen_model_path = os.environ.get("GEN_MODEL_PATH")
        policy_model_path = os.environ.get("POLICY_MODEL_PATH")

        if not gen_model_path:
            raise ValueError("环境变量缺失: GEN_MODEL_PATH")

        self.gen_max_new_tokens = int(os.environ.get("GEN_MAX_NEW_TOKENS", "1024"))

        num_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_visible == 0:
            raise RuntimeError("LocalRMActor 需要至少一个可见GPU")
        self.logger.info(f"LocalRMActor 可见 GPU 数: {num_visible}")

        if torch.cuda.is_available():
            for i in range(num_visible):
                self.logger.info(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")

        # 使用TP而不是多实例: 将TP size设置为可见GPU数量
        self.tp_size = num_visible
        self.logger.info(
            f"启用张量并行 (TP), tp_size={self.tp_size}, 仅初始化一个vLLM实例"
        )
        self.num_gen_instances = 1

        self.gen_tk = AutoTokenizer.from_pretrained(
            gen_model_path, trust_remote_code=True
        )
        if policy_model_path:
            self.dataset_tk = AutoTokenizer.from_pretrained(
                policy_model_path, trust_remote_code=True
            )
            self.logger.info(f"使用 policy tokenizer: {policy_model_path}")
        else:
            self.dataset_tk = self.gen_tk
            self.logger.warning("未设置 POLICY_MODEL_PATH，沿用生成模型 tokenizer")

        # 初始化单个vLLM实例，开启TP
        try:
            gen_instance = LLM(
                model=gen_model_path,
                tensor_parallel_size=self.tp_size,
                dtype="bfloat16",
                trust_remote_code=True,
                max_model_len=8192,
                gpu_memory_utilization=0.9,
                enforce_eager=False,
            )
        except Exception as exc:
            self.logger.error(f"vLLM实例初始化失败(TP={self.tp_size}): {exc}")
            raise
        self.gen_instances.append(gen_instance)

        self._init_test_case_dataset()
        if self.pipeline_enabled:
            self._init_pipeline_workers()
        self.initialized = True
        self.logger.info("本地RM初始化完成（Test-case模式）")

    def _combine_instruction_with_tips(
        self, original_instruction: str, tips: str
    ) -> str:
        if not tips or tips.strip() == "":
            return original_instruction
        tips_lines = [line.strip() for line in tips.split("\n") if line.strip()]
        formatted_tips = []
        for line in tips_lines:
            if line.startswith("- "):
                formatted_tips.append(line)
            elif line.strip():
                formatted_tips.append(f"- {line}")
        if not formatted_tips:
            return original_instruction
        enhanced_instruction = f"{original_instruction}\n\nTips:\n" + "\n".join(
            formatted_tips
        )
        return enhanced_instruction

    def _log_score_progress(self, batch_size: int):
        """输出当前评分进度，按prompt数统计（同一batch同类样本）"""
        prompt_batch_size = max(1, self.n_samples_per_prompt)
        prompts_processed = max(1, math.ceil(batch_size / prompt_batch_size))
        self.total_scores_completed += prompts_processed
        if self.expected_total_scores:
            percentage = min(
                100.0, self.total_scores_completed / self.expected_total_scores * 100
            )
            self.logger.info(
                "[Progress] 已完成评分 %d / %d (%.2f%%)",
                self.total_scores_completed,
                self.expected_total_scores,
                percentage,
            )
        else:
            self.logger.info(
                "[Progress] 已完成评分 %d 条样本", self.total_scores_completed
            )

    def score(self, queries: List[str], prompts: List[str], labels: List[str]):
        # 确保模型已加载
        try:
            self._init_once()
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error(f"模型初始化失败: {e}", exc_info=True)
            return {"rewards": [0.0] * len(queries)}

        if not self.pipeline_enabled:
            return self._process_batch(queries, prompts, labels)

        if self._request_queue is None:
            self._init_pipeline_workers()

        request_id, future = self._submit_pipeline_request(queries, prompts, labels)
        if self.verbose_log:
            self.logger.info(
                f"[PIPELINE] batch {request_id} 已入队 (size={len(prompts)})"
            )

        try:
            return future.result()
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error(f"流水线执行失败: {exc}", exc_info=True)
            raise

    def score_async(self, queries: List[str], prompts: List[str], labels: List[str]):
        try:
            self._init_once()
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error(f"模型初始化失败: {e}", exc_info=True)
            return {"error": str(e)}

        if not self.pipeline_enabled:
            self.logger.warning("异步模式未启用，回退同步score")
            return {"result": self._process_batch(queries, prompts, labels)}

        if self._request_queue is None:
            self._init_pipeline_workers()

        request_id, future = self._submit_pipeline_request(queries, prompts, labels)
        self._register_async_future(request_id, future)
        return {"request_id": request_id}

    @torch.inference_mode()
    def _process_batch(self, queries: List[str], prompts: List[str], labels: List[str]):
        if self.gen_tk is None or len(self.gen_instances) == 0:
            self.logger.error("gen_tk 或 gen_instances 初始化失败")
            return {"rewards": [0.0] * len(queries)}

            # 调试：检查prompts和queries参数的格式
        # if len(prompts) > 0 and len(queries) > 0:
        #     self.logger.info(f"[DEBUG] 接收到 {len(prompts)} 个prompts, {len(queries)} 个queries")
        #     self.logger.info(f"[DEBUG] prompts[0] 长度={len(prompts[0])}, queries[0] 长度={len(queries[0])}")
        #     self.logger.info(f"[DEBUG] prompts[0] 前200字符: {prompts[0][:200]}")
        #     self.logger.info(f"[DEBUG] queries[0] 前200字符: {queries[0][:200]}")
        #     self.logger.info(f"[DEBUG] prompts[0] 后200字符: {prompts[0][-200:]}")
        #     self.logger.info(f"[DEBUG] queries[0] 后200字符: {queries[0][-200:]}")
        #     self.logger.info(f"[DEBUG] prompts[0] 是否包含chat template: {('<|im_start|>' in prompts[0])}")
        #     self.logger.info(f"[DEBUG] queries[0] 是否包含chat template: {('<|im_start|>' in queries[0])}")
        #     self.logger.info(f"[DEBUG] prompts[0] 是否包含TIPS标记: {('[TIPS]' in prompts[0] or 'Instruction:' in prompts[0])}")
        #     self.logger.info(f"[DEBUG] queries[0] 是否包含TIPS标记: {('[TIPS]' in queries[0] or 'Instruction:' in queries[0])}")
        #     self.logger.info(f"[DEBUG] prompts[0] == queries[0]: {(prompts[0] == queries[0])}")

        # 第一步：提取tips/rewrite内容
        # queries: 完整的生成文本（包含TIPS模板 + 原始prompt + tips内容）
        # prompts: 原始的chat-templated prompt（可直接用于匹配数据集）
        all_ins0 = []
        all_tips = []
        all_raw_tips = []
        all_rewritten = []  # 当guidance_mode=rewrite时使用
        all_trunc_reasons = []  # 记录截断原因用于调试

        for i, (full_text, original_prompt) in enumerate(zip(queries, prompts)):
            full_text = (full_text or "").strip()
            original_prompt = (original_prompt or "").strip()

            # prompts 现在直接就是原始的 chat-templated prompt
            # 格式: "<|im_start|>system\n...<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n"
            ins0 = original_prompt

            # 从full_text中提取actor生成的指导内容（tips或rewrite）
            # full_text 格式: "TIPS模板 + ins0 + tips内容"
            # 我们需要找到 ins0 在 full_text 中的位置，然后提取后面的内容
            raw_guidance = ""
            if ins0 and ins0 in full_text:
                pos = full_text.find(ins0)
                raw_guidance = full_text[pos + len(ins0) :].strip()
            else:
                # 如果找不到，可能是因为格式问题，尝试容错
                raw_guidance = full_text

            # 解析两种标记：Tips 或 Rewrite（优先按配置解析）
            tips = ""
            rewritten = ""
            is_none_response = False  # 标记是否为 None 回答

            if self.guidance_mode == "rewrite":
                if "[REWRITE]" in raw_guidance and "[/REWRITE]" in raw_guidance:
                    rs = raw_guidance.find("[REWRITE]") + len("[REWRITE]")
                    re = raw_guidance.find("[/REWRITE]")
                    rewritten = raw_guidance[rs:re].strip()
                else:
                    # 若未按块输出，回退为整段（尽量容错）
                    rewritten = raw_guidance

                # 检查是否为 None 回答
                if rewritten and (
                    rewritten.lower().strip() == "none" or len(rewritten) < 3
                ):
                    is_none_response = True
                    self.logger.info(
                        f"[NONE_RESPONSE] 样本{i} 模型主动拒绝 rewrite (输出: {repr(rewritten[:50])})"
                    )
            else:
                if "[TIPS]" in raw_guidance and "[/TIPS]" in raw_guidance:
                    start = raw_guidance.find("[TIPS]") + len("[TIPS]")
                    end = raw_guidance.find("[/TIPS]")
                    tips = raw_guidance[start:end].strip()
                elif "[TIPS]" in raw_guidance:
                    # 处理只有开始标签没有结束标签的情况
                    start = raw_guidance.find("[TIPS]") + len("[TIPS]")
                    tips = raw_guidance[start:].strip()
                elif "[/TIPS]" in raw_guidance:
                    # 处理只有结束标签没有开始标签的情况
                    end = raw_guidance.find("[/TIPS]")
                    tips = raw_guidance[:end].strip()
                elif "<|im_start|>assistant" in raw_guidance:
                    start = raw_guidance.find("<|im_start|>assistant") + len(
                        "<|im_start|>assistant"
                    )
                    tips = raw_guidance[start:].strip()
                else:
                    tips = raw_guidance
                # self.logger.info(f"样本{i} 提取的 tips: {repr(tips)}")

                # 清理EOS token及其之后的内容
                # 增加 <|im_start|> 以防止模型生成后续的对话轮次（如 <|im_start|>assistant）
                eos_tokens = [
                    "<|im_end|>",
                    "<|end|>",
                    "<|eot_id|>",
                    "<|end_of_text|>",
                    "<|im_start|>",
                ]

                # 动态添加当前tokenizer的eos_token，适应不同的模型（如Qwen3等）
                if self.gen_tk.eos_token and self.gen_tk.eos_token not in eos_tokens:
                    eos_tokens.append(self.gen_tk.eos_token)

                trunc_reason = "length_or_none"
                for eos_token in eos_tokens:
                    if eos_token in tips:
                        tips = tips.split(eos_token)[0].strip()
                        trunc_reason = f"eos_{eos_token}"
                        break

                # 检查是否为 None 回答
                if tips and (tips.lower().strip() == "none" or len(tips) < 20):
                    is_none_response = True
                    self.logger.info(
                        f"[NONE_RESPONSE] 样本{i} 模型主动拒绝提供 tips (输出: {repr(tips[:50])})"
                    )

                if "#no_tip" in tips.lower():
                    is_none_response = True
                    self.logger.info(
                        f"[NONE_RESPONSE] 样本{i} 模型输出了 #no_tip，视为无tips"
                    )

            raw_tips_content = tips  # Capture before splitting
            if "###TIPS###" in tips:
                tips = tips.split("###TIPS###")[-1].strip()
                trunc_reason += "+split_TIPS_tag"
            # 保存提取的ins0和tips
            all_ins0.append(ins0)
            all_tips.append(
                tips if not is_none_response else ""
            )  # None 回答时清空 tips
            all_raw_tips.append(raw_tips_content if not is_none_response else "")
            all_rewritten.append(
                rewritten if not is_none_response else ""
            )  # None 回答时清空 rewritten
            all_trunc_reasons.append(trunc_reason)

        # 第二步：智能截断并构建prompt，优先保证ins0完整

        # 初始化统计变量
        overall_binary_pass_rate = None
        overall_avg_pass_rate = None
        overall_avg_reward = None
        overall_empty_diff = None

        gen_input_texts = []
        # vLLM限制: 输入tokens + 输出tokens ≤ 8192
        # 为代码生成（输出）预留空间，剩余的给输入prompt
        max_input_tokens = 8192 - self.gen_max_new_tokens

        for idx, (ins0, tips, rewritten) in enumerate(
            zip(all_ins0, all_tips, all_rewritten)
        ):
            # ins0格式: "...user内容<|im_end|>\n<|im_start|>assistant\n"
            # 需要在<|im_start|>assistant之前插入tips

            effective_ins0 = (
                rewritten if (self.guidance_mode == "rewrite" and rewritten) else ins0
            )

            # 去掉末尾的<|im_start|>assistant\n，等会儿添加tips后再加回来
            ins0_without_assistant = effective_ins0
            if ins0_without_assistant.endswith("<|im_start|>assistant\n"):
                ins0_without_assistant = ins0_without_assistant[
                    : -len("<|im_start|>assistant\n")
                ]
            elif ins0_without_assistant.endswith("<|im_start|>assistant"):
                ins0_without_assistant = ins0_without_assistant[
                    : -len("<|im_start|>assistant")
                ]

            # 步骤1: 计算基础token长度
            base_content = (
                f"{ins0_without_assistant}\n\nTips:\n"
                if self.guidance_mode == "tips"
                else f"{ins0_without_assistant}\n\n"
            )
            base_with_assistant = f"{base_content}<|im_start|>assistant\n"
            base_tokens = self.gen_tk.encode(base_with_assistant)
            base_len = len(base_tokens)

            # 步骤2: 计算tips可用的token预算
            available_for_tips = max_input_tokens - base_len

            # 步骤3: 构建最终prompt
            if available_for_tips <= 0:
                # 情况A: ins0本身就太长，放弃tips
                if self.verbose_log and idx < 2:
                    self.logger.warning(
                        f"样本{idx} ins0太长({base_len} tokens)，无法添加tips"
                    )
                gen_input_text = f"{ins0_without_assistant}<|im_start|>assistant\n"
            elif tips and len(tips) > 10:
                # 情况B: 有tips，检查是否需要截断
                tips_tokens = self.gen_tk.encode(tips)
                if len(tips_tokens) > available_for_tips:
                    # tips太长，截断到可用空间
                    if self.verbose_log and idx < 2:
                        self.logger.warning(
                            f"样本{idx} tips过长({len(tips_tokens)} tokens)，截断到{available_for_tips}"
                        )
                    truncated_tips_tokens = tips_tokens[:available_for_tips]
                    truncated_tips = self.gen_tk.decode(
                        truncated_tips_tokens, skip_special_tokens=True
                    )
                    if self.guidance_mode == "tips":
                        gen_input_text = f"{ins0_without_assistant}\n\nTips:\n{truncated_tips}\n<|im_start|>assistant\n"
                    else:
                        gen_input_text = (
                            f"{ins0_without_assistant}<|im_start|>assistant\n"
                        )
                else:
                    # tips不长，完整使用
                    if self.guidance_mode == "tips":
                        gen_input_text = f"{ins0_without_assistant}\n\nTips:\n{tips}\n<|im_start|>assistant\n"
                    else:
                        gen_input_text = (
                            f"{ins0_without_assistant}<|im_start|>assistant\n"
                        )
            else:
                # 情况C: 没有tips
                gen_input_text = f"{ins0_without_assistant}<|im_start|>assistant\n"

            # 最终安全检查
            final_tokens = self.gen_tk.encode(gen_input_text)
            if len(final_tokens) > max_input_tokens:
                if self.verbose_log and idx < 2:
                    self.logger.warning(
                        f"样本{idx} 最终prompt仍过长({len(final_tokens)} tokens)，强制截断"
                    )
                gen_input_text = self.gen_tk.decode(
                    final_tokens[:max_input_tokens], skip_special_tokens=False
                )

            gen_input_texts.append(gen_input_text)

        # Step 2.5: 预先构建 Baseline (No Tips) 的 Prompt，并合并到 Batch 中一起生成
        baseline_input_texts = []
        baseline_qids = []
        baseline_group_indices = []

        # 预先解析 QID（用于 Baseline 构建）
        # 注意：这里我们简单地预估 QID，实际评估时会再次解析所有 QID
        # 我们假设输入是按组排列的，每组 n_samples_per_prompt 个样本
        temp_question_ids = []
        for i, original_prompt in enumerate(prompts):
            prompt_text = ""
            if isinstance(original_prompt, list) and original_prompt:
                if (
                    isinstance(original_prompt[0], dict)
                    and "content" in original_prompt[0]
                ):
                    prompt_text = original_prompt[0]["content"]
            elif isinstance(original_prompt, dict) and "content" in original_prompt:
                prompt_text = original_prompt["content"]
            elif isinstance(original_prompt, str):
                # 简单尝试解析
                if original_prompt.startswith("[") and original_prompt.endswith("]"):
                    try:
                        import ast

                        parsed = ast.literal_eval(original_prompt)
                        if (
                            isinstance(parsed, list)
                            and parsed
                            and isinstance(parsed[0], dict)
                            and "content" in parsed[0]
                        ):
                            prompt_text = parsed[0]["content"]
                        else:
                            prompt_text = original_prompt
                    except:
                        prompt_text = original_prompt
                else:
                    prompt_text = original_prompt

            qid = None
            if prompt_text:
                qid = self.prompt_prefix_to_id.get(prompt_text[:200])
            if qid is None and self.question_to_id and prompt_text:
                qid = self.question_to_id.get(prompt_text.strip())

            temp_question_ids.append(qid)

        num_groups = len(prompts) // self.n_samples_per_prompt
        if num_groups == 0 and len(prompts) > 0:
            num_groups = 1

        for group_idx in range(num_groups):
            start_idx = group_idx * self.n_samples_per_prompt
            if start_idx >= len(all_ins0):
                break

            ins0 = all_ins0[start_idx]
            qid = temp_question_ids[start_idx]

            # 如果找不到 QID，baseline 也没法评测，跳过（或者生成了也白生成）
            # 这里生成是为了保持逻辑完整性

            ins0_without_assistant = ins0
            if ins0_without_assistant.endswith("<|im_start|>assistant\n"):
                ins0_without_assistant = ins0_without_assistant[
                    : -len("<|im_start|>assistant\n")
                ]
            elif ins0_without_assistant.endswith("<|im_start|>assistant"):
                ins0_without_assistant = ins0_without_assistant[
                    : -len("<|im_start|>assistant")
                ]

            baseline_text = f"{ins0_without_assistant}<|im_start|>assistant\n"
            baseline_input_texts.append(baseline_text)
            baseline_qids.append(qid)
            baseline_group_indices.append(group_idx)

        # 合并 input texts
        original_batch_size = len(gen_input_texts)
        full_input_texts = gen_input_texts + baseline_input_texts

        generation_time = None
        evaluation_time = None

        # vLLM批量生成参数
        sampling_params = SamplingParams(
            temperature=0,
            top_p=0.95,
            max_tokens=2048,
            repetition_penalty=1.1,  # 增加重复惩罚，防止 ###TIPS### 死循环
        )

        self.logger.info(
            f"开始同步生成 {len(full_input_texts)} 个代码样本 (含 {len(baseline_input_texts)} 个Baseline)..."
        )
        start_ts = time.time()

        # 并发生成
        full_codes = [None] * len(full_input_texts)

        if self.num_gen_instances == 1:
            # 单实例：直接生成
            outputs = self.gen_instances[0].generate(full_input_texts, sampling_params)
            full_codes = [output.outputs[0].text for output in outputs]
        else:
            # 多实例：轮询分配+ThreadPoolExecutor并发
            import concurrent.futures

            def generate_batch(instance_idx, texts):
                """在指定实例上生成"""
                outputs = self.gen_instances[instance_idx].generate(
                    texts, sampling_params, use_tqdm=False
                )
                return [output.outputs[0].text for output in outputs]

            # 轮询分配任务
            tasks = [[] for _ in range(self.num_gen_instances)]
            indices = [[] for _ in range(self.num_gen_instances)]
            for i, text in enumerate(full_input_texts):
                inst_id = i % self.num_gen_instances
                tasks[inst_id].append(text)
                indices[inst_id].append(i)

            # 并发执行
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_gen_instances
            ) as executor:
                futures = []
                for inst_id in range(self.num_gen_instances):
                    if tasks[inst_id]:
                        future = executor.submit(
                            generate_batch, inst_id, tasks[inst_id]
                        )
                        futures.append((future, indices[inst_id]))

                # 收集结果
                for future, task_indices in futures:
                    codes = future.result()
                    for local_idx, code in enumerate(codes):
                        global_idx = task_indices[local_idx]
                        full_codes[global_idx] = code

        generation_time = time.time() - start_ts
        self.logger.info(
            "同步生成完成，共生成 %d 个样本，耗时 %.2fs",
            len(full_codes),
            generation_time,
        )

        # 拆分结果
        all_codes = full_codes[:original_batch_size]
        baseline_codes = full_codes[original_batch_size:]

        if self.verbose_log:
            self.logger.info(f"批量生成完成，共生成 {len(all_codes)} 个样本")

        # 去除逐样本代码预览日志，仅保留必要统计

        # 第三步：Test case执行模式打分
        from openrlhf.cli.eval_test_cases import evaluate

        rewards: List[float] = []

        # 直接使用传入的prompts作为原始prompts进行数据集匹配
        question_ids = []
        for i, original_prompt in enumerate(prompts):
            # self.logger.info(f"=== RM INPUT DEBUG ===")
            # self.logger.info(f"original_prompt {i} type: {type(original_prompt)}")
            # self.logger.info(f"original_prompt {i} content: {repr(original_prompt)[:500]}")

            # 处理不同的prompt格式
            prompt_text = ""
            if isinstance(original_prompt, list) and original_prompt:
                # 如果是conversation格式，提取content
                if (
                    isinstance(original_prompt[0], dict)
                    and "content" in original_prompt[0]
                ):
                    prompt_text = original_prompt[0]["content"]
                    self.logger.info(f"Extracted content: {repr(prompt_text[:200])}")
            elif isinstance(original_prompt, dict) and "content" in original_prompt:
                prompt_text = original_prompt["content"]
                self.logger.info(f"Extracted dict content: {repr(prompt_text[:200])}")
            elif isinstance(original_prompt, str):
                # 如果是字符串，尝试解析为Python对象
                if original_prompt.startswith("[") and original_prompt.endswith("]"):
                    try:
                        import ast

                        parsed = ast.literal_eval(original_prompt)
                        if (
                            isinstance(parsed, list)
                            and parsed
                            and isinstance(parsed[0], dict)
                            and "content" in parsed[0]
                        ):
                            prompt_text = parsed[0]["content"]
                            # self.logger.info(f"Parsed string content: {repr(prompt_text[:200])}")
                        else:
                            prompt_text = original_prompt
                            # self.logger.info(f"Failed to parse string, using as-is: {repr(prompt_text[:200])}")
                    except:
                        prompt_text = original_prompt
                        # self.logger.info(f"Parse failed, using string as-is: {repr(prompt_text[:200])}")
                else:
                    # 如果是普通字符串，直接使用
                    prompt_text = original_prompt
                    # self.logger.info(f"Plain string prompt: {repr(prompt_text[:200])}")

            qid = None
            # 首先尝试使用前200个字符进行匹配（question前缀）
            if prompt_text:
                prompt_prefix = prompt_text[:200]
                qid = self.prompt_prefix_to_id.get(prompt_prefix)

            # 如果找不到，尝试直接匹配question文本
            if qid is None and self.question_to_id and prompt_text:
                qid = self.question_to_id.get(prompt_text.strip())
                if qid:
                    self.logger.info(
                        f"Matched via exact text (stripped) for prompt: {repr(prompt_text[:50])}..."
                    )

            # self.logger.info(f"Final qid: {qid}")
            question_ids.append(qid)

            # 验证所有问题都找到了对应的ID
        missing_ids = [i for i, qid in enumerate(question_ids) if qid is None]
        if missing_ids:
            self.logger.error(f"发现 {len(missing_ids)} 个prompt无法匹配到数据集ID")

            # 为无法匹配的问题返回0分
            rewards = [0.0] * len(all_ins0)
            self.logger.warning("存在无法匹配的prompt，全部返回0分")
        else:
            # 获取对应的test cases
            gt_key = os.environ.get("RM_GT_KEY", "test_cases")
            test_cases_list = [self.id_map[qid][gt_key] for qid in question_ids]

            # 构造evaluate函数需要的samples格式
            samples = [
                {
                    "task_id": question_id,
                    "prompt": self.id_map[question_id].get(
                        "question", ""
                    ),  # 使用数据集中的原始问题
                    "output": code,
                    "tests": test_cases,
                    "_identifier": f"{question_id}_{i}",
                }
                for i, (question_id, code, test_cases) in enumerate(
                    zip(question_ids, all_codes, test_cases_list)
                )
            ]

            eval_start = time.time()
            all_samples_results, pass_rates = evaluate(
                samples, n_workers=self.test_case_n_workers, test_details=True
            )
            evaluation_time = time.time() - eval_start
            self.logger.info("Test case评估完成，耗时 %.2fs", evaluation_time)

            # 原始pass_rate（用于统计）
            raw_pass_rates = pass_rates.copy()

            # 计算binary pass rate（用于统计）
            binary_pass_rates = [1.0 if x == 1.0 else 0.0 for x in raw_pass_rates]

            rewards = raw_pass_rates

            # Step 1.5: 评估 Baseline (No Tips)
            # Baseline 代码已经在前面生成好了，这里只需要进行 evaluate

            baseline_pass_rates_map = {}  # group_idx -> pass_rate

            if len(baseline_codes) > 0:
                self.logger.info(
                    f"开始评估 Baseline (No Tips), 共 {len(baseline_codes)} 条..."
                )

                b_samples = []
                for i, code in enumerate(baseline_codes):
                    qid = baseline_qids[i]
                    if qid and qid in self.id_map:
                        test_cases = self.id_map[qid][gt_key]
                        b_samples.append(
                            {
                                "task_id": qid,
                                "prompt": self.id_map[qid].get("question", ""),
                                "output": code,
                                "tests": test_cases,
                                "_identifier": f"baseline_{qid}",
                            }
                        )
                    else:
                        # 应该不会发生，除非qid在生成后变了或者map不对
                        pass

                if b_samples:
                    _, b_pass_rates = evaluate(
                        b_samples,
                        n_workers=self.test_case_n_workers,
                        test_details=False,
                    )

                    for i, pr in enumerate(b_pass_rates):
                        group_idx = baseline_group_indices[i]
                        baseline_pass_rates_map[group_idx] = pr

                    self.logger.info(f"Baseline评估完成")

            batch_none_means = {}
            none_indices = []

            # 计算 batch_none_means，优先使用实时生成的 baseline_pass_rates_map
            # num_groups 已经在上面计算过，这里可能需要重新计算，因为 rewards 是基于 all_codes 的
            current_num_groups = len(question_ids) // self.n_samples_per_prompt
            if current_num_groups == 0 and len(question_ids) > 0:
                current_num_groups = 1

            for group_idx in range(current_num_groups):
                if group_idx in baseline_pass_rates_map:
                    batch_none_means[group_idx] = baseline_pass_rates_map[group_idx]
                else:
                    # 如果该 group 没有对应的 baseline 生成结果（理论上不应发生，除非 gen 失败）
                    # 回退到预计算的 baseline 或者 0.0
                    start_idx = group_idx * self.n_samples_per_prompt
                    if start_idx < len(question_ids):
                        qid = question_ids[start_idx]
                        batch_none_means[group_idx] = (
                            self.baseline_map.get(qid, 0.0) if qid else 0.0
                        )
                    else:
                        batch_none_means[group_idx] = 0.0

            # Step 2: 应用baseline比较（细化规则）
            batch_stats = {}
            if self.use_baseline_comparison and self.baseline_map:
                baseline_rewards = []
                empty_diffs = []
                final_rewards = []

                for i, question_id in enumerate(question_ids):
                    baseline = self.baseline_map.get(question_id, 0.0)
                    current_pass_rate = raw_pass_rates[i]
                    # 使用该样本所在batch的none_reply均值作为empty_reward
                    batch_idx = i // self.n_samples_per_prompt
                    empty_pass = batch_none_means[batch_idx]

                    # if empty_pass < 1.0 - 1e-5 and current_pass_rate >= 1.0 - 1e-5:
                    #     reward_diff = 1.0
                    # elif empty_pass >= 1.0 - 1e-5 and current_pass_rate >= 1.0 - 1e-5:
                    #     reward_diff = 0.1
                    # else:
                    #     reward_diff = 0

                    if empty_pass < 1.0 - 1e-5 and current_pass_rate >= 1.0 - 1e-5:
                        reward_diff = 1.0
                    elif empty_pass >= 1.0 - 1e-5 and current_pass_rate >= 1.0 - 1e-5:
                        reward_diff = 0.1
                    else:
                        reward_diff = (current_pass_rate - empty_pass) / 3

                    empty_diff = current_pass_rate - empty_pass

                    baseline_rewards.append(reward_diff)
                    empty_diffs.append(empty_diff)
                rewards = baseline_rewards

                diffs = []
                avg_baseline = 0.0
                for i, question_id in enumerate(question_ids):
                    baseline = self.baseline_map.get(question_id, 0.0)
                    diffs.append(raw_pass_rates[i] - baseline)
                    avg_baseline += baseline

                avg_baseline /= len(question_ids) if question_ids else 1.0
                avg_improvement = sum(diffs) / len(diffs) if diffs else 0.0
                perfect_count = sum(1 for r in raw_pass_rates if abs(r - 1.0) < 1e-6)

                # Calculate non-empty stats
                non_empty_indices_set = set(none_indices)
                non_empty_pass_rates = [
                    r
                    for i, r in enumerate(raw_pass_rates)
                    if i not in non_empty_indices_set
                ]
                avg_non_empty_pass_rate = (
                    sum(non_empty_pass_rates) / len(non_empty_pass_rates)
                    if non_empty_pass_rates
                    else 0.0
                )

                # Calculate empty stats
                empty_pass_rates = [
                    r
                    for i, r in enumerate(raw_pass_rates)
                    if i in non_empty_indices_set
                ]
                avg_empty_pass_rate = (
                    sum(empty_pass_rates) / len(empty_pass_rates)
                    if empty_pass_rates
                    else 0.0
                )

                avg_empty_reward = (
                    sum(batch_none_means.values()) / len(batch_none_means)
                    if batch_none_means
                    else 0.0
                )

                batch_pass_rate = (
                    sum(raw_pass_rates) / len(raw_pass_rates) if raw_pass_rates else 0.0
                )

                batch_stats = {
                    "avg_baseline": avg_baseline,
                    "avg_empty_reward": avg_empty_reward,
                    "batch_pass_rate": batch_pass_rate,
                    "avg_non_empty_pass_rate": avg_non_empty_pass_rate,
                    "avg_empty_pass_rate": avg_empty_pass_rate,
                    "non_empty_count": len(non_empty_pass_rates),
                    "empty_count": len(empty_pass_rates),
                    "perfect_count": perfect_count,
                    "total_count": len(rewards),
                }

            # # Step 3: 应用 None 回答的基础 reward
            # none_response_bonus = 0.05
            # none_count = len(none_indices)

            # # 应用基础reward给主动拒绝样本
            # for i in none_indices:
            #     rewards[i] = none_response_bonus if (baseline >= 0.8 and empty_reward >= 0.8) else 0.0

            # Step 4: 应用长度惩罚
            if self.length_penalty_weight > 0:
                # 重新计算 tips_lengths (使用所有非none回复)
                tips_lengths = []
                for i, tips in enumerate(all_tips):
                    if i in none_indices:
                        tips_lengths.append(0)

                    elif tips:
                        tips_lengths.append(
                            len(self.gen_tk.encode(tips, add_special_tokens=False))
                        )
                    else:
                        tips_lengths.append(0)

                # 将字符阈值转换大概的 Token 阈值
                token_threshold = 1000

                length_penalties = [
                    -self.length_penalty_weight
                    * (max((length - token_threshold), 0) / 100.0)
                    for length in tips_lengths
                ]

                # 应用惩罚
                if len(length_penalties) == len(rewards):
                    for i in range(len(rewards)):
                        rewards[i] += length_penalties[i]
                else:
                    self.logger.error(
                        f"Length penalties size mismatch: {len(length_penalties)} vs {len(rewards)}"
                    )

                avg_length = (
                    sum(tips_lengths) / len(tips_lengths) if tips_lengths else 0.0
                )
                avg_penalty = (
                    sum(length_penalties) / len(length_penalties)
                    if length_penalties
                    else 0.0
                )
                self.logger.info(
                    f"平均tips长度(Tokens): {avg_length:.1f}, 平均长度惩罚: {avg_penalty:.4f}"
                )

            # ===== 统计输出 =====
            num_prompts = len(all_ins0) // self.n_samples_per_prompt

            # 打印前10个batch的tips用于观察
            max_batches_to_print = min(10, num_prompts)
            self.logger.info(f"[DEBUG] 打印前{max_batches_to_print}个batch的tips详情:")
            for batch_idx in range(max_batches_to_print):
                start_idx = batch_idx * self.n_samples_per_prompt
                end_idx = start_idx + self.n_samples_per_prompt
                self.logger.info(
                    f"[BATCH {batch_idx}] 包含样本 {start_idx} 到 {end_idx - 1}:"
                )
                for sample_idx in range(start_idx, end_idx):
                    if sample_idx < len(all_tips):
                        tips = all_tips[sample_idx]
                        raw_tips = all_raw_tips[sample_idx]  # Added raw tips
                        trunc_reason = all_trunc_reasons[sample_idx]
                        is_none = len(tips) < 20
                        pass_rate = (
                            raw_pass_rates[sample_idx]
                            if sample_idx < len(raw_pass_rates)
                            else 0.0
                        )

                        # 计算token长度
                        token_len = 0
                        if tips and self.gen_tk:
                            token_len = len(
                                self.gen_tk.encode(tips, add_special_tokens=False)
                            )

                        # 计算完整生成的token长度 (Raw Response)
                        full_token_len = 0
                        if sample_idx < len(queries) and self.gen_tk:
                            full_text = queries[sample_idx]
                            # 尝试去除prompt获取response部分
                            response_text = full_text

                            # 获取对应的prompt (prompts列表长度一定等于queries列表长度)
                            if sample_idx < len(prompts):
                                prompt_text = prompts[sample_idx]
                                if full_text.startswith(prompt_text):
                                    response_text = full_text[len(prompt_text) :]

                            full_token_len = len(
                                self.gen_tk.encode(
                                    response_text, add_special_tokens=False
                                )
                            )

                        self.logger.info(
                            f"  样本{sample_idx}: tips长度(Tokens)={token_len}, 完整响应(Tokens)={full_token_len}, 是否None={is_none}, PassRate={pass_rate:.2f}, CutBy={trunc_reason}"
                        )
                        if raw_tips:  # Print raw tips
                            self.logger.info(f"    内容: {raw_tips}")
                        else:
                            self.logger.info(f"    内容: (空)")
                self.logger.info("")
            group_binary_pass_rates = []
            group_avg_pass_rates = []
            group_final_rewards = []
            group_empty_diffs = []

            for i in range(num_prompts):
                start_idx = i * self.n_samples_per_prompt
                end_idx = start_idx + self.n_samples_per_prompt

                group_binary = binary_pass_rates[start_idx:end_idx]
                group_raw = raw_pass_rates[start_idx:end_idx]
                group_reward = rewards[start_idx:end_idx]
                group_empty_diff = empty_diffs[start_idx:end_idx]

                group_binary_pass = sum(group_binary) / len(group_binary)
                group_binary_pass_rates.append(group_binary_pass)
                group_avg_pass_rates.append(sum(group_raw) / len(group_raw))
                group_final_rewards.append(sum(group_reward) / len(group_reward))

                current_group_non_empty_diffs = [
                    empty_diffs[j]
                    for j in range(start_idx, end_idx)
                    if j not in non_empty_indices_set
                ]
                if current_group_non_empty_diffs:
                    group_empty_diffs.append(
                        sum(current_group_non_empty_diffs)
                        / len(current_group_non_empty_diffs)
                    )

            overall_binary_pass_rate = (
                sum(group_binary_pass_rates) / len(group_binary_pass_rates)
                if group_binary_pass_rates
                else 0.0
            )
            overall_avg_pass_rate = (
                sum(group_avg_pass_rates) / len(group_avg_pass_rates)
                if group_avg_pass_rates
                else 0.0
            )
            overall_avg_reward = (
                sum(group_final_rewards) / len(group_final_rewards)
                if group_final_rewards
                else 0.0
            )
            overall_empty_diff = (
                sum(group_empty_diffs) / len(group_empty_diffs)
                if group_empty_diffs
                else 0.0
            )

            # 新增统计：Good/Bad Action 定义
            # Good Action: Easy且闭嘴 OR Hard且Tips带来提升
            # Bad Action: Easy且乱说 OR Hard且Tips导致下降
            logs_good_actions = []
            logs_bad_actions = []
            logs_perfect_actions = []

            # none_indices 包含的是没有生成tips的样本索引
            none_indices_set = set(none_indices)

            for batch_idx in range(num_prompts):
                start_idx = batch_idx * self.n_samples_per_prompt
                end_idx = start_idx + self.n_samples_per_prompt

                empty_pass = batch_none_means.get(batch_idx, 0.0)
                is_easy = empty_pass >= 1.0 - 1e-5

                for i in range(start_idx, end_idx):
                    has_tips = i not in none_indices_set
                    pass_rate = raw_pass_rates[i]

                    is_good = 0.0
                    is_bad = 0.0

                    if is_easy:
                        if not has_tips:
                            is_good = 1.0  # Easy + No Tips = Good
                        else:
                            is_bad = 1.0  # Easy + Tips = Bad
                    else:
                        # Hard sample
                        if has_tips:
                            if pass_rate > empty_pass + 1e-5:
                                is_good = 1.0  # Hard + Tips + Improve = Good
                            elif pass_rate < empty_pass - 1e-5:
                                is_bad = 1.0  # Hard + Tips + Drop = Bad
                        # Hard + No Tips -> Neutral (not counted as Good or Bad)
                        # Hard + Tips + Equal -> Neutral

                    # Perfect Action: Just checks if the result is correct, regardless of baseline difficulty
                    is_perfect = 1.0 if pass_rate >= 1.0 - 1e-5 else 0.0

                    logs_good_actions.append(is_good)
                    logs_bad_actions.append(is_bad)
                    logs_perfect_actions.append(is_perfect)

            # --- Rollout Rejection Logic ---
            rollout_keep_flags = []  # 1.0 = keep, 0.0 = reject

            for batch_idx in range(num_prompts):
                start_idx = batch_idx * self.n_samples_per_prompt
                end_idx = start_idx + self.n_samples_per_prompt

                # Get group stats
                empty_pass = batch_none_means.get(batch_idx, 0.0)
                group_pass_rates = raw_pass_rates[start_idx:end_idx]

                # Rules
                is_baseline_perfect = empty_pass >= 1.0 - 1e-5
                has_perfect_sample = any(r >= 1.0 - 1e-5 for r in group_pass_rates)

                should_keep = True
                reject_reason = ""

                if is_baseline_perfect:
                    should_keep = False
                    reject_reason = "baseline_perfect"
                elif not has_perfect_sample:
                    should_keep = False
                    reject_reason = "no_perfect_sample"

                # Apply decision to all samples in this group
                keep_val = 1.0 if should_keep else 0.0
                rollout_keep_flags.extend([keep_val] * (end_idx - start_idx))

                # Log rejections for debugging (sampled log to avoid flooding)
                if not should_keep and batch_idx < 5:
                    qid = (
                        question_ids[start_idx]
                        if start_idx < len(question_ids)
                        else "unknown"
                    )
                    self.logger.info(
                        f"[Rollout Filter] Rejected group {qid}: {reject_reason} (Baseline={empty_pass:.2f}, MaxPass={max(group_pass_rates) if group_pass_rates else 0:.2f})"
                    )

            stat_good_action_rate = (
                sum(logs_good_actions) / len(logs_good_actions)
                if logs_good_actions
                else 0.0
            )
            stat_bad_action_rate = (
                sum(logs_bad_actions) / len(logs_bad_actions)
                if logs_bad_actions
                else 0.0
            )
            stat_perfect_action_rate = (
                sum(logs_perfect_actions) / len(logs_perfect_actions)
                if logs_perfect_actions
                else 0.0
            )
            stat_rollout_keep_rate = (
                sum(rollout_keep_flags) / len(rollout_keep_flags)
                if rollout_keep_flags
                else 0.0
            )

            self.logger.info("=" * 60)
            self.logger.info(
                f"📊 Reward统计 (groups={num_prompts}, n_samples_per_prompt={self.n_samples_per_prompt})"
            )

            # 输出Instruction IDs
            unique_qids = sorted(list(set([q for q in question_ids if q is not None])))
            self.logger.info(f"  Instruction IDs: {unique_qids}")

            # 1. Overall Performance
            self.logger.info(f"  [Overall Performance]")
            self.logger.info(f"  • Average Pass Rate: {overall_avg_pass_rate:.4f}")
            if batch_stats:
                self.logger.info(
                    f"  • Baseline Avg:      {batch_stats['avg_baseline']:.4f}"
                )
            self.logger.info(f"  • Binary Pass Rate:  {overall_binary_pass_rate:.4f}")

            # 2. Breakdown by Type
            if batch_stats:
                self.logger.info(f"  [Breakdown by Type]")
                self.logger.info(
                    f"  • Non-Empty Pass Rate: {batch_stats['avg_non_empty_pass_rate']:.4f} (count={batch_stats['non_empty_count']})"
                )
                self.logger.info(
                    f"  • Empty Pass Rate:     {batch_stats['avg_empty_pass_rate']:.4f} (count={batch_stats['empty_count']}, baseline={batch_stats['avg_empty_reward']:.4f})"
                )

            # 3. Reward Signals
            self.logger.info(f"  [Reward Signals]")
            self.logger.info(f"  • Final Avg Reward:  {overall_avg_reward:.4f}")
            self.logger.info(
                f"  • Avg Empty Diff:    {overall_empty_diff:.4f} (Non-Empty - EmptyValue)"
            )

            # 4. Action Stats
            self.logger.info(f"  [Action Stats]")
            self.logger.info(
                f"  • Good Action Rate:  {stat_good_action_rate:.4f} ({int(sum(logs_good_actions))}/{len(logs_good_actions)})"
            )
            self.logger.info(
                f"  • Bad Action Rate:   {stat_bad_action_rate:.4f} ({int(sum(logs_bad_actions))}/{len(logs_bad_actions)})"
            )
            self.logger.info(
                f"  • Perfect Action Rate: {stat_perfect_action_rate:.4f} ({int(sum(logs_perfect_actions))}/{len(logs_perfect_actions)})"
            )
            self.logger.info(
                f"  • Rollout Keep Rate:   {stat_rollout_keep_rate:.4f} ({int(sum(rollout_keep_flags))}/{len(rollout_keep_flags)})"
            )

            # 5. Misc
            if batch_stats:
                self.logger.info(f"  [Misc]")
                self.logger.info(
                    f"  • Perfect Count:     {batch_stats['perfect_count']}/{batch_stats['total_count']}"
                )

            self.logger.info("=" * 60)
            # 若该batch整体为0，输出完整代码与执行结果用于排查
            # if overall_binary_pass_rate == 0.0 and overall_avg_pass_rate == 0.0:
            #     self.logger.info("—— 该batch所有样本均未通过，输出详细信息用于调试 ——")
            #     for i, (s, code) in enumerate(zip(samples, all_codes)):
            #         self.logger.info(f"\n[#样本{i}] task_id={s.get('task_id')}")

            #         # 输出原始问题（用于匹配数据集）
            #         if i < len(all_ins0):
            #             original_q = all_ins0[i]
            #             self.logger.info(f"[ORIGINAL_QUESTION] (length={len(original_q)})")
            #             self.logger.info(original_q[:500] + "..." if len(original_q) > 500 else original_q)

            #         # 输出Actor生成的tips
            #         if i < len(all_tips) and all_tips[i]:
            #             tips_content = all_tips[i]
            #             self.logger.info(f"[TIPS] (length={len(tips_content)})")
            #             self.logger.info(tips_content[:500] + "..." if len(tips_content) > 500 else tips_content)

            #         # 输出最终的生成prompt（传给M_gen的）
            #         if i < len(gen_input_texts):
            #             final_prompt = gen_input_texts[i]
            #             self.logger.info(f"[FINAL_PROMPT_TO_GEN] (length={len(final_prompt)})")
            #             self.logger.info(final_prompt[:500] + "..." if len(final_prompt) > 500 else final_prompt)

            #         # 输出生成的代码
            #         self.logger.info("[GENERATED_CODE]")
            #         self.logger.info(code or "(empty)")

            #         # 输出测试结果
            #         res = all_samples_results[i] if i < len(all_samples_results) else None
            #         if res is not None:
            #             self.logger.info(f"[TEST_RESULTS]")
            #             self.logger.info(f"  pass_rate: {res.get('pass_rate', 'N/A')}")
            #             self.logger.info(f"  n_tests: {res.get('n_tests', 'N/A')}")
            #             self.logger.info(f"  base: {res.get('base', 'N/A')}")

            #         self.logger.info("-" * 80)
            # self.logger.info("=" * 60)

        # 返回rewards和统计信息（统计信息会被集成到global step输出并记录到wandb）
        result = {"rewards": rewards}

        extra_logs = {}
        if overall_binary_pass_rate is not None and len(rewards) > 0:
            extra_logs.update(
                {
                    "rm_binary_pass_rate": [overall_binary_pass_rate] * len(rewards),
                    "rm_avg_pass_rate": [overall_avg_pass_rate] * len(rewards),
                    "rm_avg_reward": [overall_avg_reward] * len(rewards),
                    "rm_avg_empty_diff": [overall_empty_diff] * len(rewards),
                    "rm_good_action_rate": [stat_good_action_rate] * len(rewards),
                    "rm_bad_action_rate": [stat_bad_action_rate] * len(rewards),
                    "rm_perfect_action_rate": [stat_perfect_action_rate] * len(rewards),
                    "rm_rollout_keep": rollout_keep_flags,
                }
            )

        # # 始终添加 Key，确保汇总时 Key 存在（即使为空列表）
        # extra_logs["rm_good_action_rate"] = logs_good_actions
        # extra_logs["rm_bad_action_rate"] = logs_bad_actions

        if extra_logs:
            result["extra_logs"] = extra_logs

        self._log_score_progress(len(rewards))
        return result
