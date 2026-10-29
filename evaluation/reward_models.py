"""Reward scorers extracted from the existing evaluation pipeline."""


def load_ace_rm(model_path: str):
    import os
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    class ValueHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            p = getattr(config, "summary_dropout_prob", 0.1)
            self.dropout = nn.Dropout(p) if p else nn.Identity()
            hidden_size = getattr(config, "hidden_size", None)
            if hasattr(config, "word_embed_proj_dim"):
                hidden_size = config.word_embed_proj_dim
            elif getattr(config, "is_encoder_decoder", False) and hasattr(
                config, "decoder"
            ):
                hidden_size = getattr(config.decoder, "hidden_size", hidden_size)
            self.summary = nn.Linear(hidden_size, 1)

        def forward(self, hidden_states):
            output = self.dropout(hidden_states)
            if output.dtype != self.summary.weight.dtype:
                output = output.to(self.summary.weight.dtype)
            return self.summary(output)

    class AceCodeRM(Qwen2ForCausalLM):
        def __init__(self, config):
            super().__init__(config)
            self.v_head = ValueHead(config)

        def forward(
            self, input_ids=None, attention_mask=None, past_key_values=None, **kwargs
        ):
            kwargs["output_hidden_states"] = True
            kwargs["return_dict"] = True
            kwargs["past_key_values"] = past_key_values
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
            last_hidden = outputs.hidden_states[-1]
            if last_hidden.device != self.v_head.summary.weight.device:
                last_hidden = last_hidden.to(self.v_head.summary.weight.device)
            value = self.v_head(last_hidden).squeeze(-1)
            if attention_mask.device != value.device:
                attention_mask = attention_mask.to(value.device)
            index = attention_mask.sum(dim=-1, keepdim=True) - 1
            return value.gather(dim=-1, index=index).squeeze()

    device_map = os.environ.get("ACE_RM_DEVICE_MAP")
    if not device_map:
        device_map = (
            "auto"
            if torch.cuda.device_count() > 1
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
    max_memory = None
    max_memory_env = os.environ.get("ACE_RM_MAX_MEMORY")
    if max_memory_env and torch.cuda.is_available():
        max_memory = {i: max_memory_env for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = os.environ.get("ACE_RM_CPU_MAX_MEMORY", "120GiB")

    kwargs = {
        "device_map": device_map,
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }
    if max_memory is not None:
        kwargs["max_memory"] = max_memory

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AceCodeRM.from_pretrained(model_path, **kwargs)
    model.eval()
    return tokenizer, model


def load_internlm_rm(model_path: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def score_ace(
    tokenizer,
    model,
    prompts: list[str],
    codes: list[str],
    batch_size: int,
    max_length: int,
) -> list[float]:
    import torch

    tokenizer.padding_side = "right"
    scores: list[float] = []
    for start in range(0, len(prompts), batch_size):
        texts = []
        for instr, code in zip(
            prompts[start : start + batch_size], codes[start : start + batch_size]
        ):
            messages = [
                {"role": "user", "content": instr},
                {"role": "assistant", "content": code},
            ]
            texts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            batch_scores = model(**inputs, use_cache=False)
        if batch_scores.dim() == 0:
            scores.append(float(batch_scores.item()))
        else:
            scores.extend(float(x) for x in batch_scores.detach().cpu().tolist())
        del inputs, batch_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[rm] scored {min(start + batch_size, len(prompts))}/{len(prompts)}",
            flush=True,
        )
    return scores


def score_internlm(
    tokenizer,
    model,
    prompts: list[str],
    codes: list[str],
    batch_size: int,
    max_length: int,
) -> list[float]:
    import torch

    del max_length
    all_scores: list[float] = []
    for start in range(0, len(prompts), batch_size):
        chats = [
            [{"role": "user", "content": instr}, {"role": "assistant", "content": code}]
            for instr, code in zip(
                prompts[start : start + batch_size], codes[start : start + batch_size]
            )
        ]
        with torch.no_grad():
            scores = model.get_scores(tokenizer, chats)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if not isinstance(scores, list):
            scores = [scores]
        if len(scores) != len(chats):
            scores = [model.get_score(tokenizer, chat) for chat in chats]
        all_scores.extend(float(x) for x in scores)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[rm] scored {min(start + batch_size, len(prompts))}/{len(prompts)}",
            flush=True,
        )
    return all_scores
