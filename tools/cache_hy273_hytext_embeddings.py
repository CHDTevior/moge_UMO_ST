#!/usr/bin/env python
"""Pre-encode HumanML3D/MotionFix captions with HY-Motion Qwen3 + CLIP-L text towers."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTextModel, CLIPTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.kimodo273_datasets import parse_hml_text_line
from models.raw_motion.hytext_cache import hytext_key, normalize_text_key


PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION = """
    Summarize human motion only from the user text for representation: action categories, key body-part movements, order/transitions, trajectory/direction, posture; include style/emotion/speed only if present. Explicitly capture laterality (left/right) when mentioned; do not guess. If multiple actions are described, indicate the count of distinct actions (e.g., actions=3) and their order. Do not invent missing info. Keep one concise paragraph.
"""

PROMPT_TEMPLATE_ENCODE_RELATIVE_EDIT = """
    Summarize the requested change from a source human motion to a target motion for representation. Capture the edited body parts, laterality, timing/order, trajectory/direction, amplitude, contacts, repetition count, speed, style, and what should be preserved when stated. Keep relative words such as higher, wider, earlier, slower, opposite, or instead of; do not rewrite the instruction as an absolute target-motion caption. Do not invent missing information. Keep one concise paragraph.
"""


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22")
    p.add_argument("--text_root", default="/mnt/afs/mogo_base/datasets/HumanML3D/texts")
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--output_dir", default="/mnt/afs/mogo_base/datasets/HumanML3D/hytext_qwen3_clipL_mlen128")
    p.add_argument("--qwen_path", default="/mnt/afs/HY-Motion-1.0/ckpts/Qwen3-8B")
    p.add_argument("--clip_path", default="/mnt/afs/HY-Motion-1.0/ckpts/clip-vit-large-patch14")
    p.add_argument("--max_length_llm", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--shard_size", type=int, default=4096)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--storage_dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--include_empty", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true")
    return p


def read_captions(path: Path) -> list[str]:
    if not path.is_file():
        return [""]
    captions: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = parse_hml_text_line(line)
            if text:
                captions.append(text)
    return captions or [""]


def collect_texts(data_root: Path, text_root: Path, splits: list[str], include_empty: bool, limit: int) -> list[dict[str, str]]:
    seen: set[str] = set()
    rows: list[dict[str, str]] = []

    def add(text: str, source: str) -> None:
        norm = normalize_text_key(text)
        key = hytext_key(norm)
        if key in seen:
            return
        seen.add(key)
        rows.append({"key": key, "text": norm, "source": source})

    if include_empty:
        add("", "__empty__")

    split_dir = data_root / "split_existing"
    motion_ids: list[str] = []
    for split in splits:
        split_path = split_dir / f"{split}.txt"
        if split_path.is_file():
            motion_ids.extend(line.strip() for line in split_path.read_text().splitlines() if line.strip())

    if motion_ids:
        for motion_id in motion_ids:
            for caption in read_captions(text_root / f"{motion_id}.txt"):
                add(caption, motion_id)
                if limit > 0 and len(rows) >= limit + int(include_empty):
                    return rows
    else:
        for text_path in sorted(text_root.glob("*.txt")):
            for caption in read_captions(text_path):
                add(caption, text_path.stem)
                if limit > 0 and len(rows) >= limit + int(include_empty):
                    return rows
    return rows


def apply_chat(
    tokenizer: Any,
    text: str,
    prompt_template: str = PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION,
) -> str:
    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def find_subseq(a: list[int], b: list[int]) -> int:
    for i in range(0, len(a) - len(b) + 1):
        if a[i : i + len(b)] == b:
            return i
    return -1


def compute_crop_start(
    tokenizer: Any,
    prompt_template: str = PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION,
) -> int:
    marker = "<BOC>"
    full_text = apply_chat(tokenizer, marker, prompt_template)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
    marker_ids = tokenizer(marker, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    pos = find_subseq(full_ids, marker_ids)
    return pos if pos >= 0 else max(0, len(full_ids) - 1)


def dtype_from_name(name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


@torch.inference_mode()
def encode_batch(
    texts: list[str],
    qwen_model: torch.nn.Module,
    qwen_tokenizer: Any,
    clip_model: torch.nn.Module,
    clip_tokenizer: Any,
    device: torch.device,
    crop_start: int,
    max_length_llm: int,
    prompt_template: str = PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    llm_text = [apply_chat(qwen_tokenizer, text, prompt_template) for text in texts]
    llm_encoding = qwen_tokenizer(
        llm_text,
        return_overflowing_tokens=False,
        truncation=True,
        return_attention_mask=True,
        max_length=crop_start + max_length_llm,
        padding="max_length",
        return_tensors="pt",
    )
    qwen_inputs = {
        "input_ids": llm_encoding["input_ids"].to(device),
        "attention_mask": llm_encoding["attention_mask"].to(device),
        "output_hidden_states": True,
        "use_cache": False,
    }
    if hasattr(qwen_model, "model"):
        llm_outputs = qwen_model.model(**qwen_inputs)
    else:
        llm_outputs = qwen_model(**qwen_inputs)
    hidden = llm_outputs.hidden_states[-1]
    ctxt_raw = hidden[:, crop_start : crop_start + max_length_llm].contiguous()
    ctxt_length = (llm_encoding["attention_mask"].sum(dim=-1).to(device) - crop_start).clamp(
        min=0,
        max=max_length_llm,
    )

    clip_encoding = clip_tokenizer(
        texts,
        return_overflowing_tokens=False,
        truncation=True,
        return_attention_mask=True,
        max_length=77,
        padding=True,
        return_tensors="pt",
    )
    clip_outputs = clip_model(
        input_ids=clip_encoding["input_ids"].to(device),
        attention_mask=clip_encoding["attention_mask"].to(device),
    )
    if hasattr(clip_outputs, "pooler_output") and clip_outputs.pooler_output is not None:
        vtxt_raw = clip_outputs.pooler_output.unsqueeze(1)
    else:
        mask = clip_encoding["attention_mask"].to(device).unsqueeze(-1).to(dtype=clip_outputs.last_hidden_state.dtype)
        pooled = (clip_outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        vtxt_raw = torch.nn.functional.normalize(pooled, p=2, dim=1).unsqueeze(1)
    return vtxt_raw.detach().cpu(), ctxt_raw.detach().cpu(), ctxt_length.detach().cpu()


def write_shard(
    output_dir: Path,
    shard_id: int,
    rows: list[dict[str, str]],
    vtxt: torch.Tensor,
    ctxt: torch.Tensor,
    ctxt_len: torch.Tensor,
    storage_dtype: str,
) -> dict[str, dict[str, object]]:
    shard_name = f"shard_{shard_id:05d}"
    shard_dir = output_dir / "shards" / shard_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    np_dtype = np.float16 if storage_dtype == "fp16" else np.float32
    np.save(shard_dir / "vtxt.npy", vtxt.float().numpy().astype(np_dtype, copy=False))
    np.save(shard_dir / "ctxt.npy", ctxt.float().numpy().astype(np_dtype, copy=False))
    np.save(shard_dir / "ctxt_len.npy", ctxt_len.numpy().astype(np.int16, copy=False))
    with (shard_dir / "texts.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {row["key"]: {"shard": shard_name, "row": i, "text": row["text"]} for i, row in enumerate(rows)}


def main() -> None:
    args = build_arg_parser().parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    text_root = Path(args.text_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and (output_dir / "index.json").is_file():
        raise FileExistsError(f"Cache already exists, pass --overwrite to rebuild: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    rows = collect_texts(data_root, text_root, splits, bool(args.include_empty), int(args.limit))
    if not rows:
        raise RuntimeError(f"No captions found under {text_root}")

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    model_dtype = dtype_from_name(args.model_dtype, device)
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, padding_side="right", trust_remote_code=True)
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    qwen_model = AutoModelForCausalLM.from_pretrained(
        args.qwen_path,
        low_cpu_mem_usage=True,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    ).eval()
    qwen_model.requires_grad_(False).to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_path)
    clip_model = CLIPTextModel.from_pretrained(args.clip_path, torch_dtype=model_dtype).eval()
    clip_model.requires_grad_(False).to(device)

    crop_start = compute_crop_start(qwen_tokenizer)
    index: dict[str, dict[str, object]] = {}
    shard_rows: list[dict[str, str]] = []
    shard_vtxt: list[torch.Tensor] = []
    shard_ctxt: list[torch.Tensor] = []
    shard_len: list[torch.Tensor] = []
    shard_id = 0
    total = len(rows)
    for start in range(0, total, int(args.batch_size)):
        batch_rows = rows[start : start + int(args.batch_size)]
        texts = [row["text"] for row in batch_rows]
        vtxt, ctxt, ctxt_len = encode_batch(
            texts,
            qwen_model,
            qwen_tokenizer,
            clip_model,
            clip_tokenizer,
            device,
            crop_start,
            int(args.max_length_llm),
        )
        for i, row in enumerate(batch_rows):
            shard_rows.append(row)
            shard_vtxt.append(vtxt[i])
            shard_ctxt.append(ctxt[i])
            shard_len.append(ctxt_len[i])
            if len(shard_rows) >= int(args.shard_size):
                shard_index = write_shard(
                    output_dir,
                    shard_id,
                    shard_rows,
                    torch.stack(shard_vtxt, dim=0),
                    torch.stack(shard_ctxt, dim=0),
                    torch.stack(shard_len, dim=0),
                    args.storage_dtype,
                )
                index.update(shard_index)
                shard_id += 1
                shard_rows, shard_vtxt, shard_ctxt, shard_len = [], [], [], []
        print(f"[cache] encoded {min(start + int(args.batch_size), total)}/{total}", flush=True)

    if shard_rows:
        shard_index = write_shard(
            output_dir,
            shard_id,
            shard_rows,
            torch.stack(shard_vtxt, dim=0),
            torch.stack(shard_ctxt, dim=0),
            torch.stack(shard_len, dim=0),
            args.storage_dtype,
        )
        index.update(shard_index)

    (output_dir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True))
    manifest = {
        "format": "hytext_memmap_v1",
        "data_root": str(data_root),
        "text_root": str(text_root),
        "splits": splits,
        "qwen_path": str(Path(args.qwen_path).expanduser()),
        "clip_path": str(Path(args.clip_path).expanduser()),
        "num_texts": len(rows),
        "max_length_llm": int(args.max_length_llm),
        "crop_start": int(crop_start),
        "ctxt_dim": 4096,
        "vtxt_dim": 768,
        "storage_dtype": args.storage_dtype,
        "prompt_template": PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"[cache] wrote {len(rows)} texts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
