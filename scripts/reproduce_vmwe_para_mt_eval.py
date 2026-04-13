#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reproduce_vmwe_mt_eval import (
    DEFAULT_MT_EVAL_ROOT,
    DEFAULT_MT_ROOT,
    DEFAULT_VMWE_ROOT,
    MetricXScorer,
    XCometScorer,
    available_gpu_count,
    canonical_model_name,
    cuda_device_index,
    clear_cuda_cache,
    load_lingua_detector,
    load_translation_rows,
    migrate_legacy_artifact,
    maybe_set_cuda_device,
    model_uses_gpu_for_translation,
    mkdir,
    norm_text,
    parse_device,
    parse_device_spec,
    parse_device_list,
    rows_match,
    translation_out_path,
    translation_output_complete,
    translator_for,
    validate_pairs,
    write_csv_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAPHRASE_ROOT = ROOT / "MWE" / "Paraphrase"
DEFAULT_MT_PARA_ROOT = ROOT / "MWE" / "MT_para"
DEFAULT_MT_PARA_EVAL_ROOT = ROOT / "MWE" / "MT_para_eval"

CANDIDATE_COLUMNS = {
    "LVC": "LVC_candidate",
    "VPC": "VPC_candidate",
    "VID": "VID_candidate",
}


@dataclass(frozen=True)
class ParaJobSpec:
    stage: str
    dataset: str
    model: Optional[str] = None
    pair: Optional[str] = None


def paraphrase_out_path(paraphrase_root: Path, dataset_name: str) -> Path:
    return paraphrase_root / f"{dataset_name}.csv"


def mt_para_out_path(mt_para_root: Path, model: str, dataset_name: str, pair: str) -> Path:
    return mt_para_root / model / f"{dataset_name}_{pair}.csv"


def mt_para_eval_out_path(mt_para_eval_root: Path, model: str, dataset_name: str, pair: str, view: str) -> Path:
    return mt_para_eval_root / model / f"{dataset_name}_{pair}_{view}.csv"


def build_para_job_specs(stage: str, datasets: Sequence[str], models: Sequence[str], pairs: Sequence[str]) -> List[ParaJobSpec]:
    if stage == "paraphrase":
        return [ParaJobSpec(stage=stage, dataset=dataset) for dataset in datasets]
    return [ParaJobSpec(stage=stage, dataset=dataset, model=model, pair=pair) for model in models for dataset in datasets for pair in pairs]


def load_vmwe_dataset(vmwe_root: Path, dataset_name: str) -> pd.DataFrame:
    if dataset_name not in CANDIDATE_COLUMNS:
        raise ValueError(f"Paraphrase pipeline only supports {sorted(CANDIDATE_COLUMNS)}.")
    path = vmwe_root / f"{dataset_name}.csv"
    df = pd.read_csv(path)
    candidate_column = CANDIDATE_COLUMNS[dataset_name]
    required = {"src", candidate_column}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}.")
    out = df.loc[:, ["src", candidate_column]].copy()
    out["src"] = out["src"].map(norm_text)
    out[candidate_column] = out[candidate_column].map(norm_text)
    out = out[(out["src"].ne("")) & (out[candidate_column].ne(""))]
    return out.drop_duplicates(ignore_index=True)


def paraphrase_output_complete(path: Path, candidate_column: str) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    required = {"src", candidate_column, "para"}
    if not required.issubset(df.columns):
        return False
    if df.empty:
        return True
    src = df["src"].map(norm_text)
    cand = df[candidate_column].map(norm_text)
    para = df["para"].map(norm_text)
    return bool(src.ne("").all() and cand.ne("").all() and para.ne("").all())


def mt_para_output_complete(path: Path, candidate_column: str) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    required = {"src", candidate_column, "para", "mt", "para_mt"}
    if not required.issubset(df.columns):
        return False
    if df.empty:
        return True
    for column in ["src", candidate_column, "para", "mt", "para_mt"]:
        if not df[column].map(norm_text).ne("").all():
            return False
    return True


def build_paraphrase_messages(dataset_name: str, sentence: str, candidate: str) -> List[Dict[str, str]]:
    if dataset_name == "LVC":
        return [
            {"role": "system", "content": """
        You are an expert in linguistics. Given a sentence containing a multi-word expression (MWE), a Light Verb Construct (LVC). Your task is to rephrase the sentence to remove the MWE while keeping the meaning intact. Think in the following steps.
        Step 1 - Look at the phrase in the sentence that will be provided to you that constitutes the MWE in the sentence.
        Step 2 - Replace that phrase with an alternative wording that removes the MWE while maintaining the sentence's meaning.
        Step 3 - Provide your output in the following format:
        Rephrased Sentence: [Sentence without MWE]"""},
            {"role": "user", "content": "Sentence: She gave a smile before walking away. || Phrase: gave a smile"},
            {"role": "assistant", "content": """
            Step 1 - Provided MWE phrase: The given phrase "gave a" is a light-verb construction (LVC), where "gave" (a light verb) combines with "smile" (a noun) to describe the act of smiling.
            Step 2 - Replace the MWE: A more lexicalized way to express this would be "smiled."
            Step 3 - Output Format:
            Rephrased Sentence: She smiled before walking away.
            """},
            {"role": "user", "content": "He took a deep breath before speaking. || Phrase: took a deep breath"},
            {"role": "assistant", "content": """
            Step 1 - Provided MWE phrase: The given phrase "took a deep breath" is a light verb construction (LVC), where "take" (a light verb) combines with "breath" (a noun) to describe the act of inhaling.
            Step 2 - Replace the MWE: A more lexicalized way to express this would be "breathed deeply."
            Step 3 - Output Format:
            Rephrased Sentence: He breathed deeply before speaking.
            """},
            {"role": "user", "content": f"Sentence: {sentence} || Phrase: {candidate}"},
        ]
    if dataset_name == "VID":
        return [
            {"role": "system", "content": """
        You are an expert in linguistics. Given a sentence containing a multi-word expression (MWE), a Verbal Idiom (VID). Your task is to rephrase the sentence to remove the MWE while keeping the meaning intact. Think in the following steps.
        Step 1 - Look at the phrase in the sentence that will be provided to you that constitutes the MWE in the sentence.
        Step 2 - Replace that phrase with an alternative wording that removes the MWE while maintaining the sentence's meaning.
        Step 3 - Provide your output in the following format:
        Rephrased Sentence: [Sentence without MWE]"""},
            {"role": "user", "content": "Sentence: I think we still might take them up on it. || Phrase: take on [pron]"},
            {"role": "assistant", "content": """
            Step 1 - Provided MWE phrase: The given phrase "take them up on" is a verbal idiom (VID), where "take up on" is an idiomatic expression meaning to accept an offer or invitation.
            Step 2 - Replace the MWE: A more literal way to express this would be "accept their offer."
            Step 3 - Output Format:
            Rephrased Sentence: I think we still might accept their offer.
            """},
            {"role": "user", "content": "Sentence: I did not pay much attention to it back then. || Phrase: pay to [pron]"},
            {"role": "assistant", "content": """
            Step 1 - Provided MWE phrase: The given phrase "pay attention to" is a verbal idiom (VID), where "pay" (verb) combines with "attention" to mean focusing or noticing something.
            Step 2 - Replace the MWE: A more literal way to express this would be "notice."
            Step 3 - Output Format:
            Rephrased Sentence: I did not notice it much back then.
            """},
            {"role": "user", "content": f"Sentence: {sentence} || Phrase: {candidate}"},
        ]
    if dataset_name == "VPC":
        return [
            {"role": "system", "content": """
        You are an expert in linguistics. Given a sentence containing a multi-word expression (MWE), a Verb Particle Construct (VPC). Your task is to rephrase the sentence to remove the MWE while keeping the meaning intact. Think in the following steps.
        Step 1 - Look at the phrase in the sentence that will be provided to you that constitutes the MWE in the sentence.
        Step 2 - Replace that phrase with an alternative wording that removes the MWE while maintaining the sentence's meaning.
        Step 3 - Provide your output in the following format:
        Rephrased Sentence: [Sentence without MWE]"""},
            {"role": "user", "content": "Sentence: His eyes welled up during the song. || Phrase: welled up"},
            {"role": "assistant", "content": """
            Step 1 - Provided MWE phrase: The given phrase "welled up" is a verb-particle construction (VPC), where "well" (verb) combines with "up" (particle) to indicate the accumulation or rising of liquid, often referring to tears forming in the eyes.
            Step 2 - Replace the MWE: A more literal way to express this would be "filled with tears."
            Step 3 - Output Format:
            Rephrased Sentence: His eyes filled with tears during the song.
            """},
            {"role": "user", "content": f"Sentence: {sentence} || Phrase: {candidate}"},
        ]
    raise ValueError(f"Unsupported dataset for paraphrasing: {dataset_name}")


class LlamaParaphraser:
    def __init__(self, model_id: str, device: str, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.cuda_devices = [spec for spec in parse_device_spec(device) if spec.startswith("cuda")]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_kwargs = {"trust_remote_code": True}
        if self.cuda_devices:
            model_kwargs["dtype"] = torch.float16
            maybe_set_cuda_device(self.cuda_devices[0])
            if len(self.cuda_devices) == 1:
                self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
                self.model.to(self.cuda_devices[0])
            else:
                max_memory = {gpu_index: "0GiB" for gpu_index in range(available_gpu_count())}
                for device_name in self.cuda_devices:
                    max_memory[cuda_device_index(device_name)] = "46GiB"
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    max_memory=max_memory,
                    **model_kwargs,
                )
            self.input_device = self.torch.device(self.cuda_devices[0])
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            self.input_device = self.torch.device("cpu")
        self.model.eval()

    def paraphrase_one(self, dataset_name: str, sentence: str, candidate: str) -> Optional[str]:
        messages = build_paraphrase_messages(dataset_name, sentence, candidate)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.input_device)
        with self.torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 128009,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
            )
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        match = re.search(r"Rephrased Sentence:\s*(.+)", response)
        if not match:
            return None
        return norm_text(match.group(1))

    def paraphrase_dataframe(self, dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
        candidate_column = CANDIDATE_COLUMNS[dataset_name]
        rows = []
        skipped = 0
        for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Paraphrasing {dataset_name}", unit="row"):
            para = self.paraphrase_one(dataset_name, getattr(row, "src"), getattr(row, candidate_column))
            if not para:
                skipped += 1
                continue
            rows.append({"src": getattr(row, "src"), candidate_column: getattr(row, candidate_column), "para": para})
        print(f"[INFO] {dataset_name}: paraphrase skipped {skipped} rows.")
        return pd.DataFrame(rows, columns=["src", candidate_column, "para"])

    def close(self) -> None:
        for name in ["model", "tokenizer", "torch"]:
            if hasattr(self, name):
                try:
                    delattr(self, name)
                except Exception:
                    pass
        gc.collect()
        clear_cuda_cache()


def build_paraphrase_outputs(
    vmwe_root: Path,
    paraphrase_root: Path,
    datasets: Sequence[str],
    paraphrase_model_id: str,
    paraphrase_device: str,
    paraphrase_max_new_tokens: int,
    overwrite: bool,
) -> None:
    if should_print_runtime_banner():
        print(f"[INFO] Visible GPUs: {available_gpu_count()}")
        print("[INFO] Paraphrase stage only runs on LVC, VPC, and VID. Non_VMWE is intentionally excluded.")
    paraphraser = LlamaParaphraser(paraphrase_model_id, paraphrase_device, paraphrase_max_new_tokens)
    try:
        for dataset_name in datasets:
            candidate_column = CANDIDATE_COLUMNS[dataset_name]
            out_path = paraphrase_out_path(paraphrase_root, dataset_name)
            if paraphrase_output_complete(out_path, candidate_column) and not overwrite:
                print(f"[INFO] Skip existing paraphrase file: {out_path}")
                continue
            src_df = load_vmwe_dataset(vmwe_root, dataset_name)
            para_df = paraphraser.paraphrase_dataframe(dataset_name, src_df)
            write_csv_atomic(para_df, out_path)
            print(f"[INFO] Wrote {out_path}")
    finally:
        paraphraser.close()


def load_mt_para_rows(path: Path, candidate_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["src", candidate_column, "para", "mt", "para_mt"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"{path} must contain {required}.")
    out = df.loc[:, required].copy()
    for column in required:
        out[column] = out[column].map(norm_text)
    out = out[(out["src"].ne("")) & (out[candidate_column].ne("")) & (out["para"].ne("")) & (out["mt"].ne("")) & (out["para_mt"].ne(""))]
    return out.reset_index(drop=True)


def should_print_runtime_banner() -> bool:
    return os.environ.get("VMWE_MT_CHILD_JOB") != "1"


def filter_joint_translations(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    detector, target = load_lingua_detector(pair)
    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Joint filtering {pair}", unit="row"):
        mt = norm_text(getattr(row, "mt"))
        para_mt = norm_text(getattr(row, "para_mt"))
        if detector.detect_language_of(mt) == target and detector.detect_language_of(para_mt) == target:
            rows.append(row._asdict())
    return pd.DataFrame(rows, columns=list(df.columns))


def build_para_translation_outputs(
    vmwe_root: Path,
    paraphrase_root: Path,
    mt_root: Path,
    mt_para_root: Path,
    models: Sequence[str],
    datasets: Sequence[str],
    pairs: Sequence[str],
    translation_device: str,
    enable_filter: bool,
    overwrite: bool,
    google_project_id: Optional[str],
) -> None:
    if should_print_runtime_banner():
        print(f"[INFO] Visible GPUs: {available_gpu_count()}")
        print("[INFO] Translation stage reuses existing MWE/MT outputs when they are already complete.")
    for model_name in models:
        canonical_model = canonical_model_name(model_name)
        translator = translator_for(model_name, translation_device, google_project_id)
        try:
            for dataset_name in datasets:
                candidate_column = CANDIDATE_COLUMNS[dataset_name]
                para_df = pd.read_csv(paraphrase_out_path(paraphrase_root, dataset_name))
                para_df["src"] = para_df["src"].map(norm_text)
                para_df[candidate_column] = para_df[candidate_column].map(norm_text)
                para_df["para"] = para_df["para"].map(norm_text)

                unique_src_df = para_df.loc[:, ["src"]].drop_duplicates(ignore_index=True)
                src_texts = unique_src_df["src"].tolist()
                para_texts = para_df["para"].tolist()

                for pair in pairs:
                    filename = f"{dataset_name}_{pair}.csv"
                    original_out = migrate_legacy_artifact(mt_root, canonical_model, filename)
                    paired_out = mt_para_out_path(mt_para_root, canonical_model, dataset_name, pair)

                    need_original = overwrite or not translation_output_complete(original_out)
                    need_paired = overwrite or not mt_para_output_complete(paired_out, candidate_column)
                    if not need_original and not need_paired:
                        print(f"[INFO] Skip existing paraphrase translation file: {paired_out}")
                        continue

                    if need_original and not need_paired and paired_out.exists():
                        paired_df = load_mt_para_rows(paired_out, candidate_column)
                        original_df = paired_df.loc[:, ["src", "mt"]].drop_duplicates(ignore_index=True)
                        write_csv_atomic(original_df, original_out)
                        print(f"[INFO] Rebuilt {original_out} from existing {paired_out}")
                    elif need_original:
                        original_translations = translator.translate_many(src_texts, pair)
                        original_df = pd.DataFrame({"src": src_texts, "mt": original_translations})
                        if enable_filter:
                            from reproduce_vmwe_mt_eval import filter_translations

                            original_df = filter_translations(original_df, pair)
                        write_csv_atomic(original_df, original_out)
                        print(f"[INFO] Wrote {original_out}")
                    if not need_paired:
                        continue

                    original_df = load_translation_rows(original_out)

                    para_mt = translator.translate_many(para_texts, pair)
                    para_mt_df = para_df.copy()
                    para_mt_df["para_mt"] = [norm_text(text) for text in para_mt]
                    combined = para_mt_df.merge(original_df, on="src", how="inner")
                    combined = combined.loc[:, ["src", candidate_column, "para", "mt", "para_mt"]]
                    if enable_filter:
                        combined = filter_joint_translations(combined, pair)
                    write_csv_atomic(combined, paired_out)
                    print(f"[INFO] Wrote {paired_out}")
        finally:
            translator.close()
            gc.collect()


def merge_original_eval_scores(
    mt_eval_root: Path,
    model_name: str,
    dataset_name: str,
    pair: str,
    paired_df: pd.DataFrame,
) -> pd.DataFrame:
    out = paired_df.copy()
    original_eval_path = migrate_legacy_artifact(
        mt_eval_root,
        canonical_model_name(model_name),
        f"{dataset_name}_{pair}.csv",
    )
    if not original_eval_path.exists():
        return out
    try:
        existing = pd.read_csv(original_eval_path)
    except Exception:
        return out
    if not {"src", "mt"}.issubset(existing.columns):
        return out
    keep_cols = ["src", "mt"]
    if "metricx_score" in existing.columns:
        keep_cols.append("metricx_score")
    if "xcomet_score" in existing.columns:
        keep_cols.append("xcomet_score")
    merged = out.merge(existing.loc[:, keep_cols], on=["src", "mt"], how="left")
    if "metricx_score" in merged.columns:
        merged = merged.rename(columns={"metricx_score": "original_metricx_score"})
    if "xcomet_score" in merged.columns:
        merged = merged.rename(columns={"xcomet_score": "original_xcomet_score"})
    return merged


def para_eval_view_key_columns(candidate_column: str, view: str) -> List[str]:
    if view == "original":
        return ["src", candidate_column, "mt"]
    if view in {"para", "mixed"}:
        return ["src", candidate_column, "para", "para_mt"]
    raise ValueError(f"Unsupported view: {view}")


def load_existing_view_scores(
    path: Path,
    expected_df: pd.DataFrame,
    candidate_column: str,
    view: str,
) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        existing = pd.read_csv(path)
    except Exception:
        return None
    key_columns = para_eval_view_key_columns(candidate_column, view)
    required = set(key_columns) | {"metricx_score", "xcomet_score"}
    if not required.issubset(existing.columns):
        return None
    if not rows_match(existing.loc[:, key_columns], expected_df.loc[:, key_columns], key_columns):
        return None
    return existing


def build_view_output_df(scored_df: pd.DataFrame, candidate_column: str, view: str) -> pd.DataFrame:
    if view == "original":
        out = scored_df.loc[:, ["src", candidate_column, "mt"]].copy()
        out["metricx_score"] = scored_df.get("original_metricx_score", pd.NA)
        out["xcomet_score"] = scored_df.get("original_xcomet_score", pd.NA)
        return out
    if view == "para":
        out = scored_df.loc[:, ["src", candidate_column, "para", "para_mt"]].copy()
        out["metricx_score"] = scored_df.get("paraphrased_metricx_score", pd.NA)
        out["xcomet_score"] = scored_df.get("paraphrased_xcomet_score", pd.NA)
        return out
    if view == "mixed":
        out = scored_df.loc[:, ["src", candidate_column, "para", "para_mt"]].copy()
        out["metricx_score"] = scored_df.get("mixed_metricx_score", pd.NA)
        out["xcomet_score"] = scored_df.get("mixed_xcomet_score", pd.NA)
        return out
    raise ValueError(f"Unsupported view: {view}")


def para_eval_view_complete(view_df: pd.DataFrame) -> bool:
    required = {"metricx_score", "xcomet_score"}
    if not required.issubset(view_df.columns):
        return False
    return bool(view_df["metricx_score"].notna().all() and view_df["xcomet_score"].notna().all())


def build_rows_for_view(df: pd.DataFrame, view: str) -> List[Dict[str, str]]:
    if view == "original":
        return [{"src": row["src"], "mt": row["mt"]} for row in df.to_dict("records")]
    if view == "paraphrased":
        return [{"src": row["para"], "mt": row["para_mt"]} for row in df.to_dict("records")]
    if view == "mixed":
        return [{"src": row["src"], "mt": row["para_mt"]} for row in df.to_dict("records")]
    raise ValueError(f"Unsupported evaluation view: {view}")


def complete_para_metric(existing: pd.DataFrame, base: pd.DataFrame, key_columns: Sequence[str], metric_column: str) -> bool:
    if metric_column not in existing.columns:
        return False
    if not rows_match(existing.loc[:, list(key_columns)], base.loc[:, list(key_columns)], key_columns):
        return False
    return bool(existing[metric_column].notna().all())


def build_para_eval_outputs(
    mt_eval_root: Path,
    mt_para_root: Path,
    mt_para_eval_root: Path,
    models: Sequence[str],
    datasets: Sequence[str],
    pairs: Sequence[str],
    metrics: Sequence[str],
    metricx_device: str,
    xcomet_device: str,
    metricx_batch_size: int,
    xcomet_batch_size: int,
    overwrite: bool,
) -> None:
    if should_print_runtime_banner():
        print(f"[INFO] Visible GPUs: {available_gpu_count()}")
        print("[INFO] Recommended setting: one 48GB GPU is enough for one evaluation model job.")
    metricx_scorer = MetricXScorer(metricx_device, metricx_batch_size) if "metricx" in metrics else None
    xcomet_scorer = XCometScorer(xcomet_device, xcomet_batch_size) if "xcomet" in metrics else None
    try:
        if metricx_scorer is not None:
            print(f"[INFO] MetricX device: {metricx_device}")
        if xcomet_scorer is not None:
            print(f"[INFO] XCOMET device: {xcomet_device}")

        for model_name in models:
            canonical_model = canonical_model_name(model_name)
            for dataset_name in datasets:
                candidate_column = CANDIDATE_COLUMNS[dataset_name]
                for pair in pairs:
                    in_path = mt_para_out_path(mt_para_root, canonical_model, dataset_name, pair)
                    if not in_path.exists():
                        print(f"[WARN] Missing paraphrase translation file, skip evaluation: {in_path}")
                        continue
                    base_df = load_mt_para_rows(in_path, candidate_column)
                    out_df = merge_original_eval_scores(mt_eval_root, canonical_model, dataset_name, pair, base_df)

                    original_view_path = mt_para_eval_out_path(mt_para_eval_root, canonical_model, dataset_name, pair, "original")
                    para_view_path = mt_para_eval_out_path(mt_para_eval_root, canonical_model, dataset_name, pair, "para")
                    mixed_view_path = mt_para_eval_out_path(mt_para_eval_root, canonical_model, dataset_name, pair, "mixed")

                    original_expected = out_df.loc[:, ["src", candidate_column, "mt"]].copy()
                    para_expected = out_df.loc[:, ["src", candidate_column, "para", "para_mt"]].copy()
                    mixed_expected = para_expected.copy()

                    if not overwrite:
                        existing_original = load_existing_view_scores(original_view_path, original_expected, candidate_column, "original")
                        if existing_original is not None:
                            out_df["original_metricx_score"] = existing_original["metricx_score"]
                            out_df["original_xcomet_score"] = existing_original["xcomet_score"]

                        existing_para = load_existing_view_scores(para_view_path, para_expected, candidate_column, "para")
                        if existing_para is not None:
                            out_df["paraphrased_metricx_score"] = existing_para["metricx_score"]
                            out_df["paraphrased_xcomet_score"] = existing_para["xcomet_score"]

                        existing_mixed = load_existing_view_scores(mixed_view_path, mixed_expected, candidate_column, "mixed")
                        if existing_mixed is not None:
                            out_df["mixed_metricx_score"] = existing_mixed["metricx_score"]
                            out_df["mixed_xcomet_score"] = existing_mixed["xcomet_score"]

                    needed_columns = [
                        "original_metricx_score",
                        "original_xcomet_score",
                        "paraphrased_metricx_score",
                        "paraphrased_xcomet_score",
                        "mixed_metricx_score",
                        "mixed_xcomet_score",
                    ]
                    for column in needed_columns:
                        if column not in out_df.columns:
                            out_df[column] = pd.NA

                    if not overwrite:
                        original_complete = para_eval_view_complete(build_view_output_df(out_df, candidate_column, "original"))
                        para_complete = para_eval_view_complete(build_view_output_df(out_df, candidate_column, "para"))
                        mixed_complete = para_eval_view_complete(build_view_output_df(out_df, candidate_column, "mixed"))
                        if original_complete and para_complete and mixed_complete:
                            print(f"[INFO] Skip existing paraphrase eval files: {original_view_path}, {para_view_path}, {mixed_view_path}")
                            continue

                    for view in ["original", "paraphrased", "mixed"]:
                        rows = build_rows_for_view(base_df, view)
                        if not rows:
                            continue
                        need_metricx_scores = "metricx" in metrics and (overwrite or not out_df[f"{view}_metricx_score"].notna().all())
                        need_xcomet_scores = "xcomet" in metrics and (overwrite or not out_df[f"{view}_xcomet_score"].notna().all())

                        if need_metricx_scores and need_xcomet_scores:
                            if metricx_scorer is None:
                                raise RuntimeError("MetricX requested but not initialized.")
                            if xcomet_scorer is None:
                                raise RuntimeError("XCOMET requested but not initialized.")
                            with ThreadPoolExecutor(max_workers=2) as executor:
                                metricx_future = executor.submit(metricx_scorer.score, rows)
                                xcomet_future = executor.submit(xcomet_scorer.score, rows)
                                out_df[f"{view}_metricx_score"] = metricx_future.result()
                                out_df[f"{view}_xcomet_score"] = xcomet_future.result()
                        else:
                            if need_metricx_scores:
                                if metricx_scorer is None:
                                    raise RuntimeError("MetricX requested but not initialized.")
                                out_df[f"{view}_metricx_score"] = metricx_scorer.score(rows)
                            if need_xcomet_scores:
                                if xcomet_scorer is None:
                                    raise RuntimeError("XCOMET requested but not initialized.")
                                out_df[f"{view}_xcomet_score"] = xcomet_scorer.score(rows)

                    original_out = build_view_output_df(out_df, candidate_column, "original")
                    para_out = build_view_output_df(out_df, candidate_column, "para")
                    mixed_out = build_view_output_df(out_df, candidate_column, "mixed")

                    write_csv_atomic(original_out, original_view_path)
                    write_csv_atomic(para_out, para_view_path)
                    write_csv_atomic(mixed_out, mixed_view_path)
                    print(f"[INFO] Wrote {original_view_path}")
                    print(f"[INFO] Wrote {para_view_path}")
                    print(f"[INFO] Wrote {mixed_view_path}")
    finally:
        if metricx_scorer is not None:
            metricx_scorer.close()
        if xcomet_scorer is not None:
            xcomet_scorer.close()


def spawn_para_job_process(
    python_exe: str,
    script_path: str,
    job: ParaJobSpec,
    args: argparse.Namespace,
    device_assignment: Tuple[Optional[str], Optional[str], Optional[str]],
) -> subprocess.Popen:
    primary_device, metricx_device, xcomet_device = device_assignment
    command = [
        python_exe,
        script_path,
        "--stage",
        job.stage,
        "--datasets",
        job.dataset,
        "--vmwe-root",
        str(args.vmwe_root),
        "--paraphrase-root",
        str(args.paraphrase_root),
        "--mt-root",
        str(args.mt_root),
        "--mt-eval-root",
        str(args.mt_eval_root),
        "--mt-para-root",
        str(args.mt_para_root),
        "--mt-para-eval-root",
        str(args.mt_para_eval_root),
        "--parallel-jobs",
        "1",
    ]

    if job.stage in {"translate", "evaluate"} and job.model:
        command.extend(["--models", job.model])
    if job.stage in {"translate", "evaluate"} and job.pair:
        command.extend(["--pairs", job.pair])
    if job.stage == "evaluate" and args.metrics:
        command.extend(["--metrics", *args.metrics])

    command.extend(["--paraphrase-model-id", args.paraphrase_model_id])
    command.extend(["--paraphrase-max-new-tokens", str(args.paraphrase_max_new_tokens)])
    command.extend(["--metricx-batch-size", str(args.metricx_batch_size)])
    command.extend(["--xcomet-batch-size", str(args.xcomet_batch_size)])

    if job.stage == "paraphrase":
        if primary_device is not None:
            command.extend(["--paraphrase-device", primary_device])
    elif job.stage == "translate":
        if primary_device is not None:
            command.extend(["--translation-device", primary_device])
    elif job.stage == "evaluate":
        if metricx_device is not None:
            command.extend(["--metricx-device", metricx_device])
        if xcomet_device is not None:
            command.extend(["--xcomet-device", xcomet_device])

    if args.google_project_id:
        command.extend(["--google-project-id", args.google_project_id])
    if args.no_filter:
        command.append("--no-filter")
    if args.overwrite:
        command.append("--overwrite")

    print(f"[INFO] Launching job on devices {device_assignment}: {job.stage} {job.dataset} {job.model or ''} {job.pair or ''}".strip())
    env = os.environ.copy()
    env["VMWE_MT_CHILD_JOB"] = "1"
    return subprocess.Popen(command, cwd=str(ROOT), env=env)


def run_parallel_jobs(
    stage: str,
    jobs: Sequence[ParaJobSpec],
    args: argparse.Namespace,
    paraphrase_devices: Sequence[str],
    translation_devices: Sequence[str],
    metricx_devices: Sequence[str],
    xcomet_devices: Sequence[str],
) -> None:
    if not jobs:
        return

    python_exe = sys.executable
    script_path = str(Path(__file__).resolve())

    if stage == "paraphrase":
        device_assignments = [(device, None, None) for device in paraphrase_devices]
        max_workers = max(1, min(len(device_assignments), len(jobs)))
    elif stage == "translate":
        if any(job.model and model_uses_gpu_for_translation(job.model) for job in jobs):
            device_assignments = [(device, None, None) for device in translation_devices]
            max_workers = max(1, min(len(device_assignments), len(jobs)))
        else:
            if args.translation_gpus:
                device_assignments = [(device, None, None) for device in translation_devices]
                max_workers = max(1, min(len(device_assignments), len(jobs)))
            else:
                device_assignments = [(None, None, None) for _ in range(max(1, min(args.parallel_jobs, len(jobs))))]
                max_workers = len(device_assignments)
    else:
        if "metricx" in args.metrics and "xcomet" in args.metrics:
            limit = min(len(metricx_devices), len(xcomet_devices))
            device_assignments = [(None, metricx_devices[i], xcomet_devices[i]) for i in range(limit)]
        elif "metricx" in args.metrics:
            device_assignments = [(None, device, None) for device in metricx_devices]
        elif "xcomet" in args.metrics:
            device_assignments = [(None, None, device) for device in xcomet_devices]
        else:
            device_assignments = [(None, None, None)]
        max_workers = max(1, min(args.parallel_jobs, len(device_assignments), len(jobs)))

    active: List[Tuple[subprocess.Popen, ParaJobSpec, Tuple[Optional[str], Optional[str], Optional[str]]]] = []
    next_job = 0

    def stop_active_processes(
        processes: Sequence[Tuple[subprocess.Popen, ParaJobSpec, Tuple[Optional[str], Optional[str], Optional[str]]]]
    ) -> None:
        for proc, _, _ in processes:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.time() + 5.0
        for proc, _, _ in processes:
            remaining = max(0.0, deadline - time.time())
            if proc.poll() is None:
                try:
                    proc.wait(timeout=remaining or 0.1)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for proc, _, _ in processes:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
    print(f"[INFO] Scheduling {len(jobs)} {stage} jobs with {max_workers} worker(s).")

    for idx in range(max_workers):
        assignment = device_assignments[idx]
        job = jobs[next_job]
        next_job += 1
        proc = spawn_para_job_process(python_exe, script_path, job, args, assignment)
        active.append((proc, job, assignment))

    try:
        while active:
            still_running: List[Tuple[subprocess.Popen, ParaJobSpec, Tuple[Optional[str], Optional[str], Optional[str]]]] = []
            for proc, job, assignment in active:
                code = proc.poll()
                if code is None:
                    still_running.append((proc, job, assignment))
                    continue
                if code != 0:
                    stop_active_processes(still_running)
                    raise RuntimeError(
                        f"Parallel job failed for {job.stage} {job.dataset} {job.model or ''} {job.pair or ''} with exit code {code}"
                    )
                print(f"[INFO] Finished job: {job.stage} {job.dataset} {job.model or ''} {job.pair or ''}".strip())
                if next_job < len(jobs):
                    next_assignment = assignment
                    next_spec = jobs[next_job]
                    next_job += 1
                    next_proc = spawn_para_job_process(python_exe, script_path, next_spec, args, next_assignment)
                    still_running.append((next_proc, next_spec, next_assignment))
            active = still_running
    except BaseException:
        stop_active_processes(active)
        raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce VMWE paraphrase -> MT -> QE outputs for LVC, VPC, and VID.",
        epilog=(
            "Notes:\n"
            "1. Non_VMWE is intentionally excluded from paraphrasing.\n"
            "2. Existing MWE/MT translation files are reused when complete, so the script only translates missing original MT outputs.\n"
            "3. Joint language filtering keeps a row only when both the original translation and the paraphrased translation are in the target language.\n"
            "4. MWE/MT_para_eval stores three evaluation views: original (src->mt), paraphrased (para->para_mt), and mixed (src->para_mt).\n"
            "5. One 48GB GPU is typically enough for one paraphrase model job, one translation job, or one evaluation model job.\n"
            "6. If multiple paraphrase or translation GPUs are provided, those stages run in parallel automatically with one job per free GPU.\n"
            "7. This code is currently designed for the 7 built-in directions: en-cs, en-de, en-es, en-ja, en-ru, en-tr, en-zh."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--stage", choices=["paraphrase", "translate", "evaluate", "all"], default="all")
    parser.add_argument("--vmwe-root", type=Path, default=DEFAULT_VMWE_ROOT)
    parser.add_argument("--paraphrase-root", type=Path, default=DEFAULT_PARAPHRASE_ROOT)
    parser.add_argument("--mt-root", type=Path, default=DEFAULT_MT_ROOT)
    parser.add_argument("--mt-eval-root", type=Path, default=DEFAULT_MT_EVAL_ROOT)
    parser.add_argument("--mt-para-root", type=Path, default=DEFAULT_MT_PARA_ROOT)
    parser.add_argument("--mt-para-eval-root", type=Path, default=DEFAULT_MT_PARA_EVAL_ROOT)
    parser.add_argument("--datasets", nargs="+", default=["LVC", "VPC", "VID"], choices=["LVC", "VPC", "VID"])
    parser.add_argument("--models", nargs="+", default=["GemmaX2"], help="Translation models: Google GemmaX2 LLaMAX phi4 Madlad M2M100 opus seamless")
    parser.add_argument("--pairs", nargs="+", default=["en-cs"], help="Target pairs like en-cs en-de en-es en-tr en-ja en-ru en-zh")
    parser.add_argument("--metrics", nargs="+", default=["metricx", "xcomet"], choices=["metricx", "xcomet"])
    parser.add_argument("--paraphrase-model-id", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--paraphrase-max-new-tokens", type=int, default=256, help="Maximum generated tokens for one paraphrase response.")
    parser.add_argument("--paraphrase-device", default="auto", help="Device for the paraphrase model, e.g. cuda:0 or cpu")
    parser.add_argument("--translation-device", default="auto", help="Device for translation, e.g. cuda:1 or cpu")
    parser.add_argument("--metricx-device", default="auto", help="Device for MetricX evaluation")
    parser.add_argument("--xcomet-device", default="auto", help="Device for XCOMET evaluation")
    parser.add_argument("--paraphrase-gpus", nargs="*", default=None, help="Pool for paraphrase jobs, e.g. cuda:0 cuda:1. If multiple GPUs are provided, paraphrasing runs in parallel automatically.")
    parser.add_argument("--translation-gpus", nargs="*", default=None, help="Pool for non-Google MT jobs, e.g. cuda:0 cuda:1. If multiple GPUs are provided, translation runs in parallel automatically.")
    parser.add_argument("--metricx-gpus", nargs="*", default=None, help="Optional pool for parallel MetricX jobs, e.g. cuda:2 cuda:3")
    parser.add_argument("--xcomet-gpus", nargs="*", default=None, help="Optional pool for parallel XCOMET jobs, e.g. cuda:4 cuda:5")
    parser.add_argument("--parallel-jobs", type=int, default=1, help="Maximum number of parallel jobs for Google-only translation and for evaluation.")
    parser.add_argument("--metricx-batch-size", type=int, default=1)
    parser.add_argument("--xcomet-batch-size", type=int, default=1)
    parser.add_argument("--google-project-id", default=None)
    parser.add_argument("--no-filter", action="store_true", help="Disable language filtering.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validate_pairs(args.pairs)
    paraphrase_device = parse_device(args.paraphrase_device)
    translation_device = parse_device(args.translation_device)
    metricx_device = parse_device(args.metricx_device)
    xcomet_device = parse_device(args.xcomet_device)
    paraphrase_devices = parse_device_list(args.paraphrase_gpus, args.paraphrase_device)
    translation_devices = parse_device_list(args.translation_gpus, args.translation_device)
    metricx_devices = parse_device_list(args.metricx_gpus, args.metricx_device)
    xcomet_devices = parse_device_list(args.xcomet_gpus, args.xcomet_device)

    paraphrase_parallel = args.stage in {"paraphrase", "all"} and args.paraphrase_gpus is not None and len(paraphrase_devices) > 1
    translation_parallel = False
    if args.stage in {"translate", "all"}:
        translation_parallel = (
            (any(model_uses_gpu_for_translation(model) for model in args.models) and args.translation_gpus is not None and len(translation_devices) > 1)
            or (not any(model_uses_gpu_for_translation(model) for model in args.models) and args.parallel_jobs > 1)
        )
    evaluation_parallel = args.stage in {"evaluate", "all"} and args.parallel_jobs > 1

    if paraphrase_parallel:
        paraphrase_jobs = build_para_job_specs("paraphrase", args.datasets, args.models, args.pairs)
        run_parallel_jobs("paraphrase", paraphrase_jobs, args, paraphrase_devices, translation_devices, metricx_devices, xcomet_devices)
    elif args.stage in {"paraphrase", "all"}:
        build_paraphrase_outputs(
            vmwe_root=args.vmwe_root,
            paraphrase_root=args.paraphrase_root,
            datasets=args.datasets,
            paraphrase_model_id=args.paraphrase_model_id,
            paraphrase_device=paraphrase_device,
            paraphrase_max_new_tokens=args.paraphrase_max_new_tokens,
            overwrite=args.overwrite,
        )

    if translation_parallel:
        translate_jobs = build_para_job_specs("translate", args.datasets, args.models, args.pairs)
        run_parallel_jobs("translate", translate_jobs, args, paraphrase_devices, translation_devices, metricx_devices, xcomet_devices)
    elif args.stage in {"translate", "all"}:
        build_para_translation_outputs(
            vmwe_root=args.vmwe_root,
            paraphrase_root=args.paraphrase_root,
            mt_root=args.mt_root,
            mt_para_root=args.mt_para_root,
            models=args.models,
            datasets=args.datasets,
            pairs=args.pairs,
            translation_device=translation_device,
            enable_filter=not args.no_filter,
            overwrite=args.overwrite,
            google_project_id=args.google_project_id,
        )

    if evaluation_parallel:
        eval_jobs = build_para_job_specs("evaluate", args.datasets, args.models, args.pairs)
        run_parallel_jobs("evaluate", eval_jobs, args, paraphrase_devices, translation_devices, metricx_devices, xcomet_devices)
    elif args.stage in {"evaluate", "all"}:
        build_para_eval_outputs(
            mt_eval_root=args.mt_eval_root,
            mt_para_root=args.mt_para_root,
            mt_para_eval_root=args.mt_para_eval_root,
            models=args.models,
            datasets=args.datasets,
            pairs=args.pairs,
            metrics=args.metrics,
            metricx_device=metricx_device,
            xcomet_device=xcomet_device,
            metricx_batch_size=args.metricx_batch_size,
            xcomet_batch_size=args.xcomet_batch_size,
            overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
