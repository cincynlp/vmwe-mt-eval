#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import os
import re
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VMWE_ROOT = ROOT / "datasets" / "VMWE"
DEFAULT_MT_ROOT = ROOT / "MWE" / "MT"
DEFAULT_MT_EVAL_ROOT = ROOT / "MWE" / "MT_eval"

LANGUAGE_DISPLAY = {
    "en-cs": "Czech",
    "en-de": "German",
    "en-es": "Spanish",
    "en-ja": "Japanese",
    "en-ru": "Russian",
    "en-tr": "Turkish",
    "en-zh": "Chinese",
}

LANGUAGE_SHORT = {
    "en-cs": "cs",
    "en-de": "de",
    "en-es": "es",
    "en-ja": "ja",
    "en-ru": "ru",
    "en-tr": "tr",
    "en-zh": "zh",
}

LINGUA_LANGUAGE = {
    "en-cs": "CZECH",
    "en-de": "GERMAN",
    "en-es": "SPANISH",
    "en-ja": "JAPANESE",
    "en-ru": "RUSSIAN",
    "en-tr": "TURKISH",
    "en-zh": "CHINESE",
}

SEAMLESS_LANG = {
    "en-cs": "ces",
    "en-de": "deu",
    "en-es": "spa",
    "en-ja": "jpn",
    "en-ru": "rus",
    "en-tr": "tur",
    "en-zh": "cmn",
}

OPUS_MODEL = {
    "en-cs": "Helsinki-NLP/opus-mt-en-cs",
    "en-de": "Helsinki-NLP/opus-mt-en-de",
    "en-es": "Helsinki-NLP/opus-mt-en-es",
    "en-ja": "Helsinki-NLP/opus-mt-en-jap",
    "en-ru": "Helsinki-NLP/opus-mt-en-ru",
    "en-tr": "Helsinki-NLP/opus-mt-en-trk",
    "en-zh": "Helsinki-NLP/opus-mt-en-zh",
}

DATASET_FILES = {
    "LVC": "LVC.csv",
    "VPC": "VPC.csv",
    "VID": "VID.csv",
    "Non_VMWE": "Non_VMWE.csv",
}

SUPPORTED_PAIRS = tuple(LANGUAGE_DISPLAY.keys())

LEGACY_MODEL_DIRS = {
    "Google": ("google",),
    "M2M100": ("M2M",),
}


@dataclass(frozen=True)
class JobSpec:
    stage: str
    model: str
    dataset: str
    pair: str


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def norm_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def ensure_python_package(import_name: str, pip_name: Optional[str] = None):
    try:
        return __import__(import_name)
    except ImportError:
        package_name = pip_name or import_name
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return __import__(import_name)


def parse_device(device: str) -> str:
    device = device.strip()
    if device == "auto":
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def parse_device_spec(device_spec: str) -> List[str]:
    raw_parts = [str(part).strip() for part in str(device_spec).split(",") if str(part).strip()]
    if not raw_parts:
        return []
    expanded: List[str] = []
    inferred_prefix: Optional[str] = None
    for part in raw_parts:
        if ":" in part:
            parsed = parse_device(part)
            expanded.append(parsed)
            if parsed.startswith("cuda:"):
                inferred_prefix = "cuda"
            elif parsed.startswith("cpu"):
                inferred_prefix = "cpu"
            else:
                inferred_prefix = None
            continue
        if inferred_prefix == "cuda" and part.isdigit():
            expanded.append(f"cuda:{part}")
            continue
        expanded.append(parse_device(part))
    return expanded


def parse_device_list(values: Optional[Sequence[str]], fallback: str) -> List[str]:
    if values:
        devices = [parse_device(value) for value in values if str(value).strip()]
        if devices:
            return devices
    return [parse_device(fallback)]


def validate_pairs(pairs: Sequence[str]) -> None:
    unsupported = [pair for pair in pairs if pair not in LANGUAGE_DISPLAY]
    if unsupported:
        supported = ", ".join(SUPPORTED_PAIRS)
        invalid = ", ".join(unsupported)
        raise ValueError(
            f"Unsupported language pair(s): {invalid}. "
            f"This code is currently designed for these 7 directions: {supported}. "
            "If your model supports more directions, add the corresponding mappings yourself and rerun."
        )


def model_uses_gpu_for_translation(model_name: str) -> bool:
    return canonical_model_name(model_name) != "Google"


def any_translation_job_uses_gpu(models: Sequence[str]) -> bool:
    return any(model_uses_gpu_for_translation(model) for model in models)


def maybe_set_cuda_device(device: str) -> None:
    if not device.startswith("cuda"):
        return
    try:
        import torch
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        torch.cuda.set_device(index)
    except Exception:
        pass


def cuda_device_index(device: str) -> int:
    if not device.startswith("cuda"):
        raise ValueError(f"Not a CUDA device string: {device}")
    return int(device.split(":", 1)[1]) if ":" in device else 0


def clear_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def available_gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def should_print_runtime_banner() -> bool:
    return os.environ.get("VMWE_MT_CHILD_JOB") != "1"


def dataset_path(vmwe_root: Path, dataset_name: str) -> Path:
    explicit = vmwe_root / DATASET_FILES.get(dataset_name, f"{dataset_name}.csv")
    if explicit.exists():
        return explicit
    if dataset_name == "Non_VMWE":
        fallback = vmwe_root / "Normal.csv"
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Could not find dataset CSV for {dataset_name} under {vmwe_root}")


def load_dataset_rows(vmwe_root: Path, dataset_name: str) -> pd.DataFrame:
    path = dataset_path(vmwe_root, dataset_name)
    df = pd.read_csv(path)
    if "src" not in df.columns:
        raise ValueError(f"{path} must contain a 'src' column.")
    out = df.loc[:, ["src"]].copy()
    out["src"] = out["src"].map(norm_text)
    out = out[out["src"].ne("")]
    return out.drop_duplicates(ignore_index=True)


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    mkdir(path.parent)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)


def load_translation_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"src", "mt"}.issubset(df.columns):
        raise ValueError(f"{path} must contain src and mt columns.")
    out = df.loc[:, ["src", "mt"]].copy()
    out["src"] = out["src"].map(norm_text)
    out["mt"] = out["mt"].map(norm_text)
    out = out[(out["src"].ne("")) & (out["mt"].ne(""))]
    return out.drop_duplicates(ignore_index=True)


def translation_output_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if not {"src", "mt"}.issubset(df.columns):
        return False
    if df.empty:
        return True
    src = df["src"].map(norm_text)
    mt = df["mt"].map(norm_text)
    return bool(src.ne("").all() and mt.ne("").all())


def rows_match(left: pd.DataFrame, right: pd.DataFrame, key_columns: Sequence[str]) -> bool:
    if len(left) != len(right):
        return False
    for column in key_columns:
        if column not in left.columns or column not in right.columns:
            return False
        left_values = left[column].fillna("").map(norm_text).tolist()
        right_values = right[column].fillna("").map(norm_text).tolist()
        if left_values != right_values:
            return False
    return True


def eval_metric_complete(path: Path, base_rows: pd.DataFrame, metric_column: str) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    required = {"src", "mt", metric_column}
    if not required.issubset(df.columns):
        return False
    compare = df.loc[:, ["src", "mt", metric_column]].copy()
    if not rows_match(compare.loc[:, ["src", "mt"]], base_rows.loc[:, ["src", "mt"]], ["src", "mt"]):
        return False
    return bool(compare[metric_column].notna().all())


def translation_out_path(mt_root: Path, model: str, dataset_name: str, pair: str) -> Path:
    return mt_root / model / f"{dataset_name}_{pair}.csv"


def eval_out_path(mt_eval_root: Path, model: str, dataset_name: str, pair: str) -> Path:
    return mt_eval_root / model / f"{dataset_name}_{pair}.csv"


def migrate_legacy_artifact(root: Path, model_name: str, filename: str) -> Path:
    canonical = canonical_model_name(model_name)
    canonical_path = root / canonical / filename
    if canonical_path.exists():
        return canonical_path

    for legacy_dir in LEGACY_MODEL_DIRS.get(canonical, ()):
        legacy_path = root / legacy_dir / filename
        if legacy_path.exists():
            mkdir(canonical_path.parent)
            legacy_path.replace(canonical_path)
            return canonical_path
    return canonical_path


class BaseTranslator:
    def __init__(self, device: str):
        self.device = device

    def translate_many(self, texts: Sequence[str], pair: str) -> List[str]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def _release_torch_state(self, *attr_names: str) -> None:
        for name in attr_names:
            if hasattr(self, name):
                try:
                    delattr(self, name)
                except Exception:
                    pass
        gc.collect()
        clear_cuda_cache()


class GoogleTranslator(BaseTranslator):
    def __init__(self, device: str, project_id: Optional[str]):
        super().__init__(device)
        from google.cloud import translate_v3

        self.client = translate_v3.TranslationServiceClient()
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise RuntimeError("Google translation requires --google-project-id or GOOGLE_CLOUD_PROJECT.")
        self.parent = f"projects/{self.project_id}/locations/global"

    def translate_one(self, text: str, pair: str) -> str:
        response = self.client.translate_text(
            contents=[text],
            target_language_code=LANGUAGE_SHORT[pair],
            parent=self.parent,
            mime_type="text/plain",
            source_language_code="en",
        )
        return response.translations[0].translated_text if response.translations else ""

    def translate_many(self, texts: Sequence[str], pair: str) -> List[str]:
        return [self.translate_one(text, pair) for text in tqdm(texts, desc=f"Google {pair}", unit="sent")]


class GemmaX2Translator(BaseTranslator):
    model_name = "ModelSpace/GemmaX2-28-9B-v0.1"

    def __init__(self, device: str):
        super().__init__(device)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(device)
        self.model.eval()

    def translate_one(self, text: str, pair: str) -> str:
        prompt = (
            f"Translate this from English to {LANGUAGE_DISPLAY[pair]}:\n"
            f"English: {text}\n"
            "Translation:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        with self.torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.split("Translation:")[-1].strip()

    def translate_many(self, texts: Sequence[str], pair: str) -> List[str]:
        return [self.translate_one(text, pair) for text in tqdm(texts, desc=f"GemmaX2 {pair}", unit="sent")]

    def close(self) -> None:
        self._release_torch_state("model", "tokenizer", "torch")


class LLaMAXTranslator(BaseTranslator):
    model_name = "LLaMAX/LLaMAX3-8B-Alpaca"

    def __init__(self, device: str):
        super().__init__(device)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(device)
        self.model.eval()

    def translate_one(self, text: str, pair: str) -> str:
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n"
            f"### Instruction:\nTranslate the following sentences from English to {LANGUAGE_DISPLAY[pair]}.\n"
            f"### Input:\n{text}\n### Response:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        generated = self.model.generate(inputs.input_ids, pad_token_id=128009, max_length=1000)
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return decoded.split("Response:")[-1].strip()

    def translate_many(self, texts: Sequence[str], pair: str) -> List[str]:
        return [self.translate_one(text, pair) for text in tqdm(texts, desc=f"LLaMAX {pair}", unit="sent")]

    def close(self) -> None:
        self._release_torch_state("model", "tokenizer", "torch")


class Phi4Translator(BaseTranslator):
    model_name = "microsoft/Phi-4-multimodal-instruct"

    def __init__(self, device: str):
        super().__init__(device)
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        self.torch = torch
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*image_processor_class argument is deprecated.*", category=FutureWarning)
            warnings.filterwarnings("ignore", message=r".*Using a slow image processor as `use_fast` is unset.*")
            warnings.filterwarnings("ignore", message=r".*Please specify CheckpointImpl\.NO_REENTRANT.*", category=FutureWarning)
            warnings.filterwarnings("ignore", message=r".*Flash Attention 2\.0 with a model not initialized on GPU.*")
            warnings.filterwarnings("ignore", message=r".*`torch_dtype` is deprecated! Use `dtype` instead!.*")
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if device.startswith("cuda") else "auto",
            )
        self.model.to(device)
        self.model.eval()
        self.generation_config = GenerationConfig.from_pretrained(self.model_name, filename="generation_config.json")
        if not hasattr(self.generation_config, "num_logits_to_keep"):
            self.generation_config.num_logits_to_keep = 1

    def translate_one(self, text: str, pair: str) -> str:
        target = LANGUAGE_DISPLAY[pair]
        prompt = (
            f"<|system|>You are an expert translator skilled in rendering English text into fluent, natural {target} "
            f"while preserving the original meaning, tone, and context.<|end|>"
            f"<|user|>Please accurately translate the following English sentence into {target}:\n\n"
            f"\"{text}\"\n<|end|><|assistant|>"
        )
        inputs = self.processor(text=prompt, images=None, return_tensors="pt").to(self.model.device)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*Please specify CheckpointImpl\.NO_REENTRANT.*", category=FutureWarning)
            generated = self.model.generate(**inputs, max_new_tokens=200, generation_config=self.generation_config)
        input_len = inputs["input_ids"].shape[1]
        generated = generated[:, input_len:]
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return decoded.split("<|assistant|>")[-1].strip()

    def translate_many(self, texts: Sequence[str], pair: str) -> List[str]:
        return [self.translate_one(text, pair) for text in tqdm(texts, desc=f"phi4 {pair}", unit="sent")]

    def close(self) -> None:
        self._release_torch_state("model", "processor", "generation_config", "torch")


class MadladTranslator(BaseTranslator):
    model_name = "google/madlad400-10b-mt"

    def __init__(self, device: str):
        super().__init__(device)
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.torch = torch
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch.float16 if device.startswith("cuda") else None)
        self.model.to(device)
        self.model.eval()

    def translate_many(self, texts: Sequence[str], pair: str) -> List[str]:
        prefix = f"<2{LANGUAGE_SHORT[pair]}> "
        prefixed = [prefix + text for text in texts]
        outputs: List[str] = []
        for text in tqdm(prefixed, desc=f"Madlad {pair}", unit="sent"):
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.model.device)
            with self.torch.no_grad():
                generated = self.model.generate(input_ids=inputs, max_length=512)
            outputs.append(self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
        return outputs

    def close(self) -> None:
        self._release_torch_state("model", "tokenizer", "torch")


class M2M100Translator(BaseTranslator):
    model_name = "facebook/m2m100_1.2B"

    def __init__(self, device: str):
        super().__init__(device)
        import torch
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        self.torch = torch
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(device)
        self.model.eval()

    def translate_one(self, text: str, pair: str) -> str:
        target = LANGUAGE_SHORT[pair]
        self.tokenizer.src_lang = "en"
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        generated = self.model.generate(**encoded, forced_bos_token_id=self.tokenizer.get_lang_id(target), max_new_tokens=512)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    def translate_many(self, texts: Sequence[str], pair: str) -> List[str]:
        return [self.translate_one(text, pair) for text in tqdm(texts, desc=f"M2M100 {pair}", unit="sent")]

    def close(self) -> None:
        self._release_torch_state("model", "tokenizer", "torch")


class OpusTranslator(BaseTranslator):
    def __init__(self, device: str):
        super().__init__(device)
        self.loaded_pair: Optional[str] = None
        self.tokenizer = None
        self.model = None
        import torch
        self.torch = torch

    def _ensure_pair(self, pair: str) -> None:
        if self.loaded_pair == pair:
            return
        from transformers import MarianMTModel, MarianTokenizer

        model_name = OPUS_MODEL[pair]
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.loaded_pair = pair

    def translate_many(self, texts: Sequence[str], pair: str) -> List[str]:
        self._ensure_pair(pair)
        outputs: List[str] = []
        for text in tqdm(texts, desc=f"Opus {pair}", unit="sent"):
            inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            with self.torch.no_grad():
                generated = self.model.generate(**inputs)
            outputs.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return outputs

    def close(self) -> None:
        self._release_torch_state("model", "tokenizer", "torch")
        self.loaded_pair = None


class SeamlessTranslator(BaseTranslator):
    model_name = "facebook/seamless-m4t-v2-large"

    def __init__(self, device: str):
        super().__init__(device)
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = SeamlessM4Tv2ForTextToText.from_pretrained(self.model_name)
        self.model.to(device)
        self.model.eval()

    def translate_many(self, texts: Sequence[str], pair: str) -> List[str]:
        tgt_lang = SEAMLESS_LANG[pair]
        outputs: List[str] = []
        for text in tqdm(texts, desc=f"Seamless {pair}", unit="sent"):
            inputs = self.processor(text=[text], src_lang="eng", tgt_lang=tgt_lang, return_tensors="pt", padding=True).to(self.device)
            with self.torch.no_grad():
                generated = self.model.generate(**inputs, tgt_lang=tgt_lang)
            outputs.extend(self.processor.batch_decode(generated, skip_special_tokens=True))
        return outputs

    def close(self) -> None:
        self._release_torch_state("model", "processor", "torch")


TRANSLATOR_REGISTRY = {
    "google": GoogleTranslator,
    "gemmax2": GemmaX2Translator,
    "llamax": LLaMAXTranslator,
    "phi4": Phi4Translator,
    "madlad": MadladTranslator,
    "m2m100": M2M100Translator,
    "m2m": M2M100Translator,
    "opus": OpusTranslator,
    "seamless": SeamlessTranslator,
}


def canonical_model_name(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "", name.lower())
    aliases = {
        "gemmax2": "GemmaX2",
        "google": "Google",
        "llamax": "LLaMAX",
        "phi4": "phi4",
        "madlad": "Madlad",
        "m2m100": "M2M100",
        "m2m": "M2M100",
        "opus": "opus",
        "seamless": "seamless",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported model: {name}")
    return aliases[key]


def translator_for(model_name: str, device: str, google_project_id: Optional[str]) -> BaseTranslator:
    key = re.sub(r"[^a-z0-9]+", "", model_name.lower())
    cls = TRANSLATOR_REGISTRY[key]
    if cls is GoogleTranslator:
        return cls(device=device, project_id=google_project_id)
    return cls(device=device)


def load_lingua_detector(pair: str):
    lingua = ensure_python_package("lingua", "lingua-language-detector")
    supported = [getattr(lingua.Language, name) for name in sorted(set(LINGUA_LANGUAGE.values()))]
    detector = lingua.LanguageDetectorBuilder.from_languages(*supported).build()
    target = getattr(lingua.Language, LINGUA_LANGUAGE[pair])
    return detector, target


def filter_translations(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    detector, target = load_lingua_detector(pair)
    keep_rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Filtering {pair}", unit="row"):
        mt = norm_text(getattr(row, "mt", ""))
        detected = detector.detect_language_of(mt)
        if detected == target:
            keep_rows.append({"src": getattr(row, "src"), "mt": mt})
    return pd.DataFrame(keep_rows, columns=["src", "mt"])


class MetricXScorer:
    def __init__(self, device: str, batch_size: int):
        import datasets
        import torch
        import transformers
        from transformers import DataCollatorWithPadding

        self.datasets = datasets
        self.torch = torch
        self.transformers = transformers
        self.DataCollatorWithPadding = DataCollatorWithPadding
        self.MT5ForRegression = build_metricx_model_classes()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xxl")
        self.device = device
        self.cuda_devices = [spec for spec in parse_device_spec(device) if spec.startswith("cuda")]
        if self.cuda_devices:
            maybe_set_cuda_device(self.cuda_devices[0])
            if len(self.cuda_devices) == 1:
                device_index = cuda_device_index(self.cuda_devices[0])
                self.model = self.MT5ForRegression.from_pretrained(
                    "google/metricx-24-hybrid-xxl-v2p6",
                    dtype="auto",
                    device_map={"": device_index},
                )
            else:
                max_memory = {gpu_index: "46GiB" for gpu_index in range(available_gpu_count())}
                for device_name in self.cuda_devices:
                    max_memory[cuda_device_index(device_name)] = "46GiB"
                for gpu_index in range(available_gpu_count()):
                    if f"cuda:{gpu_index}" not in self.cuda_devices:
                        max_memory[gpu_index] = "0GiB"
                self.model = self.MT5ForRegression.from_pretrained(
                    "google/metricx-24-hybrid-xxl-v2p6",
                    dtype="auto",
                    device_map="auto",
                    max_memory=max_memory,
                )
        else:
            self.model = self.MT5ForRegression.from_pretrained("google/metricx-24-hybrid-xxl-v2p6")
        self.model.eval()
        self.batch_size = batch_size
        self.max_input_length = 1536

        class CustomTrainer(transformers.Trainer):
            def _move_model_to_device(self, model, device):
                return model

        self.CustomTrainer = CustomTrainer

    def score(self, rows: List[Dict[str, str]]) -> List[float]:
        def _make_input(example):
            example["input"] = "source: " + example["src"] + " candidate: " + example["mt"]
            return example

        def _tokenize(example):
            return self.tokenizer(example["input"], max_length=self.max_input_length, truncation=True, padding=False)

        ds = self.datasets.Dataset.from_list(rows)
        ds = ds.map(_make_input)
        ds = ds.map(_tokenize)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask"], device="cpu")

        training_args = self.transformers.TrainingArguments(
            output_dir=str(ROOT / ".metricx_tmp"),
            per_device_eval_batch_size=self.batch_size,
            dataloader_pin_memory=False,
            report_to=[],
        )
        trainer = self.CustomTrainer(
            model=self.model,
            args=training_args,
            data_collator=self.DataCollatorWithPadding(tokenizer=self.tokenizer),
        )
        predictions, _, _ = trainer.predict(test_dataset=ds)
        scores = predictions.squeeze().tolist()
        if isinstance(scores, float):
            scores = [scores]
        return [float(score) for score in scores]

    def close(self) -> None:
        for name in ["model", "tokenizer", "datasets", "torch", "transformers", "DataCollatorWithPadding", "CustomTrainer", "MT5ForRegression"]:
            if hasattr(self, name):
                try:
                    delattr(self, name)
                except Exception:
                    pass
        gc.collect()
        clear_cuda_cache()


class XCometScorer:
    def __init__(self, device: str, batch_size: int):
        maybe_set_cuda_device(device)
        from comet import download_model, load_from_checkpoint

        self.device = device
        model_path = download_model("Unbabel/XCOMET-XXL")
        self.model = load_from_checkpoint(model_path)
        self.batch_size = batch_size
        self.gpus = 1 if device.startswith("cuda") else 0
        self.devices = [cuda_device_index(device)] if device.startswith("cuda") else None
        self.accelerator = "cuda" if device.startswith("cuda") else "cpu"

    def score(self, rows: List[Dict[str, str]]) -> List[float]:
        model_output = self.model.predict(
            rows,
            batch_size=self.batch_size,
            gpus=self.gpus,
            devices=self.devices,
            accelerator=self.accelerator,
        )
        return [float(score) for score in model_output.scores]

    def close(self) -> None:
        for name in ["model"]:
            if hasattr(self, name):
                try:
                    delattr(self, name)
                except Exception:
                    pass
        gc.collect()
        clear_cuda_cache()


def build_metricx_model_classes():
    import torch
    from torch import nn
    import transformers
    import transformers.modeling_outputs

    BaseModelOutput = transformers.modeling_outputs.BaseModelOutput
    ModelOutput = transformers.modeling_outputs.ModelOutput

    MT5Config = transformers.models.mt5.modeling_mt5.MT5Config
    MT5PreTrainedModel = transformers.models.mt5.modeling_mt5.MT5PreTrainedModel
    MT5Stack = transformers.models.mt5.modeling_mt5.MT5Stack
    head_mask_warning = transformers.models.mt5.modeling_mt5.__HEAD_MASK_WARNING_MSG

    @dataclasses.dataclass
    class MT5ForRegressionOutput(ModelOutput):
        loss: object = None
        predictions: object = None

    class MT5ForRegression(MT5PreTrainedModel):
        """Standalone MetricX regression model, vendored from the local MetricX implementation."""

        def __init__(self, config):
            super().__init__(config)
            self.model_dim = config.d_model
            # MetricX scoring is pure regression inference; decoder KV caching is unnecessary
            # and breaks with newer MT5 cache semantics used by current transformers releases.
            self.config.use_cache = False
            self.shared = nn.Embedding(config.vocab_size, config.d_model)

            encoder_config = copy.deepcopy(config)
            encoder_config.is_decoder = False
            encoder_config.use_cache = False
            encoder_config.is_encoder_decoder = False
            self.encoder = MT5Stack(encoder_config, self.shared)

            decoder_config = copy.deepcopy(config)
            decoder_config.is_decoder = True
            decoder_config.is_encoder_decoder = False
            decoder_config.num_layers = config.num_decoder_layers
            self.decoder = MT5Stack(decoder_config, self.shared)

            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.post_init()
            self.model_parallel = False
            self.device_map = None

        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            # This wrapper never generates autoregressively, so cache-free decoding is the
            # stable path across transformers versions.
            use_cache = False
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if head_mask is not None and decoder_head_mask is None:
                if self.config.num_layers == self.config.num_decoder_layers:
                    warnings.warn(head_mask_warning, FutureWarning)
                    decoder_head_mask = head_mask

            if encoder_outputs is None:
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                )

            hidden_states = encoder_outputs[0]

            if self.model_parallel:
                torch.cuda.set_device(self.decoder.first_device)

            batch_size = input_ids.size(0)
            decoder_input_ids = torch.LongTensor([0]).repeat(batch_size).reshape(-1, 1)
            target_device = input_ids.device if input_ids is not None else hidden_states.device
            decoder_input_ids = decoder_input_ids.to(target_device)
            if decoder_attention_mask is None:
                decoder_attention_mask = torch.ones_like(decoder_input_ids, device=target_device)

            if self.model_parallel:
                torch.cuda.set_device(self.decoder.first_device)
                hidden_states = hidden_states.to(self.decoder.first_device)
                if decoder_input_ids is not None:
                    decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.decoder.first_device)
                if decoder_attention_mask is not None:
                    decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = decoder_outputs[0]

            if self.model_parallel:
                torch.cuda.set_device(self.encoder.first_device)
                self.lm_head = self.lm_head.to(self.encoder.first_device)
                sequence_output = sequence_output.to(self.lm_head.weight.device)

            if self.config.tie_word_embeddings:
                sequence_output = sequence_output * (self.model_dim**-0.5)

            lm_logits = self.lm_head(sequence_output)
            predictions = lm_logits[:, 0, 250089]
            predictions = torch.clamp(predictions, 0, 25)

            loss = None
            if labels is not None:
                loss_fct = nn.MSELoss()
                labels = labels.to(predictions.device)
                loss = loss_fct(predictions.view(-1), labels.view(-1))

            return MT5ForRegressionOutput(loss=loss, predictions=predictions)

    return MT5ForRegression


def build_translation_outputs(
    vmwe_root: Path,
    mt_root: Path,
    models: Sequence[str],
    datasets_to_translate: Sequence[str],
    pairs: Sequence[str],
    translation_device: str,
    enable_filter: bool,
    overwrite: bool,
    google_project_id: Optional[str],
) -> None:
    if should_print_runtime_banner():
        print(f"[INFO] Visible GPUs: {available_gpu_count()}")
        print("[INFO] Translation uses one job per (model, dataset, pair). Non-Google MT uses one device at a time; Google MT can run without GPUs.")
        print("[INFO] Recommended setting: one 48GB GPU is enough for one translation model job.")
    for model_name in models:
        canonical_model = canonical_model_name(model_name)
        translator = translator_for(model_name, translation_device, google_project_id)
        try:
            for dataset_name in datasets_to_translate:
                src_df = load_dataset_rows(vmwe_root, dataset_name)
                texts = src_df["src"].tolist()
                for pair in pairs:
                    filename = f"{dataset_name}_{pair}.csv"
                    out_path = migrate_legacy_artifact(mt_root, canonical_model, filename)
                    if translation_output_complete(out_path) and not overwrite:
                        print(f"[INFO] Skip existing translation file: {out_path}")
                        continue
                    translations = translator.translate_many(texts, pair)
                    out_df = pd.DataFrame({"src": texts, "mt": translations})
                    if enable_filter:
                        out_df = filter_translations(out_df, pair)
                    write_csv_atomic(out_df, out_path)
                    print(f"[INFO] Wrote {out_path}")
        finally:
            translator.close()
            gc.collect()


def build_eval_outputs(
    mt_root: Path,
    mt_eval_root: Path,
    models: Sequence[str],
    datasets_to_eval: Sequence[str],
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
        if "metricx" in metrics:
            print("[INFO] MetricX is heavy; in practice assign it its own 48GB GPU.")
        if "xcomet" in metrics:
            print("[INFO] XCOMET is heavy; in practice assign it its own 48GB GPU.")

    metricx_scorer = MetricXScorer(metricx_device, metricx_batch_size) if "metricx" in metrics else None
    xcomet_scorer = XCometScorer(xcomet_device, xcomet_batch_size) if "xcomet" in metrics else None
    try:
        if metricx_scorer is not None:
            print(f"[INFO] MetricX device: {metricx_device}")
        if xcomet_scorer is not None:
            print(f"[INFO] XCOMET device: {xcomet_device}")

        for model_name in models:
            canonical_model = canonical_model_name(model_name)
            for dataset_name in datasets_to_eval:
                for pair in pairs:
                    filename = f"{dataset_name}_{pair}.csv"
                    in_path = migrate_legacy_artifact(mt_root, canonical_model, filename)
                    if not in_path.exists():
                        print(f"[WARN] Missing translation file, skip evaluation: {in_path}")
                        continue
                    out_path = migrate_legacy_artifact(mt_eval_root, canonical_model, filename)
                    base_rows = load_translation_rows(in_path)
                    need_metricx = "metricx" in metrics
                    need_xcomet = "xcomet" in metrics
                    if not overwrite:
                        metricx_done = (not need_metricx) or eval_metric_complete(out_path, base_rows, "metricx_score")
                        xcomet_done = (not need_xcomet) or eval_metric_complete(out_path, base_rows, "xcomet_score")
                        if metricx_done and xcomet_done:
                            print(f"[INFO] Skip existing eval file: {out_path}")
                            continue

                    rows = base_rows.to_dict("records")
                    out_df = base_rows.copy()
                    if out_path.exists() and not overwrite:
                        try:
                            existing = pd.read_csv(out_path)
                        except Exception:
                            existing = pd.DataFrame()
                        if rows_match(existing, base_rows, ["src", "mt"]):
                            if "metricx_score" in existing.columns:
                                out_df["metricx_score"] = existing["metricx_score"]
                            if "xcomet_score" in existing.columns:
                                out_df["xcomet_score"] = existing["xcomet_score"]

                    if "metricx_score" not in out_df.columns:
                        out_df["metricx_score"] = pd.Series([pd.NA] * len(out_df), dtype="Float64")
                    if "xcomet_score" not in out_df.columns:
                        out_df["xcomet_score"] = pd.Series([pd.NA] * len(out_df), dtype="Float64")

                    if rows:
                        need_metricx_scores = need_metricx and not out_df["metricx_score"].notna().all()
                        need_xcomet_scores = need_xcomet and not out_df["xcomet_score"].notna().all()

                        if need_metricx_scores and need_xcomet_scores:
                            if metricx_scorer is None:
                                raise RuntimeError("MetricX scoring requested but MetricX scorer is unavailable.")
                            if xcomet_scorer is None:
                                raise RuntimeError("XCOMET scoring requested but XCOMET scorer is unavailable.")
                            with ThreadPoolExecutor(max_workers=2) as executor:
                                metricx_future = executor.submit(metricx_scorer.score, rows)
                                xcomet_future = executor.submit(xcomet_scorer.score, rows)
                                out_df["metricx_score"] = metricx_future.result()
                                out_df["xcomet_score"] = xcomet_future.result()
                        else:
                            if need_metricx_scores:
                                if metricx_scorer is None:
                                    raise RuntimeError("MetricX scoring requested but MetricX scorer is unavailable.")
                                out_df["metricx_score"] = metricx_scorer.score(rows)
                            if need_xcomet_scores:
                                if xcomet_scorer is None:
                                    raise RuntimeError("XCOMET scoring requested but XCOMET scorer is unavailable.")
                                out_df["xcomet_score"] = xcomet_scorer.score(rows)

                    out_df = out_df.loc[:, ["src", "mt", "metricx_score", "xcomet_score"]]
                    write_csv_atomic(out_df, out_path)
                    print(f"[INFO] Wrote {out_path}")
    finally:
        if metricx_scorer is not None:
            metricx_scorer.close()
        if xcomet_scorer is not None:
            xcomet_scorer.close()


def build_job_specs(stage: str, models: Sequence[str], datasets: Sequence[str], pairs: Sequence[str]) -> List[JobSpec]:
    return [JobSpec(stage=stage, model=model, dataset=dataset, pair=pair) for model in models for dataset in datasets for pair in pairs]


def run_parallel_jobs(
    stage: str,
    jobs: Sequence[JobSpec],
    args: argparse.Namespace,
    translation_devices: Sequence[str],
    metricx_devices: Sequence[str],
    xcomet_devices: Sequence[str],
) -> None:
    if not jobs:
        return

    python_exe = sys.executable
    script_path = str(Path(__file__).resolve())

    if stage == "translate":
        if any(model_uses_gpu_for_translation(job.model) for job in jobs):
            device_assignments = [(device, None) for device in translation_devices]
        else:
            if args.translation_gpus:
                device_assignments = [(device, None) for device in translation_devices]
            else:
                device_assignments = [(None, None) for _ in range(max(1, min(args.parallel_jobs, len(jobs))))]
    else:
        if "metricx" in args.metrics and "xcomet" in args.metrics:
            limit = min(len(metricx_devices), len(xcomet_devices))
            device_assignments = [(metricx_devices[i], xcomet_devices[i]) for i in range(limit)]
        elif "metricx" in args.metrics:
            device_assignments = [(device, None) for device in metricx_devices]
        elif "xcomet" in args.metrics:
            device_assignments = [(None, device) for device in xcomet_devices]
        else:
            device_assignments = [("cpu", None)]

    if stage == "translate" and any(model_uses_gpu_for_translation(job.model) for job in jobs):
        max_workers = max(1, min(len(device_assignments), len(jobs)))
    else:
        max_workers = max(1, min(args.parallel_jobs, len(device_assignments), len(jobs)))
    active: List[Tuple[subprocess.Popen, JobSpec, Tuple[Optional[str], Optional[str]]]] = []
    next_job = 0

    def stop_active_processes(processes: Sequence[Tuple[subprocess.Popen, JobSpec, Tuple[Optional[str], Optional[str]]]]) -> None:
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
        proc = spawn_job_process(python_exe, script_path, job, args, assignment)
        active.append((proc, job, assignment))

    try:
        while active:
            still_running: List[Tuple[subprocess.Popen, JobSpec, Tuple[Optional[str], Optional[str]]]] = []
            for proc, job, assignment in active:
                code = proc.poll()
                if code is None:
                    still_running.append((proc, job, assignment))
                    continue
                if code != 0:
                    stop_active_processes(still_running)
                    raise RuntimeError(f"Parallel job failed for {job.stage} {job.model} {job.dataset} {job.pair} with exit code {code}")
                print(f"[INFO] Finished job: {job.stage} {job.model} {job.dataset} {job.pair}")
                if next_job < len(jobs):
                    next_assignment = assignment
                    next_spec = jobs[next_job]
                    next_job += 1
                    next_proc = spawn_job_process(python_exe, script_path, next_spec, args, next_assignment)
                    still_running.append((next_proc, next_spec, next_assignment))
            active = still_running
    except BaseException:
        stop_active_processes(active)
        raise


def spawn_job_process(
    python_exe: str,
    script_path: str,
    job: JobSpec,
    args: argparse.Namespace,
    device_assignment: Tuple[Optional[str], Optional[str]],
) -> subprocess.Popen:
    metricx_device, xcomet_device = device_assignment
    command = [
        python_exe,
        script_path,
        "--stage",
        job.stage,
        "--models",
        job.model,
        "--datasets",
        job.dataset,
        "--pairs",
        job.pair,
        "--vmwe-root",
        str(args.vmwe_root),
        "--mt-root",
        str(args.mt_root),
        "--mt-eval-root",
        str(args.mt_eval_root),
        "--parallel-jobs",
        "1",
    ]

    if job.stage == "translate":
        if metricx_device is not None:
            command.extend(["--translation-device", metricx_device])
    else:
        if args.metrics:
            command.extend(["--metrics", *args.metrics])
        command.extend(["--metricx-batch-size", str(args.metricx_batch_size)])
        command.extend(["--xcomet-batch-size", str(args.xcomet_batch_size)])
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

    print(f"[INFO] Launching job on devices {device_assignment}: {job.stage} {job.model} {job.dataset} {job.pair}")
    env = os.environ.copy()
    env["VMWE_MT_CHILD_JOB"] = "1"
    return subprocess.Popen(command, cwd=str(ROOT), env=env)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce VMWE MT and QE outputs from datasets/VMWE into MWE/MT and MWE/MT_eval.",
        epilog=(
            "Notes:\n"
            "1. Filtering with lingua-language-detector is enabled by default because LLM-based MT can return the wrong target language.\n"
            "2. Google translation needs google-cloud-translate credentials and project configuration.\n"
            "3. Google-only translation does not require a GPU; if no translation GPU pool is provided it can still run, including in parallel.\n"
            "4. Recommended setting: one 48GB GPU is enough for one non-Google translation job.\n"
            "5. Recommended setting: one 48GB GPU is enough for one MetricX or XCOMET evaluation job.\n"
            "6. For stable scheduling, run one non-Google MT model per GPU and one evaluation model per GPU."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--stage", choices=["translate", "evaluate", "all"], default="all")
    parser.add_argument("--vmwe-root", type=Path, default=DEFAULT_VMWE_ROOT)
    parser.add_argument("--mt-root", type=Path, default=DEFAULT_MT_ROOT)
    parser.add_argument("--mt-eval-root", type=Path, default=DEFAULT_MT_EVAL_ROOT)
    parser.add_argument("--models", nargs="+", default=["GemmaX2"], help="Models: Google GemmaX2 LLaMAX phi4 Madlad M2M100 opus seamless")
    parser.add_argument("--pairs", nargs="+", default=["en-cs"], help="Target pairs like en-cs en-de en-es en-tr en-ja en-ru en-zh")
    parser.add_argument("--datasets", nargs="+", default=["LVC", "VPC", "VID", "Non_VMWE"], help="Datasets: LVC VPC VID Non_VMWE")
    parser.add_argument("--metrics", nargs="+", default=["metricx", "xcomet"], choices=["metricx", "xcomet"])
    parser.add_argument("--translation-device", default="auto", help="Device for non-Google MT, e.g. cuda:0 or cpu. One 48GB GPU is typically enough for one translation job.")
    parser.add_argument("--metricx-device", default="auto", help="Device for MetricX, e.g. cuda:1 or cpu. One 48GB GPU is typically enough for one MetricX job.")
    parser.add_argument("--xcomet-device", default="auto", help="Device for XCOMET, e.g. cuda:2 or cpu. One 48GB GPU is typically enough for one XCOMET job.")
    parser.add_argument("--translation-gpus", nargs="*", default=None, help="Pool for non-Google MT jobs, e.g. cuda:0 cuda:1. If multiple GPUs are provided, translation runs in parallel automatically. Recommend one 48GB GPU per concurrent translation job.")
    parser.add_argument("--metricx-gpus", nargs="*", default=None, help="Optional pool for parallel MetricX jobs, e.g. cuda:1 cuda:2. Recommend one 48GB GPU per concurrent MetricX job.")
    parser.add_argument("--xcomet-gpus", nargs="*", default=None, help="Optional pool for parallel XCOMET jobs, e.g. cuda:3 cuda:4. Recommend one 48GB GPU per concurrent XCOMET job.")
    parser.add_argument("--parallel-jobs", type=int, default=1, help="Maximum number of parallel jobs for Google-only translation and for evaluation.")
    parser.add_argument("--metricx-batch-size", type=int, default=1)
    parser.add_argument("--xcomet-batch-size", type=int, default=1)
    parser.add_argument("--google-project-id", default=None)
    parser.add_argument("--no-filter", action="store_true", help="Disable target-language filtering.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validate_pairs(args.pairs)
    translation_device = parse_device(args.translation_device)
    metricx_device = parse_device(args.metricx_device)
    xcomet_device = parse_device(args.xcomet_device)
    translation_devices = parse_device_list(args.translation_gpus, args.translation_device)
    metricx_devices = parse_device_list(args.metricx_gpus, args.metricx_device)
    xcomet_devices = parse_device_list(args.xcomet_gpus, args.xcomet_device)

    translation_parallel = False
    if args.stage in {"translate", "all"}:
        translation_parallel = (
            (any_translation_job_uses_gpu(args.models) and args.translation_gpus is not None and len(translation_devices) > 1)
            or (not any_translation_job_uses_gpu(args.models) and args.parallel_jobs > 1)
        )

    evaluation_parallel = args.stage in {"evaluate", "all"} and args.parallel_jobs > 1

    if translation_parallel:
        translate_jobs = build_job_specs("translate", args.models, args.datasets, args.pairs)
        run_parallel_jobs("translate", translate_jobs, args, translation_devices, metricx_devices, xcomet_devices)
    elif args.stage in {"translate", "all"}:
        build_translation_outputs(
            vmwe_root=args.vmwe_root,
            mt_root=args.mt_root,
            models=args.models,
            datasets_to_translate=args.datasets,
            pairs=args.pairs,
            translation_device=translation_device,
            enable_filter=not args.no_filter,
            overwrite=args.overwrite,
            google_project_id=args.google_project_id,
        )

    if evaluation_parallel:
        eval_jobs = build_job_specs("evaluate", args.models, args.datasets, args.pairs)
        run_parallel_jobs("evaluate", eval_jobs, args, translation_devices, metricx_devices, xcomet_devices)
    elif args.stage in {"evaluate", "all"}:
        build_eval_outputs(
            mt_root=args.mt_root,
            mt_eval_root=args.mt_eval_root,
            models=args.models,
            datasets_to_eval=args.datasets,
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
