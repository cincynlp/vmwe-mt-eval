#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WMT_ROOT = ROOT / "datasets" / "WMT"
DEFAULT_VMWE_ROOT = ROOT / "datasets" / "VMWE"
DEFAULT_PRESET_ROOT = ROOT / "preset" / "WMT"

LIGHT_VERBS = {"have", "take", "make", "get", "put", "give", "pay", "do", "offer", "raise"}


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv_with_columns(rows: List[Dict[str, object]], columns: Sequence[str], out_csv: Path) -> Path:
    mkdir(out_csv.parent)
    pd.DataFrame(rows, columns=list(columns)).to_csv(out_csv, index=False)
    return out_csv


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def year_to_short(year: int) -> int:
    return year % 100


def normalize_system_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def load_spacy():
    import spacy

    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        try:
            from spacy.cli import download
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception as exc:
            raise RuntimeError("spaCy model 'en_core_web_sm' is required.") from exc


def ensure_python_package(import_name: str, pip_name: Optional[str] = None):
    try:
        return __import__(import_name)
    except ImportError:
        package_name = pip_name or import_name
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return __import__(import_name)
        except Exception as exc:
            raise RuntimeError(f"Python package '{package_name}' is required.") from exc


def iter_wmt_system_files(wmt_root: Path) -> Iterator[Tuple[int, str, str, str, Path]]:
    for year_dir in sorted(p for p in wmt_root.iterdir() if p.is_dir() and p.name.isdigit()):
        year = int(year_dir.name)
        for category_dir in sorted(p for p in year_dir.iterdir() if p.is_dir() and p.name in {"Human", "MT"}):
            category = category_dir.name
            for pair_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
                pair = pair_dir.name
                for csv_path in sorted(pair_dir.glob("*.csv")):
                    yield year, category, pair, csv_path.stem, csv_path


def build_unique_src(wmt_root: Path, out_csv: Path) -> Path:
    seen = set()
    rows: List[Dict[str, str]] = []
    for _year, _category, _pair, _system, csv_path in tqdm(
        list(iter_wmt_system_files(wmt_root)),
        desc="Collecting unique WMT sources",
        unit="file",
    ):
        df = pd.read_csv(csv_path, usecols=["src"])
        for src in df["src"].astype(str):
            src = norm_text(src)
            if src and src not in seen:
                seen.add(src)
                rows.append({"src": src})
    mkdir(out_csv.parent)
    pd.DataFrame(rows, columns=["src"]).to_csv(out_csv, index=False)
    return out_csv


def load_unique_sentences(path: Path) -> List[str]:
    df = pd.read_csv(path)
    return [norm_text(src) for src in df["src"].astype(str) if norm_text(src)]


def load_vid_dictionary(vmwe_root: Path) -> List[str]:
    dict_csv = vmwe_root / "VID_dictionary.csv"
    dict_txt = vmwe_root / "VID_dictionary.txt"
    vid_csv = vmwe_root / "VID.csv"

    if dict_csv.exists():
        df = pd.read_csv(dict_csv)
        candidate_col = next((col for col in df.columns if col.lower() in {"vid_candidate", "candidate", "idiom"}), None)
        if candidate_col:
            return sorted({norm_text(v) for v in df[candidate_col].astype(str) if norm_text(v)})
    if dict_txt.exists():
        with dict_txt.open(encoding="utf-8") as f:
            return sorted({norm_text(line) for line in f if norm_text(line)})
    if vid_csv.exists():
        df = pd.read_csv(vid_csv)
        if "VID_candidate" in df.columns:
            return sorted({norm_text(v) for v in df["VID_candidate"].astype(str) if norm_text(v)})
    raise FileNotFoundError("Could not find a verbal idiom dictionary under datasets/VMWE.")


def extract_lvc_candidates(sentences: Sequence[str], out_csv: Path) -> Path:
    nlp = load_spacy()
    rows: List[Dict[str, str]] = []
    for doc in tqdm(nlp.pipe(sentences, batch_size=512), total=len(sentences), desc="Extracting WMT LVC candidates"):
        for token in doc:
            if token.lemma_.lower() in LIGHT_VERBS and token.pos_ == "VERB":
                for child in token.children:
                    if child.pos_ in {"NOUN", "PROPN"} and child.dep_ in {"dobj", "obj"}:
                        rows.append({"src": doc.text, "LVC_Candidate": f"{token.lemma_} {child.lemma_}"})
    df = pd.DataFrame(rows, columns=["src", "LVC_Candidate"]).drop_duplicates(ignore_index=True)
    return write_csv_with_columns(df.to_dict("records"), ["src", "LVC_Candidate"], out_csv)


def extract_vpc_candidates(sentences: Sequence[str], out_csv: Path) -> Path:
    nlp = load_spacy()
    rows: List[Dict[str, str]] = []
    for doc in tqdm(nlp.pipe(sentences, batch_size=512), total=len(sentences), desc="Extracting WMT VPC candidates"):
        candidate = None
        for token in doc:
            if token.pos_ == "VERB":
                for child in token.children:
                    if child.dep_ == "prt" and child.pos_ == "ADP":
                        candidate = f"{token.text} {child.text}"
                        break
            if candidate:
                break
        if candidate:
            rows.append({"src": doc.text, "VPC_Candidate": candidate})
    df = pd.DataFrame(rows, columns=["src", "VPC_Candidate"]).drop_duplicates(ignore_index=True)
    return write_csv_with_columns(df.to_dict("records"), ["src", "VPC_Candidate"], out_csv)


def normalize_vid_token(token) -> str:
    if token.pos_ == "PRON":
        return "[pron]"
    return token.lemma_.lower()


def extract_vid_candidates(sentences: Sequence[str], idioms: Sequence[str], out_csv: Path, threshold: float = 50.0) -> Path:
    nlp = load_spacy()
    sacrebleu = ensure_python_package("sacrebleu")

    idiom_dict: Dict[str, List[str]] = defaultdict(list)
    for idiom in idioms:
        first_word = idiom.split()[0] if idiom else ""
        idiom_dict[first_word].append(idiom)

    rows: List[Dict[str, object]] = []
    seen = set()
    for doc in tqdm(nlp.pipe(sentences, batch_size=256), total=len(sentences), desc="Extracting WMT VID candidates"):
        candidates: List[str] = []
        for token in doc:
            if token.pos_ != "VERB":
                continue
            for child in token.children:
                if child.dep_ in {"dobj", "prt"}:
                    candidates.append(f"{token.lemma_} {child.lemma_}")
                elif child.dep_ == "prep":
                    subtree = sorted(list(child.subtree), key=lambda t: t.i)
                    subtree_text = " ".join(normalize_vid_token(t) for t in subtree if not t.is_punct)
                    candidates.append(f"{token.lemma_} {subtree_text}")

        for candidate in candidates:
            first_word = candidate.split()[0] if candidate else ""
            relevant_idioms = idiom_dict.get(first_word) or list(idioms)
            for idiom in relevant_idioms:
                score = sacrebleu.sentence_bleu(candidate, [idiom]).score
                if score > threshold:
                    key = (doc.text, candidate, idiom)
                    if key not in seen:
                        seen.add(key)
                        rows.append(
                            {
                                "src": doc.text,
                                "VID_Candidate": candidate,
                                "Matching_Idiom": idiom,
                                "BLEU_Score": score,
                            }
                        )
    df = pd.DataFrame(rows, columns=["src", "VID_Candidate", "Matching_Idiom", "BLEU_Score"]).drop_duplicates(ignore_index=True)
    return write_csv_with_columns(df.to_dict("records"), ["src", "VID_Candidate", "Matching_Idiom", "BLEU_Score"], out_csv)


def find_preset_file(preset_root: Path, kind: str) -> Path:
    candidates = [
        preset_root / f"{kind}.csv",
        preset_root / f"WMT_{kind}_Classified_2017_to_2024.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find preset file for {kind} under {preset_root}")


def candidate_col_for_kind(kind: str) -> str:
    return f"{kind}_Candidate"


def normalize_classification_df(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    candidate_col = candidate_col_for_kind(kind)
    rename_map = {}
    for col in df.columns:
        low = col.lower().replace(" ", "_")
        if low == "src":
            rename_map[col] = "src"
        elif low == candidate_col.lower():
            rename_map[col] = candidate_col
        elif low == "prediction":
            rename_map[col] = "Prediction"
    df = df.rename(columns=rename_map)
    required = ["src", candidate_col, "Prediction"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {kind}: {missing}")
    return df.loc[:, required].copy()


def use_preset_classification(kind: str, preset_root: Path, out_path: Path) -> Path:
    df = pd.read_csv(find_preset_file(preset_root, kind))
    df = normalize_classification_df(df, kind)
    mkdir(out_path.parent)
    df.to_csv(out_path, index=False)
    return out_path


def load_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for API classification mode.") from exc
    return OpenAI()


def get_lvc_candidate_lemmas(nlp, lvc_candidate: str) -> Tuple[Optional[str], Optional[str]]:
    parts = lvc_candidate.split("_") if "_" in lvc_candidate else lvc_candidate.split()
    if len(parts) < 2:
        return None, None
    try:
        return nlp(parts[0])[0].lemma_.lower(), nlp(parts[-1])[0].lemma_.lower()
    except Exception:
        return None, None


def validate_lvc_structure(nlp, src_text: str, lvc_candidate: str) -> bool:
    first, last = get_lvc_candidate_lemmas(nlp, lvc_candidate)
    if not first or not last:
        return False
    doc = nlp(src_text.replace('"', ""))
    candidate_verbs = [token for token in doc if token.lemma_.lower() == first and token.pos_ == "VERB"]
    candidate_nouns = [token for token in doc if token.lemma_.lower() == last and token.pos_ == "NOUN"]
    if not candidate_verbs or not candidate_nouns:
        return False
    for noun in candidate_nouns:
        if noun.head in candidate_verbs:
            return True
        for ancestor in noun.ancestors:
            if ancestor in candidate_verbs:
                return True
    return False


def generate_lvc_prompt(verb_lemma: str, noun_lemma: str, context: str) -> str:
    lvc_definitions = (
        "Key Definitions:\n"
        "1. Abstract Noun (N-ABS): Refers to non-physical concepts (e.g., 'decision', 'discussion', 'analysis').\n"
        "2. Predicative Noun (N-PRED): Can assign semantic roles (e.g., 'offer' in 'make an offer' has a recipient).\n"
        "3. Light Verb (V-LIGHT): Contributes only grammatical features (tense, aspect) not core meaning.\n"
        "4. NP-Reduction Test: Valid if nominalization works (e.g., 'make a decision' → 'the decision').\n\n"
    )
    decision_tree = (
        "Enhanced Decision Protocol:\n"
        "1. N-ABS: Is the noun abstract?\n"
        "   - Consider: Can it refer to an event/state/activity rather than physical object?\n"
        "   - Example: 'give a speech' → 'speech' is abstract (✓), 'build a house' → concrete (✗)\n\n"
        "2. N-PRED: Does the noun have predicate-like arguments?\n"
        "   - Check for: Possible agents/patients/recipients in context\n"
        "   - Example: 'take a walk' → walk doesn't take arguments (✗), 'make a claim' → claim has content (✓)\n\n"
        "3. V-SUBJ-N-ARG: Is verb's subject a semantic argument of noun?\n"
        "   - Example: In 'She gave a speech', 'she' is agent of both verb and noun (✓)\n\n"
        "4. V-LIGHT: Does verb only provide grammatical features?\n"
        "   - Warning: Reject if verb adds manner/direction/intensity\n"
        "   - Example: 'take a walk' → 'take' is light (✓), 'enjoy a walk' → 'enjoy' adds meaning (✗)\n\n"
        "5. V-REDUC: Does NP-reduction preserve event meaning?\n"
        "   - Test: Try replacing with noun phrase alone\n"
        "   - Example: 'have a discussion' → 'the discussion' works (✓), 'read a book' → 'the book' loses event (✗)\n\n"
    )
    examples = (
        "Positive Example (C):\n"
        "Context: 'The committee reached a decision'\n"
        "- decision: abstract (✓), predicative (✓)\n"
        "- 'reached' subject is decision's agent (✓)\n"
        "- 'reached' is light (✓), NP-reduction 'the decision' works (✓)\n\n"
        "Negative Example (A):\n"
        "Context: 'They constructed a building'\n"
        "- building: concrete (✗) → Exit at N-ABS\n"
    )
    instructions = (
        "Analysis Instructions:\n"
        "1. Systematically apply each test in order\n"
        "2. For borderline cases, favor deeper analysis\n"
        "3. Final Answer Format: 'Final Answer: [LETTER]'\n\n"
        "Classification Options:\n"
        "A: Noun not abstract     B: Noun not predicative\n"
        "C: LVC detected          D: Subject mismatch\n"
        "E: Verb adds meaning     F: NP-reduction fails\n"
    )
    clean_context = context.replace('"', "")
    return (
        f"<|im_start|>system<|im_sep|>\n"
        "Linguistic Analysis Task: Identify Light Verb Constructions (LVCs)\n\n"
        f"{lvc_definitions}{decision_tree}{examples}{instructions}"
        f"Target Analysis:\nContext: {clean_context}\n- VERB: {verb_lemma}\n- NOUN: {noun_lemma}\n"
        "<|im_end|>\n"
        "<|im_start|>user<|im_sep|>\n"
        "Perform step-by-step analysis and provide classification.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant<|im_sep|>\n"
    )


def extract_lvc_decision(output: str) -> str:
    for pattern in [r"(?i)Final Answer:\s*([A-F])\b", r"\b([A-F])\b(?![\.\d])"]:
        match = re.search(pattern, output)
        if match:
            return match.group(1).upper()
    return "C"


def get_vpc_components(nlp, vpc_candidate: str) -> Tuple[Optional[str], Optional[str]]:
    doc = nlp(vpc_candidate.replace("_", " "))
    for token in doc:
        if token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ == "prt":
                    return token.lemma_.lower(), child.text.lower()
    return None, None


def validate_vpc_structure(nlp, src_text: str, vpc_candidate: str) -> bool:
    verb_lemma, particle = get_vpc_components(nlp, vpc_candidate)
    if not verb_lemma or not particle:
        return False
    doc = nlp(src_text)
    verbs = [token for token in doc if token.lemma_.lower() == verb_lemma and token.pos_ == "VERB"]
    return any(child.dep_ == "prt" and child.text.lower() == particle for verb in verbs for child in verb.children)


def generate_vpc_prompt(verb_lemma: str, particle: str, context: str) -> str:
    decision_tree = (
        "VPC Decision Tree:\n"
        "1. Is the second element a particle (e.g., 'up', 'off')?\n"
        "   - No → Not VPC (A)\n"
        "   - Yes → Continue\n"
        "2. Remove the particle from the combination. Does the remaining verb convey the same meaning as the full verb-particle phrase?\n"
        "   - Yes → Not VPC (B)\n"
        "   - No → Continue\n"
        "3. Does the inclusion of the particle create a non-compositional meaning that is significantly different from the verb's original meaning?\n"
        "   - No → Not VPC (C)\n"
        "   - Yes → VPC (D)\n"
    )
    options = (
        "Options:\n"
        "A: Not a particle\n"
        "B: Meaning remains similar without the particle\n"
        "C: Particle does not significantly alter the meaning\n"
        "D: Valid VPC (Particle significantly alters meaning)\n\n"
    )
    return (
        f"<|im_start|>system<|im_sep|>\n"
        "Determine if the following verb-particle combination constitutes a valid Verb-Particle Construction (VPC).\n\n"
        f"Context: {context}\n"
        f"Combination: {verb_lemma} {particle}\n\n"
        f"{decision_tree}{options}"
        "Provide step-by-step reasoning and conclude with 'Final Answer: [LETTER]'.\n"
        "<|im_end|>\n"
        "<|im_start|>user<|im_sep|>\n"
        "Analyze the VPC and provide your classification.\n"
        "<|im_end|>\n"
    )


def extract_vpc_decision(output: str) -> str:
    for pattern in [r"(?i)Final Answer:\s*([A-D])\b", r"\b([A-D])\b(?![\.\d])"]:
        match = re.search(pattern, output)
        if match:
            return match.group(1).upper()
    return "D"


def validate_vid_structure(nlp, candidate: str, _sentence: str) -> bool:
    doc = nlp(candidate)
    if len(doc) < 2:
        return False
    verbs = [token for token in doc if token.pos_ == "VERB"]
    if not verbs:
        return False
    return any(token != verbs[0] and token.head == verbs[0] for token in doc)


def generate_vid_prompt(candidate: str, context: str) -> str:
    decision_tree = (
        "VID Decision Tree:\n"
        "1. [CRAN] Contains cranberry word?\n"
        "   Yes → VID\n"
        "   No → Next test\n\n"
        "2. [LEX] Regular replacement changes meaning?\n"
        "   Yes → VID\n"
        "   No → Next test\n\n"
        "3. [MORPH] Morphological changes affect meaning?\n"
        "   Yes → VID\n"
        "   No → Next test\n\n"
        "4. [MORPHSYNT] Morphosyntactic changes affect meaning?\n"
        "   Yes → VID\n"
        "   No → Next test\n\n"
        "5. [SYNT] Syntactic changes affect meaning?\n"
        "   Yes → VID\n"
        "   No → Not VID\n\n"
        "Examples:\n"
        "- VID: 'kick the bucket', 'let the cat out of the bag'\n"
        "- Non-VID: 'take a walk', 'make a decision'\n\n"
        "Instructions:\n"
        "1. Analyze each test sequentially\n"
        "2. Provide brief reasoning for each test\n"
        "3. Conclude with 'Final Answer: [Yes/No]'\n"
    )
    return (
        f"<|im_start|>system<|im_sep|>\n"
        "Verbal Idiom (VID) Classification Task\n\n"
        f"Context Sentence: {context}\n"
        f"Candidate Phrase: {candidate}\n\n"
        f"{decision_tree}"
        "<|im_end|>\n"
        "<|im_start|>user<|im_sep|>\n"
        "Is this candidate a Verbal Idiom (VID)? Apply the decision tree.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant<|im_sep|>\n"
    )


def extract_vid_decision(output: str) -> str:
    patterns = [
        r"Final Answer:\s*(Yes|No)",
        r"(?i)\b(VID|Not VID)\b",
        r"(?i)\b(Yes|No)\b(?!.*\b(Yes|No)\b)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            decision = match.group(1).lower()
            return "positive" if decision in {"yes", "vid"} else "negative"
    return "negative"


def call_responses_api(client, model: str, prompt: str, max_output_tokens: int, temperature: float, top_p: float) -> str:
    response = client.responses.create(
        model=model,
        input=[{"role": "system", "content": prompt}],
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return response.output_text


def classify_with_api(kind: str, input_csv: Path, out_csv: Path, model: str) -> Path:
    nlp = load_spacy()
    client = load_openai_client()
    candidate_col = candidate_col_for_kind(kind)
    try:
        df = pd.read_csv(input_csv)
    except pd.errors.EmptyDataError:
        return write_csv_with_columns([], ["src", candidate_col, "Prediction"], out_csv)
    if df.empty:
        return write_csv_with_columns([], ["src", candidate_col, "Prediction"], out_csv)
    output_rows: List[Dict[str, str]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Classifying {kind} candidates"):
        src = norm_text(row["src"])
        candidate = norm_text(row[candidate_col])
        prediction = "negative"
        if kind == "LVC":
            if validate_lvc_structure(nlp, src, candidate):
                first, last = get_lvc_candidate_lemmas(nlp, candidate)
                if first and last:
                    prompt = generate_lvc_prompt(first, last, src)
                    letter = extract_lvc_decision(call_responses_api(client, model, prompt, 900, 0.3, 0.9))
                    prediction = "positive" if letter == "C" else "negative"
        elif kind == "VPC":
            if validate_vpc_structure(nlp, src, candidate):
                verb_lemma, particle = get_vpc_components(nlp, candidate)
                if verb_lemma and particle:
                    prompt = generate_vpc_prompt(verb_lemma, particle, src)
                    letter = extract_vpc_decision(call_responses_api(client, model, prompt, 900, 0.8, 0.75))
                    prediction = "positive" if letter == "D" else "negative"
        elif kind == "VID":
            if validate_vid_structure(nlp, candidate, src):
                prompt = generate_vid_prompt(candidate, src)
                prediction = extract_vid_decision(call_responses_api(client, model, prompt, 900, 0.7, 0.85))
        output_rows.append({"src": src, candidate_col: candidate, "Prediction": prediction})

    return write_csv_with_columns(output_rows, ["src", candidate_col, "Prediction"], out_csv)


def compile_idiom_patterns(idioms: Sequence[str]) -> List[Tuple[str, re.Pattern[str]]]:
    patterns = []
    pronoun_pattern = r"(?:I|you|he|she|it|we|they|my|your|his|her|its|our|their)"
    for idiom in idioms:
        if "[pron]" in idiom:
            parts = idiom.split("[pron]")
            escaped_parts = [re.escape(part) for part in parts]
            pattern_str = pronoun_pattern.join(escaped_parts)
            pattern = re.compile(r"\b" + pattern_str + r"\b", re.IGNORECASE)
        else:
            pattern = re.compile(r"\b" + re.escape(idiom) + r"\b", re.IGNORECASE)
        patterns.append((idiom, pattern))
    return patterns


def compute_global_normal_set(sentences: Sequence[str], idioms: Sequence[str]) -> set[str]:
    idiom_patterns = compile_idiom_patterns(idioms)
    idiom_hits = set()
    for sentence in tqdm(sentences, desc="Filtering idioms for WMT normal baseline"):
        if any(pattern.search(sentence) for _idiom, pattern in idiom_patterns):
            idiom_hits.add(sentence)

    nlp = load_spacy()
    light_hits = set()
    vpc_hits = set()
    for doc in tqdm(nlp.pipe(sentences, batch_size=256), total=len(sentences), desc="Filtering light-verb/VPC normals"):
        sentence = doc.text
        has_light = False
        has_vpc = False
        for token in doc:
            if token.lemma_.lower() in LIGHT_VERBS and token.pos_ == "VERB":
                has_light = True
            if token.pos_ == "VERB":
                for child in token.children:
                    if child.dep_ == "prt" and child.pos_ == "ADP":
                        has_vpc = True
                        break
            if has_light and has_vpc:
                break
        if has_light:
            light_hits.add(sentence)
        if has_vpc:
            vpc_hits.add(sentence)

    excluded = idiom_hits | light_hits | vpc_hits
    return {sentence for sentence in sentences if sentence not in excluded}


def load_positive_srcs(path: Path) -> set[str]:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return set()
    if df.empty or "Prediction" not in df.columns or "src" not in df.columns:
        return set()
    return {norm_text(src) for src in df[df["Prediction"].astype(str).str.lower() == "positive"]["src"].astype(str)}


def avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def normal_candidate_file_path(wmt_root: Path, category: str, year: int, pair: str, system: str) -> Path:
    safe_system = re.sub(r"[^A-Za-z0-9._-]+", "_", system)
    return wmt_root / "normal_candidates" / category / str(year) / pair / f"{safe_system}_non_vmwe.csv"


def summarize_system_file(
    wmt_root: Path,
    category: str,
    year: int,
    pair: str,
    system: str,
    path: Path,
    vid_srcs: set[str],
    lvc_srcs: set[str],
    vpc_srcs: set[str],
    normal_srcs: set[str],
) -> Dict[str, float]:
    df = pd.read_csv(path, usecols=["src", "score"])
    vid_scores: List[float] = []
    lvc_scores: List[float] = []
    vpc_scores: List[float] = []
    normal_scores: List[float] = []
    normal_rows: List[Dict[str, float]] = []

    for _, row in df.iterrows():
        src = norm_text(row["src"])
        try:
            score = float(row["score"])
        except Exception:
            continue
        if src in vid_srcs:
            vid_scores.append(score)
        elif src in lvc_srcs:
            lvc_scores.append(score)
        elif src in vpc_srcs:
            vpc_scores.append(score)
        elif src in normal_srcs:
            normal_scores.append(score)
            normal_rows.append({"src": src, "score": score})

    normal_path = normal_candidate_file_path(wmt_root, category, year, pair, system)
    mkdir(normal_path.parent)
    pd.DataFrame(normal_rows, columns=["src", "score"]).to_csv(normal_path, index=False)

    return {
        "VID_avg": avg(vid_scores),
        "LVC_avg": avg(lvc_scores),
        "VPC_avg": avg(vpc_scores),
        "Normal_avg": avg(normal_scores),
    }


def resolve_ranked_mt_file(pair_dir: Path, ranked_name: str) -> Optional[Path]:
    if not pair_dir.exists():
        return None
    ranked_key = normalize_system_key(ranked_name)
    files = list(pair_dir.glob("*.csv"))
    exact = next((p for p in files if normalize_system_key(p.stem) == ranked_key), None)
    if exact:
        return exact
    return next((p for p in files if ranked_key in normalize_system_key(p.stem) or normalize_system_key(p.stem) in ranked_key), None)


def build_mt_summary(
    wmt_root: Path,
    ranking_csv: Path,
    vid_srcs: set[str],
    lvc_srcs: set[str],
    vpc_srcs: set[str],
    normal_srcs: set[str],
) -> pd.DataFrame:
    ranking_df = pd.read_csv(ranking_csv)
    rows: List[Dict[str, object]] = []
    for _, row in ranking_df.iterrows():
        year_short = int(row["year"])
        year = 2000 + year_short if year_short < 100 else year_short
        pair = norm_text(row["language pair"])
        pair_dir = wmt_root / str(year) / "MT" / pair
        for col in ["rank1", "rank2", "rank3", "rank4"]:
            system_name = norm_text(row.get(col, ""))
            if not system_name:
                continue
            system_file = resolve_ranked_mt_file(pair_dir, system_name)
            if not system_file:
                continue
            summary = summarize_system_file(wmt_root, "MT", year, pair, system_file.stem, system_file, vid_srcs, lvc_srcs, vpc_srcs, normal_srcs)
            rows.append({"Year": year_short, "Language_Pair": pair, "System": system_file.stem, **summary})
    return pd.DataFrame(rows).drop_duplicates(ignore_index=True)


def build_human_summary(
    wmt_root: Path,
    vid_srcs: set[str],
    lvc_srcs: set[str],
    vpc_srcs: set[str],
    normal_srcs: set[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for year, category, pair, system, csv_path in iter_wmt_system_files(wmt_root):
        if category != "Human":
            continue
        summary = summarize_system_file(wmt_root, category, year, pair, system, csv_path, vid_srcs, lvc_srcs, vpc_srcs, normal_srcs)
        rows.append({"Year": year_to_short(year), "Language_Pair": pair, "System": system, **summary})
    return pd.DataFrame(rows).drop_duplicates(ignore_index=True)


def copy_preset_final_summary(src: Path, dst: Path) -> Path:
    mkdir(dst.parent)
    shutil.copyfile(src, dst)
    return dst


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WMT VMWE candidate, classification, and summary files.")
    parser.add_argument("--wmt-root", type=Path, default=DEFAULT_WMT_ROOT)
    parser.add_argument("--vmwe-root", type=Path, default=DEFAULT_VMWE_ROOT)
    parser.add_argument("--preset-root", type=Path, default=DEFAULT_PRESET_ROOT)
    parser.add_argument("--classification", choices=["none", "preset", "api"], default="preset")
    parser.add_argument("--openai-model", default="gpt-4o")
    parser.add_argument("--use-preset-final-summary", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    wmt_root = args.wmt_root
    vmwe_root = args.vmwe_root
    preset_root = args.preset_root

    unique_src_csv = wmt_root / "WMT_unique_src.csv"
    lvc_candidates_csv = wmt_root / "WMT_LVC_candidates.csv"
    vpc_candidates_csv = wmt_root / "WMT_VPC_candidates.csv"
    vid_candidates_csv = wmt_root / "WMT_VID_candidates.csv"

    lvc_classified_csv = wmt_root / "LVC.csv"
    vpc_classified_csv = wmt_root / "VPC.csv"
    vid_classified_csv = wmt_root / "VID.csv"

    mt_summary_csv = wmt_root / "WMT_MT_VMWE_vs_non_VMWE.csv"
    human_summary_csv = wmt_root / "WMT_Human_VMWE_vs_non_VMWE.csv"
    combined_summary_csv = wmt_root / "WMT_summary.csv"
    normal_src_csv = wmt_root / "WMT_normal_sources.csv"

    build_unique_src(wmt_root, unique_src_csv)
    sentences = load_unique_sentences(unique_src_csv)
    idioms = load_vid_dictionary(vmwe_root)

    extract_lvc_candidates(sentences, lvc_candidates_csv)
    extract_vpc_candidates(sentences, vpc_candidates_csv)
    extract_vid_candidates(sentences, idioms, vid_candidates_csv)

    if args.classification == "preset":
        use_preset_classification("LVC", preset_root, lvc_classified_csv)
        use_preset_classification("VPC", preset_root, vpc_classified_csv)
        use_preset_classification("VID", preset_root, vid_classified_csv)
    elif args.classification == "api":
        classify_with_api("LVC", lvc_candidates_csv, lvc_classified_csv, args.openai_model)
        classify_with_api("VPC", vpc_candidates_csv, vpc_classified_csv, args.openai_model)
        classify_with_api("VID", vid_candidates_csv, vid_classified_csv, args.openai_model)

    vid_srcs = load_positive_srcs(vid_classified_csv)
    lvc_srcs = load_positive_srcs(lvc_classified_csv)
    vpc_srcs = load_positive_srcs(vpc_classified_csv)

    normal_srcs = compute_global_normal_set(sentences, idioms)
    pd.DataFrame({"src": sorted(normal_srcs)}).to_csv(normal_src_csv, index=False)

    if args.use_preset_final_summary:
        copy_preset_final_summary(preset_root / "WMT_MT.csv", mt_summary_csv)
        copy_preset_final_summary(preset_root / "WMT_Human.csv", human_summary_csv)
        mt_df = pd.read_csv(mt_summary_csv)
        human_df = pd.read_csv(human_summary_csv)
    else:
        mt_df = build_mt_summary(wmt_root, preset_root / "WMT_system_rankings.csv", vid_srcs, lvc_srcs, vpc_srcs, normal_srcs)
        human_df = build_human_summary(wmt_root, vid_srcs, lvc_srcs, vpc_srcs, normal_srcs)
        mt_df.to_csv(mt_summary_csv, index=False)
        human_df.to_csv(human_summary_csv, index=False)

    combined_df = pd.concat(
        [
            mt_df.assign(Category="MT"),
            human_df.assign(Category="Human"),
        ],
        ignore_index=True,
    )
    combined_df.to_csv(combined_summary_csv, index=False)

    print(f"[INFO] Wrote {unique_src_csv}")
    print(f"[INFO] Wrote {lvc_candidates_csv}")
    print(f"[INFO] Wrote {vid_candidates_csv}")
    print(f"[INFO] Wrote {vpc_candidates_csv}")
    print(f"[INFO] Wrote {lvc_classified_csv}")
    print(f"[INFO] Wrote {vid_classified_csv}")
    print(f"[INFO] Wrote {vpc_classified_csv}")
    print(f"[INFO] Wrote {normal_src_csv}")
    print(f"[INFO] Wrote {mt_summary_csv}")
    print(f"[INFO] Wrote {human_summary_csv}")
    print(f"[INFO] Wrote {combined_summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
