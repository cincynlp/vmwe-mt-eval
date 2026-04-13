#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
import random
import re
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = ROOT / "raw_dataset" / "VMWE"
DEFAULT_OUT_ROOT = ROOT / "datasets" / "VMWE"
MIN_WORDS = 10
MAX_WORDS = 40
LIGHT_VERBS = {"have", "take", "make", "get", "put", "give", "pay", "do", "offer", "raise"}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    kind: str
    url: str
    archive_name: str
    aliases: Tuple[str, ...]


SOURCES: Tuple[SourceSpec, ...] = (
    SourceSpec(
        name="lvc",
        kind="direct",
        url="https://cogcomp.seas.upenn.edu/software/tools/MWE_LVC_token.tar.gz",
        archive_name="MWE_LVC_token.tar.gz",
        aliases=("MWE_LVC_token",),
    ),
    SourceSpec(
        name="vpc",
        kind="direct",
        url="http://cogcomp.seas.upenn.edu/software/tools/pvcDataSubmission.tar.gz",
        archive_name="pvcDataSubmission.tar.gz",
        aliases=("pvcDataSubmission",),
    ),
    SourceSpec(
        name="bnc",
        kind="direct",
        url="https://llds.ling-phil.ox.ac.uk/llds/xmlui/bitstream/handle/20.500.14106/2554/2554.zip?sequence=4&isAllowed=y",
        archive_name="2554.zip",
        aliases=("download", "Texts", "2554"),
    ),
    SourceSpec(
        name="epie",
        kind="github",
        url="https://github.com/prateeksaxena2809/EPIE_Corpus",
        archive_name="EPIE_Corpus-master.tar.gz",
        aliases=("EPIE_Corpus-master", "EPIE_Corpus-main", "EPIE_Corpus"),
    ),
    SourceSpec(
        name="magpie",
        kind="github",
        url="https://github.com/hslh/magpie-corpus",
        archive_name="magpie-corpus-master.tar.gz",
        aliases=("magpie-corpus-master", "magpie-corpus-main", "magpie-corpus"),
    ),
)


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def import_requests():
    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Downloading requires the 'requests' package.") from exc
    return requests


def download_http(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        return dest
    requests = import_requests()
    mkdir(dest.parent)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fout:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fout.write(chunk)
    return dest


def parse_github_repo_url(url: str) -> Tuple[str, str, str]:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Unsupported GitHub URL: {url}")
    owner, repo = parts[0], parts[1]
    branch = "master"
    if len(parts) >= 4 and parts[2] == "tree":
        branch = parts[3]
    return owner, repo, branch


def download_github(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        return dest
    owner, repo, branch = parse_github_repo_url(url)
    archive_url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/refs/heads/{branch}"
    requests = import_requests()
    mkdir(dest.parent)
    with requests.get(archive_url, stream=True, timeout=120) as resp:
        if resp.status_code == 404 and branch == "master":
            archive_url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/refs/heads/main"
            resp = requests.get(archive_url, stream=True, timeout=120)
        resp.raise_for_status()
        with dest.open("wb") as fout:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fout.write(chunk)
    return dest


def safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    mkdir(dest_dir)
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            resolved = (dest_dir / member.name).resolve()
            if not str(resolved).startswith(str(dest_dir.resolve())):
                raise RuntimeError(f"Refusing to extract path outside destination: {member.name}")
        tar.extractall(dest_dir, filter="data")


def safe_extract_zip(archive_path: Path, dest_dir: Path) -> None:
    mkdir(dest_dir)
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.infolist():
            resolved = (dest_dir / member.filename).resolve()
            if not str(resolved).startswith(str(dest_dir.resolve())):
                raise RuntimeError(f"Refusing to extract path outside destination: {member.filename}")
        zf.extractall(dest_dir)


def extract_if_needed(archive_path: Path, dest_dir: Path) -> None:
    if dest_dir.exists() and any(dest_dir.iterdir()):
        return
    if archive_path.suffix == ".zip":
        safe_extract_zip(archive_path, dest_dir)
    else:
        safe_extract_tar(archive_path, dest_dir)


def find_alias_root(base: Path, aliases: Sequence[str]) -> Optional[Path]:
    if not base.exists():
        return None
    for alias in aliases:
        direct = base / alias
        if direct.exists():
            return direct
        hits = sorted(p for p in base.rglob("*") if p.name == alias)
        if hits:
            return hits[0]
    if any(base.iterdir()):
        return base
    return None


def ensure_sources(raw_root: Path, skip_download: bool, force: bool) -> Dict[str, Path]:
    downloads_dir = mkdir(raw_root / "downloads")
    extracted_dir = mkdir(raw_root / "extracted")
    resolved: Dict[str, Path] = {}

    for spec in SOURCES:
        archive_path = downloads_dir / spec.archive_name
        if not skip_download:
            if spec.kind == "direct":
                download_http(spec.url, archive_path, force=force)
            elif spec.kind == "github":
                download_github(spec.url, archive_path, force=force)
            else:
                raise ValueError(f"Unsupported source kind: {spec.kind}")
        if archive_path.exists():
            extract_if_needed(archive_path, extracted_dir / spec.name)
        found = find_alias_root(extracted_dir / spec.name, spec.aliases)
        if not found:
            raise FileNotFoundError(f"Could not resolve extracted root for source '{spec.name}'.")
        resolved[spec.name] = found
    return resolved


def norm_space(text: str) -> str:
    text = text.replace("\u00A0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_surface(text: str) -> str:
    text = norm_space(text)
    text = re.sub(r"\s+([,.;:?!%])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    text = text.replace(" n't", "n't").replace(" 's", "'s").replace(" 're", "'re").replace(" 've", "'ve")
    return text


def detokenize(tokens: Sequence[str]) -> str:
    if not tokens:
        return ""
    out: List[str] = []
    no_space_before = {".", ",", ";", ":", "?", "!", "%", ")", "]", "}", "n't", "'s", "'re", "'ve", "'d", "'ll", "'m"}
    no_space_after = {"(", "[", "{", '"', "``"}
    for tok in tokens:
        tok = norm_space(tok)
        if not tok:
            continue
        if not out:
            out.append(tok)
            continue
        if tok in no_space_before or re.fullmatch(r"[.,;:?!%]+", tok):
            out[-1] = out[-1] + tok
        elif out[-1] in no_space_after:
            out[-1] = out[-1] + tok
        else:
            out.append(" " + tok)
    return clean_surface("".join(out))


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def resolve_bnc_xml_subset(xml_path: Path, wanted_ids: Sequence[str]) -> Dict[str, Tuple[str, List[str]]]:
    wanted = set(wanted_ids)
    found: Dict[str, Tuple[str, List[str]]] = {}
    context = ET.iterparse(xml_path, events=("end",))
    for _event, elem in context:
        if local_name(elem.tag) != "s":
            continue
        sent_id = elem.attrib.get("n")
        if sent_id not in wanted:
            elem.clear()
            continue
        pieces: List[str] = []
        word_tokens: List[str] = []
        for node in elem.iter():
            name = local_name(node.tag)
            if name not in {"w", "c"}:
                continue
            tok = norm_space("".join(node.itertext()))
            if not tok:
                continue
            pieces.append(tok)
            if name == "w":
                word_tokens.append(tok)
        found[sent_id] = (detokenize(pieces), word_tokens)
        wanted.discard(sent_id)
        elem.clear()
        if not wanted:
            break
    return found


class BNCSentenceResolver:
    def __init__(self, bnc_root: Path):
        self.texts_root = self._locate_texts_root(bnc_root)
        self.cache: Dict[Path, Dict[str, Tuple[str, List[str]]]] = {}
        cpu_count = os.cpu_count() or 4
        self.max_workers = max(1, min(16, cpu_count))

    def _locate_texts_root(self, root: Path) -> Path:
        candidates = [
            root / "Texts",
            root / "download" / "Texts",
            root,
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        raise FileNotFoundError(f"Could not locate BNC Texts directory under {root}")

    def _load_file(self, xml_path: Path) -> Dict[str, Tuple[str, List[str]]]:
        if xml_path in self.cache:
            return self.cache[xml_path]
        tree = ET.parse(xml_path)
        root = tree.getroot()
        sent_map: Dict[str, Tuple[str, List[str]]] = {}
        for sent in root.iter():
            if local_name(sent.tag) != "s":
                continue
            sent_id = sent.attrib.get("n")
            if not sent_id:
                continue
            pieces: List[str] = []
            word_tokens: List[str] = []
            for node in sent.iter():
                name = local_name(node.tag)
                if name not in {"w", "c"}:
                    continue
                tok = norm_space("".join(node.itertext()))
                if not tok:
                    continue
                pieces.append(tok)
                if name == "w":
                    word_tokens.append(tok)
            sent_map[sent_id] = (detokenize(pieces), word_tokens)
        self.cache[xml_path] = sent_map
        return sent_map

    def _parse_reference(self, reference: str) -> Optional[Tuple[Path, str]]:
        parts = [p for p in reference.strip().split("/") if p]
        if len(parts) < 3:
            return None
        xml_path = self.texts_root.joinpath(*parts[:-1])
        sent_id = parts[-1]
        return xml_path, sent_id

    def resolve(self, reference: str) -> Optional[Tuple[str, List[str]]]:
        parsed = self._parse_reference(reference)
        if not parsed:
            return None
        xml_path, sent_id = parsed
        if not xml_path.exists():
            return None
        sent_map = self._load_file(xml_path)
        return sent_map.get(sent_id)

    def resolve_many(self, references: Sequence[str]) -> Dict[str, Tuple[str, List[str]]]:
        grouped: Dict[Path, set[str]] = {}
        ref_keys: Dict[Tuple[Path, str], str] = {}
        out: Dict[str, Tuple[str, List[str]]] = {}
        for reference in references:
            parsed = self._parse_reference(reference)
            if not parsed:
                continue
            xml_path, sent_id = parsed
            if not xml_path.exists():
                continue
            grouped.setdefault(xml_path, set()).add(sent_id)
            ref_keys[(xml_path, sent_id)] = reference

        jobs: Dict[object, Tuple[Path, Sequence[str]]] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(grouped)))) as executor:
            for xml_path, wanted_ids in grouped.items():
                cached = self.cache.get(xml_path, {})
                missing_ids = sorted(sent_id for sent_id in wanted_ids if sent_id not in cached)
                if not missing_ids:
                    continue
                future = executor.submit(resolve_bnc_xml_subset, xml_path, tuple(missing_ids))
                jobs[future] = (xml_path, tuple(missing_ids))

            if jobs:
                print(f"[INFO] Resolving {len(jobs)} BNC XML files with {min(self.max_workers, len(jobs))} workers.")
                for future in tqdm(
                    as_completed(jobs),
                    total=len(jobs),
                    desc="Resolving BNC XML files",
                    unit="file",
                ):
                    xml_path, _missing_ids = jobs[future]
                    found_map = future.result()
                    cache_map = self.cache.setdefault(xml_path, {})
                    cache_map.update(found_map)

        for xml_path, wanted_ids in grouped.items():
            sent_map = self.cache.get(xml_path, {})
            for sent_id in wanted_ids:
                value = sent_map.get(sent_id)
                if value is not None:
                    out[ref_keys[(xml_path, sent_id)]] = value
        return out


def parse_bool_label(value: str) -> bool:
    value = norm_space(value).lower()
    return value in {"1", "true", "t", "yes", "y", "positive", "pos", "+"}


def parse_index_field(value: str) -> List[int]:
    value = norm_space(value)
    if not value:
        return []
    indices: List[int] = []
    for chunk in re.split(r"[;,/ ]+", value):
        if not chunk:
            continue
        if re.fullmatch(r"\d+-\d+", chunk):
            start, end = chunk.split("-", 1)
            a, b = int(start), int(end)
            if a <= b:
                indices.extend(range(a, b + 1))
            else:
                indices.extend(range(b, a + 1))
        elif chunk.isdigit():
            indices.append(int(chunk))
    return indices


def normalize_candidate(text: str) -> str:
    return clean_surface(text.replace("_", " "))


def extract_candidate_from_indices(tokens: Sequence[str], index_field: str) -> str:
    indices = parse_index_field(index_field)
    chosen = [tokens[i - 1] for i in indices if 1 <= i <= len(tokens)]
    return clean_surface(" ".join(chosen))


def find_subsequence(tokens: Sequence[str], pattern: Sequence[str]) -> Optional[int]:
    if not pattern or len(pattern) > len(tokens):
        return None
    for start in range(len(tokens) - len(pattern) + 1):
        if list(tokens[start : start + len(pattern)]) == list(pattern):
            return start
    return None


def extract_vpc_candidate(tokens: Sequence[str], index_field: str, candidate_hint: str) -> str:
    span_candidate = extract_candidate_from_indices(tokens, index_field)
    hint = normalize_candidate(candidate_hint)
    if not hint:
        return span_candidate

    raw_parts = [norm_space(part) for part in str(candidate_hint).split("_") if norm_space(part)]
    if len(raw_parts) <= 1:
        return span_candidate or hint

    particle_parts = [normalize_candidate(part) for part in raw_parts[1:] if normalize_candidate(part)]
    if not particle_parts:
        return span_candidate or hint

    indices = [i for i in parse_index_field(index_field) if 1 <= i <= len(tokens)]
    span_tokens = [tokens[i - 1] for i in indices]
    if not span_tokens:
        return hint

    span_lower = [norm_space(tok).lower() for tok in span_tokens]
    particle_lower = [part.lower() for part in particle_parts]
    particle_start = find_subsequence(span_lower, particle_lower)
    if particle_start is None and particle_lower:
        try:
            particle_start = span_lower.index(particle_lower[0])
        except ValueError:
            particle_start = None

    if particle_start is not None:
        if particle_start > 0:
            lexical_verb = span_tokens[particle_start - 1]
        else:
            prev_token_index = indices[0] - 1
            lexical_verb = tokens[prev_token_index - 1] if 1 <= prev_token_index <= len(tokens) else span_tokens[0]
    else:
        # Some spans contain an auxiliary+verb pair like "had taken";
        # use the rightmost token from the indexed span as the inflected lexical verb.
        lexical_verb = span_tokens[-1]

    return clean_surface(" ".join([lexical_verb, *particle_parts]))


def find_first_existing(base: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pattern in patterns:
        direct = base / pattern
        if direct.exists():
            return direct
        hits = sorted(base.rglob(pattern))
        if hits:
            return hits[0]
    return None


def find_dataset_file(root: Path, kind: str) -> Path:
    ranked: List[Tuple[int, Path]] = []
    for path in root.rglob("*.txt"):
        low = path.name.lower()
        score = 0
        if kind == "lvc" and "lvc" in low:
            score += 10
        if kind == "vpc" and ("pvc" in low or "vpc" in low):
            score += 10
        try:
            with path.open(encoding="utf-8", errors="ignore") as f:
                preview = [next(f, "") for _ in range(5)]
        except OSError:
            continue
        tab_lines = sum(1 for line in preview if line.count("\t") >= 3)
        score += tab_lines
        if score > 0:
            ranked.append((score, path))
    if ranked:
        ranked.sort(key=lambda item: (-item[0], str(item[1])))
        return ranked[0][1]
    raise FileNotFoundError(f"Could not find a text dataset file for {kind} in {root}")


def parse_lvc_or_vpc_dataset(dataset_path: Path, resolver: BNCSentenceResolver, candidate_col: str) -> pd.DataFrame:
    pending: List[Tuple[str, str, str]] = []
    with dataset_path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            if parts[0].lower().startswith("reference"):
                continue
            reference = parts[0].strip()
            if len(parts) == 3:
                candidate_hint = parts[1].strip()
                label = parts[2].strip()
                index_field = ""
            else:
                label = parts[2].strip() if len(parts) >= 3 else ""
                candidate_hint = parts[3].strip() if len(parts) >= 4 else ""
                index_field = parts[4].strip() if len(parts) >= 5 else ""
            if not parse_bool_label(label):
                continue
            pending.append((reference, candidate_hint, index_field))

    print(f"[INFO] {candidate_col}: resolving {len(pending)} positive samples from BNC. Please be patient; BNC is large.")
    resolved = resolver.resolve_many([reference for reference, _candidate_hint, _index_field in pending])
    rows: List[Dict[str, str]] = []
    for reference, candidate_hint, index_field in tqdm(
        pending,
        desc=f"Building {candidate_col}",
        unit="row",
    ):
        sent_info = resolved.get(reference)
        if not sent_info:
            continue
        src_text, word_tokens = sent_info
        if candidate_col == "VPC_candidate":
            candidate = extract_vpc_candidate(word_tokens, index_field, candidate_hint)
        else:
            candidate = extract_candidate_from_indices(word_tokens, index_field) or normalize_candidate(candidate_hint)
        if not src_text or not candidate:
            continue
        rows.append({"src": src_text, candidate_col: candidate})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(ignore_index=True)


def ensure_nltk():
    try:
        import nltk  # type: ignore
        from nltk import data  # type: ignore
    except ImportError:
        return None

    needed = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for resource_path, download_name in needed:
        try:
            data.find(resource_path)
        except LookupError:
            try:
                ok = nltk.download(download_name, quiet=True)
            except Exception:
                return None
            if not ok:
                return None
            try:
                data.find(resource_path)
            except LookupError:
                return None
    return nltk


COMMON_VERBS = {
    "be", "am", "is", "are", "was", "were", "been", "being", "have", "has", "had", "do", "does", "did",
    "get", "gets", "got", "gotten", "give", "gives", "gave", "given", "make", "makes", "made",
    "take", "takes", "took", "taken", "come", "comes", "came", "go", "goes", "went", "gone",
    "keep", "keeps", "kept", "let", "run", "runs", "ran", "put", "puts", "set", "sets", "fall", "falls",
    "pull", "push", "look", "looks", "looked", "hold", "holds", "held", "turn", "turns", "turned",
}


def contains_verb(text: str) -> bool:
    nltk = ensure_nltk()
    if nltk is not None:
        try:
            tokens = nltk.word_tokenize(text)
            tags = nltk.pos_tag(tokens)
            return any(tag.startswith("VB") for _, tag in tags)
        except Exception:
            pass
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())
    return any(tok in COMMON_VERBS or tok.endswith(("ed", "ing")) for tok in tokens)


def load_nonempty_lines(path: Path) -> List[str]:
    with path.open(encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f if line.strip()]


def sample_frame(df: pd.DataFrame, n: int, seed: Optional[int]) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    if seed is None:
        return df.sample(n=n).reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def build_vid_from_epie(root: Path, sample_size: int, seed: Optional[int]) -> pd.DataFrame:
    candidates_file = find_first_existing(root, ["Formal_Idioms_Candidates.txt", "*Formal*Candidates*.txt"])
    words_file = find_first_existing(root, ["Formal_Idioms_Words.txt", "*Formal*Words*.txt"])
    labels_file = find_first_existing(root, ["Formal_Idioms_Labels.txt", "*Formal*Labels*.txt"])
    if not candidates_file or not words_file or not labels_file:
        raise FileNotFoundError("Could not find aligned EPIE formal idiom files.")

    candidates = load_nonempty_lines(candidates_file)
    sentences = load_nonempty_lines(words_file)
    labels = load_nonempty_lines(labels_file)

    rows: List[Dict[str, str]] = []
    for idx, candidate in enumerate(candidates):
        if idx >= len(sentences) or idx >= len(labels):
            break
        if labels[idx] != "1":
            continue
        if not contains_verb(candidate):
            continue
        rows.append({"src": sentences[idx], "VID_candidate": norm_space(candidate)})

    df = pd.DataFrame(rows).drop_duplicates(ignore_index=True)
    return sample_frame(df, sample_size, seed)


def normalize_context(context) -> List[str]:
    if isinstance(context, list):
        return [norm_space(str(x)) for x in context if norm_space(str(x))]
    if isinstance(context, str):
        return [norm_space(context)] if norm_space(context) else []
    return []


def build_vid_from_magpie(root: Path, sample_size: int, seed: Optional[int]) -> pd.DataFrame:
    data_file = find_first_existing(root, ["MAGPIE_unfiltered.jsonl", "*.jsonl"])
    if not data_file:
        raise FileNotFoundError("Could not find a MAGPIE JSONL file.")

    rows: List[Dict[str, str]] = []
    with data_file.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            idiom = norm_space(str(obj.get("idiom", "")))
            label = norm_space(str(obj.get("label", ""))).lower()
            if label != "i" or not idiom or not contains_verb(idiom):
                continue
            contexts = normalize_context(obj.get("context", []))
            matched = next((sent for sent in contexts if idiom.lower() in sent.lower()), None)
            if not matched:
                continue
            rows.append({"src": matched, "VID_candidate": idiom})

    df = pd.DataFrame(rows).drop_duplicates(ignore_index=True)
    return sample_frame(df, sample_size, seed)


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


def extract_sentences_with_length(xml_file: Path) -> List[str]:
    sentences: List[str] = []
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for sent in root.iter():
            if local_name(sent.tag) != "s":
                continue
            text = " ".join(" ".join(sent.itertext()).strip().split())
            text = re.sub(r'\s+([.,;:!?)\]}"\'»])', r"\1", text)
            text = re.sub(r'([«\[({"\'])\s+', r"\1", text)
            if not text:
                continue
            word_count = len(text.split())
            if MIN_WORDS <= word_count <= MAX_WORDS:
                sentences.append(text)
    except ET.ParseError:
        return []
    return sentences


def extract_sentences_from_corpus(root_dir: Path) -> List[str]:
    xml_files = sorted(root_dir.rglob("*.xml"))
    corpus_sentences: List[str] = []
    for xml_file in tqdm(xml_files, desc="Processing BNC XML for contrast", unit="file"):
        corpus_sentences.extend(extract_sentences_with_length(xml_file))
    return corpus_sentences


def compile_idiom_patterns(idioms: Sequence[str]) -> List[Tuple[str, re.Pattern[str]]]:
    idiom_patterns = []
    pronoun_pattern = r"(?:I|you|he|she|it|we|they|my|your|his|her|its|our|their)"
    for idiom in idioms:
        if not idiom:
            continue
        if "[pron]" in idiom:
            parts = idiom.split("[pron]")
            escaped_parts = [re.escape(part) for part in parts]
            pattern_str = pronoun_pattern.join(escaped_parts)
            pattern = re.compile(r"\b" + pattern_str + r"\b", re.IGNORECASE)
        else:
            pattern = re.compile(r"\b" + re.escape(idiom) + r"\b", re.IGNORECASE)
        idiom_patterns.append((idiom, pattern))
    return idiom_patterns


def filter_idioms(sentences: Sequence[str], idioms: Sequence[str]) -> List[str]:
    patterns = compile_idiom_patterns(idioms)
    filtered = []
    for sentence in tqdm(sentences, desc="Filtering idioms for contrast", unit="sent"):
        for _idiom, pattern in patterns:
            if pattern.search(sentence):
                filtered.append(sentence)
                break
    return filtered


def filter_light_verbs(sentences: Sequence[str]) -> List[str]:
    nlp = load_spacy()
    filtered = []
    for doc in tqdm(nlp.pipe(sentences, batch_size=256), total=len(sentences), desc="Filtering light verbs for contrast"):
        if any(token.lemma_.lower() in LIGHT_VERBS and token.pos_ == "VERB" for token in doc):
            filtered.append(doc.text)
    return filtered


def filter_verb_prep(sentences: Sequence[str]) -> List[str]:
    nlp = load_spacy()
    filtered = []
    for doc in tqdm(nlp.pipe(sentences, batch_size=256), total=len(sentences), desc="Filtering verb+preposition for contrast"):
        found = False
        for token in doc:
            if token.pos_ == "VERB":
                for child in token.children:
                    if child.dep_ == "prt" and child.pos_ == "ADP":
                        filtered.append(doc.text)
                        found = True
                        break
            if found:
                break
    return filtered


def build_non_vmwe_contrast(bnc_root: Path, idioms: Sequence[str], sample_size: int, seed: Optional[int]) -> pd.DataFrame:
    if seed is not None:
        random.seed(seed)
    texts_root = BNCSentenceResolver(bnc_root).texts_root
    all_sentences = extract_sentences_from_corpus(texts_root)
    if not all_sentences:
        return pd.DataFrame(columns=["src"])

    chunk_sentences = random.sample(all_sentences, min(100000, len(all_sentences)))
    idiom_sentences = set(filter_idioms(chunk_sentences, idioms))
    light_verb_sentences = set(filter_light_verbs(chunk_sentences))
    verb_prep_sentences = set(filter_verb_prep(chunk_sentences))
    exclude = idiom_sentences | light_verb_sentences | verb_prep_sentences
    normal_baseline = [sentence for sentence in chunk_sentences if sentence not in exclude]
    sampled = random.sample(normal_baseline, min(sample_size, len(normal_baseline))) if normal_baseline else []
    return pd.DataFrame({"src": sampled})


def write_csv(path: Path, df: pd.DataFrame, columns: Sequence[str]) -> Path:
    mkdir(path.parent)
    df = df.loc[:, list(columns)].copy()
    df.to_csv(path, index=False)
    return path


def build_datasets(raw_root: Path, out_root: Path, skip_download: bool, force: bool, sample_size: int, seed: Optional[int]) -> Dict[str, Path]:
    sources = ensure_sources(raw_root=raw_root, skip_download=skip_download, force=force)

    resolver = BNCSentenceResolver(sources["bnc"])
    lvc_dataset_file = find_dataset_file(sources["lvc"], "lvc")
    vpc_dataset_file = find_dataset_file(sources["vpc"], "vpc")

    lvc_df = parse_lvc_or_vpc_dataset(lvc_dataset_file, resolver, "LVC_candidate")
    vpc_df = parse_lvc_or_vpc_dataset(vpc_dataset_file, resolver, "VPC_candidate")
    epie_df = build_vid_from_epie(sources["epie"], sample_size=sample_size, seed=seed)
    magpie_df = build_vid_from_magpie(sources["magpie"], sample_size=sample_size, seed=seed)
    vid_df = pd.concat([epie_df, magpie_df], ignore_index=True).drop_duplicates(ignore_index=True)
    vid_dictionary = pd.DataFrame({"VID_candidate": sorted(set(vid_df["VID_candidate"].astype(str)))})
    normal_df = build_non_vmwe_contrast(sources["bnc"], vid_dictionary["VID_candidate"].tolist(), sample_size=2000, seed=seed)

    outputs = {
        "LVC": write_csv(out_root / "LVC.csv", lvc_df, ["src", "LVC_candidate"]),
        "VPC": write_csv(out_root / "VPC.csv", vpc_df, ["src", "VPC_candidate"]),
        "VID": write_csv(out_root / "VID.csv", vid_df, ["src", "VID_candidate"]),
        "VID_dictionary": write_csv(out_root / "VID_dictionary.csv", vid_dictionary, ["VID_candidate"]),
        "Non_VMWE": write_csv(out_root / "Non_VMWE.csv", normal_df, ["src"]),
    }
    with (out_root / "VID_dictionary.txt").open("w", encoding="utf-8") as f:
        for idiom in vid_dictionary["VID_candidate"].tolist():
            f.write(f"{idiom}\n")
    return outputs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download, extract, and construct VMWE datasets.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT, help=f"Raw download/extraction directory. Default: {DEFAULT_RAW_ROOT}")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help=f"Output dataset directory. Default: {DEFAULT_OUT_ROOT}")
    parser.add_argument("--skip-download", action="store_true", help="Use already downloaded archives/extracted trees.")
    parser.add_argument("--force-download", action="store_true", help="Redownload archives even if already present.")
    parser.add_argument("--vid-sample-size", type=int, default=1000, help="Sample size per VID corpus (EPIE and MAGPIE).")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed. Default matches the original scripts: no fixed seed.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    outputs = build_datasets(
        raw_root=args.raw_root,
        out_root=args.out_root,
        skip_download=args.skip_download,
        force=args.force_download,
        sample_size=args.vid_sample_size,
        seed=args.seed,
    )
    for name, path in outputs.items():
        print(f"[INFO] Wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
