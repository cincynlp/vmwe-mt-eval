#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import html
import io
import json
import os
import re
import shutil
import sys
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

import pandas as pd
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = ROOT / "raw_dataset"
DEFAULT_OUT_ROOT = ROOT / "datasets" / "WMT"

ALLOWED_YEARS = tuple(range(2017, 2025))
TARGET_TGT_LANGS = ("cs", "de", "ru", "zh")
TARGET_PAIRS = {("en", lang) for lang in TARGET_TGT_LANGS}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    kind: str
    url: str
    archive_name: str
    aliases: Tuple[str, ...]


YEAR_SOURCES: Dict[int, Tuple[SourceSpec, ...]] = {
    2017: (
        SourceSpec(
            name="submission",
            kind="direct",
            url="http://data.statmt.org/wmt17/translation-task/wmt17-submitted-data-v1.0.tgz",
            archive_name="wmt17-submitted-data-v1.0.tgz",
            aliases=("wmt17-submitted-data",),
        ),
        SourceSpec(
            name="human_eval",
            kind="direct",
            url="http://www.scss.tcd.ie/~ygraham/newstest2017-system-level-human.tar.gz",
            archive_name="newstest2017-system-level-human.tar.gz",
            aliases=("newstest2017-system-level-human",),
        ),
    ),
    2018: (
        SourceSpec(
            name="submission",
            kind="direct",
            url="http://data.statmt.org/wmt18/translation-task/wmt18-submitted-data-v1.0.1.tgz",
            archive_name="wmt18-submitted-data-v1.0.1.tgz",
            aliases=("wmt18-submitted-data",),
        ),
        SourceSpec(
            name="human_eval",
            kind="direct",
            url="http://www.scss.tcd.ie/~ygraham/newstest2018-humaneval.tar.gz",
            archive_name="newstest2018-humaneval.tar.gz",
            aliases=("newstest2018-humaneval",),
        ),
    ),
    2019: (
        SourceSpec(
            name="submission",
            kind="direct",
            url="http://data.statmt.org/wmt19/translation-task/wmt19-submitted-data-v3.tgz",
            archive_name="wmt19-submitted-data-v3.tgz",
            aliases=("wmt19-submitted-data",),
        ),
        SourceSpec(
            name="human_eval",
            kind="direct",
            url="https://www.scss.tcd.ie/~ygraham/newstest2019-humaneval.tar.gz",
            archive_name="newstest2019-humaneval.tar.gz",
            aliases=("newstest2019-humaneval",),
        ),
    ),
    2020: (
        SourceSpec(
            name="submission",
            kind="gdrive",
            url="https://drive.google.com/file/d/1v6G89fdUitX5c_tWBtev4iRd8nyaCYY_/view?usp=sharing",
            archive_name="newstest2020sgm-v2.tar.gz",
            aliases=("newstest2020sgm-v2", "wmt20-news-task-primary-submissions", "newstest2020"),
        ),
        SourceSpec(
            name="human_eval",
            kind="github",
            url="https://github.com/WMT-Metrics-task/wmt20-metrics/tree/main/manual-evaluation/DA",
            archive_name="wmt20-metrics-main.tar.gz",
            aliases=("wmt20-metrics-main", "WMT20_data"),
        ),
    ),
    2021: (
        SourceSpec(
            name="repo",
            kind="github",
            url="https://github.com/wmt-conference/wmt21-news-systems/tree/main/xml",
            archive_name="wmt21-news-systems-main.tar.gz",
            aliases=("wmt21-news-systems-main",),
        ),
    ),
    2022: (
        SourceSpec(
            name="repo",
            kind="github",
            url="https://github.com/wmt-conference/wmt22-news-systems/tree/main/xml",
            archive_name="wmt22-news-systems-main.tar.gz",
            aliases=("wmt22-news-systems-main",),
        ),
    ),
    2023: (
        SourceSpec(
            name="repo",
            kind="github",
            url="https://github.com/wmt-conference/wmt23-news-systems/tree/master/xml",
            archive_name="wmt23-news-systems-master.tar.gz",
            aliases=("wmt23-news-systems-master",),
        ),
    ),
    2024: (
        SourceSpec(
            name="repo",
            kind="github",
            url="https://github.com/wmt-conference/wmt24-news-systems/tree/main/xml",
            archive_name="wmt24-news-systems-main.tar.gz",
            aliases=("wmt24-news-systems-main",),
        ),
    ),
}


def norm_text(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("\u00A0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_system_type(system: str) -> str:
    """Classify a system name as 'Human' or 'MT'."""
    sys_lower = str(system).lower()
    human_indicators = [
        "human", "ref", "translator", "reference", "refa", "refb",
        "translator-a", "translator-b", "translator-c", "human-refa"
    ]
    if any(ind in sys_lower for ind in human_indicators):
        return "Human"
    return "MT"


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())


def token_set(text: str) -> set[str]:
    return set(t for t in re.split(r"[^A-Za-z0-9]+", (text or "").lower()) if t)


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    mkdir(dest_dir)
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            member_path = dest_dir / member.name
            resolved = member_path.resolve()
            if not str(resolved).startswith(str(dest_dir.resolve())):
                raise RuntimeError(f"Refusing to extract path outside destination: {member.name}")
        tar.extractall(dest_dir, filter="data")


def parse_github_repo_url(url: str) -> Tuple[str, str, str]:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Unsupported GitHub URL: {url}")
    owner, repo = parts[0], parts[1]
    branch = "main"
    if len(parts) >= 4 and parts[2] == "tree":
        branch = parts[3]
    return owner, repo, branch


def import_requests():
    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Downloading requires the 'requests' package. Install it or rerun with --skip-download."
        ) from exc
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


def extract_gdrive_file_id(url: str) -> str:
    m = re.search(r"/file/d/([^/]+)", url)
    if m:
        return m.group(1)
    parsed = urlparse(url)
    q = parse_qs(parsed.query)
    if "id" in q and q["id"]:
        return q["id"][0]
    raise ValueError(f"Could not parse Google Drive file id from {url}")


def download_gdrive(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        return dest
    requests = import_requests()
    file_id = extract_gdrive_file_id(url)
    mkdir(dest.parent)
    session = requests.Session()
    params = {"id": file_id, "export": "download"}
    resp = session.get("https://drive.google.com/uc", params=params, stream=True, timeout=120)
    resp.raise_for_status()

    confirm_token = None
    for key, value in resp.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break
    if confirm_token:
        resp.close()
        params["confirm"] = confirm_token
        resp = session.get("https://drive.google.com/uc", params=params, stream=True, timeout=120)
        resp.raise_for_status()

    first_chunk = next(resp.iter_content(chunk_size=1024 * 1024), b"")
    if first_chunk.lstrip().startswith(b"<!DOCTYPE html") or b"Virus scan warning" in first_chunk:
        html_text = first_chunk.decode("utf-8", errors="ignore")
        action_match = re.search(r'action="([^"]+)"', html_text)
        hidden_pairs = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)">', html_text))
        if not action_match or not hidden_pairs:
            raise RuntimeError("Google Drive download returned an HTML interstitial that could not be parsed.")
        resp.close()
        download_url = action_match.group(1)
        resp = session.get(download_url, params=hidden_pairs, stream=True, timeout=120)
        resp.raise_for_status()
        first_chunk = next(resp.iter_content(chunk_size=1024 * 1024), b"")

    with dest.open("wb") as fout:
        if first_chunk:
            fout.write(first_chunk)
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fout.write(chunk)
    return dest


def download_github(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        return dest
    owner, repo, branch = parse_github_repo_url(url)
    archive_url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/refs/heads/{branch}"
    return download_http(archive_url, dest, force=force)


def extract_if_needed(archive_path: Path, extract_dir: Path) -> Path:
    if extract_dir.exists() and any(extract_dir.iterdir()):
        return extract_dir
    safe_extract_tar(archive_path, extract_dir)
    return extract_dir


def find_alias_root(search_roots: Sequence[Path], aliases: Sequence[str]) -> Optional[Path]:
    for base in search_roots:
        if not base.exists():
            continue
        for alias in aliases:
            direct = base / alias
            if direct.exists():
                return direct
        for alias in aliases:
            hits = sorted(base.rglob(alias))
            if hits:
                return hits[0]
    return None


def resolve_source_root(year: int, spec: SourceSpec, raw_root: Path) -> Path:
    extracted_base = raw_root / str(year) / "extracted" / spec.name
    search_roots = [extracted_base, ROOT]
    found = find_alias_root(search_roots, spec.aliases)
    if not found:
        raise FileNotFoundError(
            f"Could not locate extracted data for year {year} source '{spec.name}'. "
            f"Looked for aliases: {spec.aliases}"
        )
    return found


def ensure_year_downloads(year: int, raw_root: Path, skip_download: bool, force: bool) -> Dict[str, Path]:
    year_dir = raw_root / str(year)
    downloads_dir = mkdir(year_dir / "downloads")
    extracted_dir = mkdir(year_dir / "extracted")

    resolved: Dict[str, Path] = {}
    for spec in YEAR_SOURCES[year]:
        archive_path = downloads_dir / spec.archive_name
        if not skip_download:
            if spec.kind == "direct":
                download_http(spec.url, archive_path, force=force)
            elif spec.kind == "gdrive":
                download_gdrive(spec.url, archive_path, force=force)
            elif spec.kind == "github":
                download_github(spec.url, archive_path, force=force)
            else:
                raise ValueError(f"Unsupported source kind: {spec.kind}")
            extract_if_needed(archive_path, extracted_dir / spec.name)
        resolved[spec.name] = resolve_source_root(year, spec, raw_root)
    return resolved


def read_text_lines(path: Path) -> List[str]:
    with path.open(encoding="utf-8", errors="ignore") as f:
        return [norm_text(line.rstrip("\n")) for line in f]


DOC_BLOCK_RE = re.compile(r"<doc\b[^>]*\bdocid=\"([^\"]+)\"[^>]*>(.*?)</doc>", re.I | re.S)
SEG_RE = re.compile(r"<seg\b[^>]*\bid=\"(\d+)\"[^>]*>(.*?)</seg>", re.I | re.S)


def strip_tags(text: str) -> str:
    return norm_text(re.sub(r"<[^>]+>", "", text or ""))


def parse_sgm_docseg(path: Path) -> Dict[Tuple[str, int], str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    out: Dict[Tuple[str, int], str] = {}
    for docid, block in DOC_BLOCK_RE.findall(text):
        segs = []
        for sid_raw, seg_content in SEG_RE.findall(block):
            try:
                seg_id = int(sid_raw)
            except ValueError:
                continue
            segs.append((seg_id, strip_tags(seg_content)))
        for seg_id, seg_text in sorted(segs, key=lambda x: x[0]):
            out[(docid, seg_id)] = seg_text
    return out


def parse_sgm_flat(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    flat: List[str] = []
    for _docid, block in DOC_BLOCK_RE.findall(text):
        segs = []
        for sid_raw, seg_content in SEG_RE.findall(block):
            try:
                seg_id = int(sid_raw)
            except ValueError:
                continue
            segs.append((seg_id, strip_tags(seg_content)))
        flat.extend(seg_text for _seg_id, seg_text in sorted(segs, key=lambda x: x[0]))
    return flat


def parse_plain_mt(path: Path) -> List[str]:
    with path.open(encoding="utf-8", errors="ignore") as f:
        return [norm_text(line.rstrip("\n")) for line in f]


def choose_best_numeric_candidate(system_name: str, needed_sids: Sequence[int], candidates: Dict[Path, List[str]]) -> Optional[Path]:
    sys_tokens = token_set(system_name)
    best_path: Optional[Path] = None
    best_cov = -1
    best_sim = -1.0
    for path, lines in candidates.items():
        cov = sum(1 for sid in needed_sids if 1 <= sid <= len(lines))
        name_tokens = token_set(path.name)
        sim = len(sys_tokens & name_tokens) / (len(sys_tokens | name_tokens) or 1)
        if cov > best_cov or (cov == best_cov and sim > best_sim):
            best_path = path
            best_cov = cov
            best_sim = sim
    return best_path


def choose_best_docseg_candidate(
    system_name: str,
    needed_keys: Sequence[Tuple[str, int]],
    candidates: Dict[Path, Dict[Tuple[str, int], str]],
) -> Optional[Path]:
    sys_tokens = token_set(system_name)
    best_path: Optional[Path] = None
    best_cov = -1
    best_sim = -1.0
    for path, mapping in candidates.items():
        cov = sum(1 for key in needed_keys if key in mapping)
        name_tokens = token_set(path.name)
        sim = len(sys_tokens & name_tokens) / (len(sys_tokens | name_tokens) or 1)
        if cov > best_cov or (cov == best_cov and sim > best_sim):
            best_path = path
            best_cov = cov
            best_sim = sim
    return best_path


def write_rows(out_root: Path, year: int, pair: str, system: str, rows: List[Dict[str, object]]) -> Optional[Path]:
    if not rows:
        return None
    sys_type = get_system_type(system)
    target_dir = mkdir(out_root / str(year) / sys_type / pair)
    out_path = target_dir / f"{sanitize_filename(system)}.csv"
    pd.DataFrame(rows, columns=["src", "mt", "ref", "score"]).to_csv(out_path, index=False)
    return out_path


def collapse_segment_duplicates(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    text_cols: Sequence[str] = ("src", "mt", "ref"),
    score_col: str = "score",
) -> pd.DataFrame:
    agg_spec = {score_col: "mean"}
    for col in text_cols:
        agg_spec[col] = "first"
    return df.groupby(list(group_cols), dropna=False, sort=False).agg(agg_spec).reset_index()


def find_first_existing(base: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pattern in patterns:
        direct = base / pattern
        if direct.exists():
            return direct
        hits = sorted(base.rglob(pattern))
        if hits:
            return hits[0]
    return None


def process_2017_2018(year: int, sources: Dict[str, Path], out_root: Path) -> List[Dict[str, object]]:
    if year == 2017:
        langs = ["cs", "de", "ru", "zh"]
    else:
        langs = ["cs", "de", "ru", "zh"]

    # Look for the raw "good-stnd-redup" files first for BOTH years, then fallback to "seg-scores"
    eval_patterns = [
        "anon-proc-hits-sys-combined/analysis/ad-en{lang}-good-stnd-redup.csv.gz",
        "proc-hits-sys-appraise/analysis/ad-en{lang}-good-stnd-redup.csv.gz",
        "analysis/ad-en{lang}-good-stnd-redup.csv",         # 2018 raw file pattern
        "analysis/ad-en{lang}-good-stnd-redup.csv.gz",      # 2018 raw file compressed
        "analysis/ad-seg-scores-en-{lang}.csv",             # Fallback to seg-scores
        "anon-proc-hits-sys-combined/analysis/ad-seg-scores-en-{lang}.csv.gz",
    ]

    submission_root = sources["submission"]
    eval_root = sources["human_eval"]
    year_logs: List[Dict[str, object]] = []

    for lang in langs:
        src_file = find_first_existing(
            submission_root,
            [
                f"txt/sources/newstest{year}-en{lang}-src.en",
                f"sgm/sources/newstest{year}-en{lang}-src.en.sgm",
            ],
        )
        ref_file = find_first_existing(
            submission_root,
            [
                f"sgm/references/newstest{year}-en{lang}-ref.{lang}.sgm",
                f"txt/references/newstest{year}-en{lang}-ref.{lang}",
            ],
        )
        eval_file = find_first_existing(eval_root, [p.format(lang=lang) for p in eval_patterns])
        
        if not src_file or not ref_file or not eval_file:
            print(f"[WARN] {year} en-{lang}: missing src/ref/eval; skipping.")
            continue

        src_lines = parse_sgm_flat(src_file) if src_file.suffix == ".sgm" else read_text_lines(src_file)
        ref_lines = parse_sgm_flat(ref_file) if ref_file.suffix == ".sgm" else read_text_lines(ref_file)
        
        df = pd.read_csv(eval_file, sep=r"\s+", engine="python", compression="infer")
        
        # Detect column structure automatically based on which file we found
        if "sys_id" in df.columns:
            sys_col, sid_col, score_col = "sys_id", "sid", "score"
        elif "SYS" in df.columns:
            sys_col, sid_col, score_col = "SYS", "SID", "Z.SCR"
        else:
            df.columns = ["hit_id", "worker", "src_lang", "trg_lang", "type", "sid", "sys_id", "na", "ref_type", "score", "raw_score"]
            sys_col, sid_col, score_col = "sys_id", "sid", "score"

        # Strip whitespace to catch "REFERENCE "
        df["system"] = df[sys_col].astype(str).str.strip().apply(lambda x: x.split(".")[0])

        candidate_files = sorted(
            p for p in submission_root.rglob(f"newstest{year}*.en-{lang}*")
            if p.is_file() and "system-outputs" in p.parts and ".history" not in p.name
        )
        numeric_candidates: Dict[Path, List[str]] = {}
        for path in candidate_files:
            try:
                numeric_candidates[path] = parse_sgm_flat(path) if path.suffix == ".sgm" else parse_plain_mt(path)
            except Exception:
                continue

        for system, group in df.groupby("system", sort=False):
            rows: List[Dict[str, object]] = []
            needed_sids = [int(v) for v in group[sid_col].tolist() if pd.notna(v)]
            chosen = choose_best_numeric_candidate(system, needed_sids, numeric_candidates)
            
            if chosen:
                mt_lines = numeric_candidates[chosen]
            elif get_system_type(system) == "Human":
                mt_lines = ref_lines
            else:
                mt_lines = []
                
            for _, row in group.iterrows():
                sid = int(row[sid_col])
                src = src_lines[sid - 1] if 1 <= sid <= len(src_lines) else ""
                ref = ref_lines[sid - 1] if 1 <= sid <= len(ref_lines) else ""
                mt = mt_lines[sid - 1] if 1 <= sid <= len(mt_lines) else ""
                if not str(mt).strip():
                    continue
                rows.append({"src": src, "mt": mt, "ref": ref, "score": float(row[score_col])})
                
            # Guarantee the output file is named REFERENCE.csv regardless of what the WMT file called it
            output_system_name = "REFERENCE" if get_system_type(system) == "Human" else system

            out_path = write_rows(out_root, year, f"en-{lang}", output_system_name, rows)
            if out_path:
                year_logs.append({"year": year, "pair": f"en-{lang}", "system": output_system_name, "rows": len(rows), "path": str(out_path)})
                
    return year_logs


def process_2019(sources: Dict[str, Path], out_root: Path) -> List[Dict[str, object]]:
    submission_root = sources["submission"]
    eval_root = sources["human_eval"]
    logs: List[Dict[str, object]] = []

    for tgt_lang in ["cs", "de", "ru", "zh"]:
        pair = f"en-{tgt_lang}"
        eval_file = find_first_existing(eval_root, [f"appraise-doclevel-humaneval-newstest2019/analysis/ad-seg-scores-{pair}.csv", f"analysis/ad-seg-scores-{pair}.csv"])
        src_sgm = find_first_existing(submission_root, [f"sgm/sources/newstest2019-en{tgt_lang}-src.en.sgm"])
        ref_sgm = find_first_existing(submission_root, [f"sgm/references/newstest2019-en{tgt_lang}-ref.{tgt_lang}.sgm"])
        if not eval_file or not src_sgm or not ref_sgm:
            print(f"[WARN] 2019 {pair}: missing segment score file or source/reference SGM.")
            continue

        src_lines = parse_sgm_flat(src_sgm)
        ref_lines = parse_sgm_flat(ref_sgm)
        df = pd.read_csv(eval_file, sep=r"\s+", engine="python")
        df["system"] = df["SYS"].astype(str).apply(lambda x: x if x == "REFERENCE" else x.split(".")[0])

        candidate_files = sorted(
            p for p in submission_root.rglob(f"newstest2019*.en-{tgt_lang}*")
            if p.is_file() and "system-outputs" in p.parts and ".history" not in p.name
        )
        numeric_candidates: Dict[Path, List[str]] = {}
        for path in candidate_files:
            try:
                numeric_candidates[path] = parse_sgm_flat(path) if path.suffix == ".sgm" else parse_plain_mt(path)
            except Exception:
                continue

        for system, group in df.groupby("system", sort=False):
            rows: List[Dict[str, object]] = []
            needed_sids = [int(v) for v in group["SID"].tolist() if pd.notna(v)]
            chosen = choose_best_numeric_candidate(system, needed_sids, numeric_candidates)
            
            if chosen:
                mt_lines = numeric_candidates[chosen]
            elif get_system_type(system) == "Human":
                mt_lines = ref_lines
            else:
                mt_lines = []
            for _, row in group.iterrows():
                sid = int(row["SID"])
                src = src_lines[sid - 1] if 1 <= sid <= len(src_lines) else ""
                ref = ref_lines[sid - 1] if 1 <= sid <= len(ref_lines) else ""
                mt = mt_lines[sid - 1] if 1 <= sid <= len(mt_lines) else ""
                if not str(mt).strip():
                    continue
                rows.append({"src": src, "mt": mt, "ref": ref, "score": float(row["Z.SCR"])})
            out_path = write_rows(out_root, 2019, pair, system, rows)
            if out_path:
                logs.append({"year": 2019, "pair": pair, "system": system, "rows": len(rows), "path": str(out_path)})


    return logs


def process_2020(sources: Dict[str, Path], out_root: Path) -> List[Dict[str, object]]:
    submission_root = sources["submission"]
    human_eval_root = sources["human_eval"]
    logs: List[Dict[str, object]] = []

    score_files = sorted(human_eval_root.rglob("ad-seg-scores-en-*.csv"))
    if not score_files:
        score_files = sorted(ROOT.rglob("2020/ad-seg-scores-en-*.csv"))

    for score_file in score_files:
        target_lang = score_file.stem.split("-")[-1]
        if ("en", target_lang) not in TARGET_PAIRS:
            continue
        src_sgm = find_first_existing(
            submission_root,
            [f"sgm/sources/newstest2020-en{target_lang}-src.en.sgm"],
        )
        ref_sgm = find_first_existing(
            submission_root,
            [
                f"sgm/references/newstest2020-en{target_lang}-ref.{target_lang}.sgm",
                f"sgm/references/newstestB2020-en{target_lang}-ref.{target_lang}.sgm",
            ],
        )
        if not src_sgm or not ref_sgm:
            print(f"[WARN] 2020 en-{target_lang}: missing source/reference SGM.")
            continue
        src_map = parse_sgm_docseg(src_sgm)
        ref_map = parse_sgm_docseg(ref_sgm)

        system_sgm_paths = sorted(
            p for p in submission_root.rglob(f"newstest2020*.en-{target_lang}*.sgm")
            if p.is_file() and "system-outputs" in p.parts and ".history" not in p.name
        )
        system_txt_paths = [
            p for p in submission_root.rglob(f"newstest2020*.en-{target_lang}*")
            if p.is_file() and p.suffix != ".sgm" and "system-outputs" in p.parts and ".history" not in p.name
        ]
        sgm_candidates = {path: parse_sgm_docseg(path) for path in system_sgm_paths}
        txt_candidates = {path: parse_plain_mt(path) for path in system_txt_paths}

        df = pd.read_csv(score_file, sep=r"\s+", engine="python")
        df["system"] = df["SYS"].astype(str).apply(lambda x: x.split(".")[0])

        rows_by_system: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        needed_docsegs: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        needed_sids: Dict[str, List[int]] = defaultdict(list)

        global_sid_lookup = {key: idx for idx, key in enumerate(src_map.keys(), start=1)}

        for _, row in df.iterrows():
            sid = str(row["SID"])
            if "_" not in sid:
                continue
            docid, seg_index_raw = sid.rsplit("_", 1)
            try:
                seg_id = int(seg_index_raw) + 1
            except ValueError:
                continue
            key = (docid, seg_id)
            system = row["system"]
            rows_by_system[system].append({"key": key, "src": src_map.get(key, ""), "ref": ref_map.get(key, ""), "score": float(row["Z.SCR"])})
            needed_docsegs[system].append(key)
            if key in global_sid_lookup:
                needed_sids[system].append(global_sid_lookup[key])

        for system, rows in rows_by_system.items():
            chosen_sgm = choose_best_docseg_candidate(system, needed_docsegs[system], sgm_candidates)
            chosen_txt = choose_best_numeric_candidate(system, needed_sids[system], txt_candidates) if txt_candidates else None
            out_rows = []
            is_human = get_system_type(system) == "Human"
            
            for row in rows:
                if chosen_sgm:
                    mt = sgm_candidates[chosen_sgm].get(row["key"], "")
                elif chosen_txt:
                    sid = global_sid_lookup.get(row["key"])
                    lines = txt_candidates[chosen_txt]
                    mt = lines[sid - 1] if sid and 1 <= sid <= len(lines) else ""
                elif is_human:
                    mt = row["ref"]  # Fallback to the reference text
                else:
                    mt = ""
                if not str(mt).strip():
                    continue
                out_rows.append({"src": row["src"], "mt": mt, "ref": row["ref"], "score": row["score"]})
            out_path = write_rows(out_root, 2020, f"en-{target_lang}", system, out_rows)
            if out_path:
                logs.append({"year": 2020, "pair": f"en-{target_lang}", "system": system, "rows": len(out_rows), "path": str(out_path)})


    return logs


def parse_xml21_like(xml_path: Path, src_lang: str = "en") -> Tuple[Dict[Tuple[str, int], str], Dict[Tuple[str, int], str], Dict[str, Dict[Tuple[str, int], str]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    src_map: Dict[Tuple[str, int], str] = {}
    ref_map: Dict[Tuple[str, int], str] = {}
    hyp_map: Dict[str, Dict[Tuple[str, int], str]] = defaultdict(dict)

    for doc in root.findall(".//doc"):
        doc_id = (doc.attrib.get("id") or "").strip()
        if not doc_id:
            continue

        src_elem = doc.find(f'./src[@lang="{src_lang}"]')
        if src_elem is None:
            src_elem = doc.find("./src")
        if src_elem is not None:
            for seg in src_elem.findall(".//seg"):
                sid_raw = (seg.attrib.get("id") or "").strip()
                if not sid_raw:
                    continue
                try:
                    seg_id = int(sid_raw)
                except ValueError:
                    continue
                src_map[(doc_id, seg_id)] = norm_text("".join(seg.itertext()))

        ref_elems = doc.findall("./ref")
        if ref_elems:
            for seg in ref_elems[0].findall(".//seg"):
                sid_raw = (seg.attrib.get("id") or "").strip()
                if not sid_raw:
                    continue
                try:
                    seg_id = int(sid_raw)
                except ValueError:
                    continue
                ref_map[(doc_id, seg_id)] = norm_text("".join(seg.itertext()))

        for hyp in doc.findall("./hyp") + doc.findall(".//hyp"):
            system = (hyp.attrib.get("system") or "").strip()
            if not system:
                continue
            for seg in hyp.findall(".//seg"):
                sid_raw = (seg.attrib.get("id") or "").strip()
                if not sid_raw:
                    continue
                try:
                    seg_id = int(sid_raw)
                except ValueError:
                    continue
                hyp_map[system][(doc_id, seg_id)] = norm_text("".join(seg.itertext()))

        for tgt in doc.findall("./tgt"):
            system = (tgt.attrib.get("system") or "").strip()
            if not system:
                continue
            for seg in tgt.findall(".//seg"):
                sid_raw = (seg.attrib.get("id") or "").strip()
                if not sid_raw:
                    continue
                try:
                    seg_id = int(sid_raw)
                except ValueError:
                    continue
                hyp_map[system][(doc_id, seg_id)] = norm_text("".join(seg.itertext()))

    return src_map, ref_map, hyp_map


def process_2021(sources: Dict[str, Path], out_root: Path) -> List[Dict[str, object]]:
    repo_root = sources["repo"]
    score_file = find_first_existing(repo_root, ["humaneval/newstest2021-EX_and_XY/wmt21-regular.20210930.csv"])
    if not score_file:
        raise FileNotFoundError("Could not find wmt21-regular.20210930.csv in the 2021 repo.")

    columns = [
        "User ID",
        "System Name",
        "Segment ID",
        "Quality control item",
        "Source Language",
        "Target Language",
        "Score",
        "Document ID",
        "Does score represent whole document",
        "Annotation Start",
        "Annotation End",
    ]
    lang_map = {"eng": "en", "ces": "cs", "zho": "zh", "deu": "de", "isl": "is", "jpn": "ja", "rus": "ru", "ha": "ha"}

    df = pd.read_csv(score_file, header=None, names=columns)
    df = df[df["Quality control item"] == "TGT"].copy()
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df["zscore"] = df.groupby("User ID")["Score"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) else 1.0))
    df["Segment ID"] = pd.to_numeric(df["Segment ID"], errors="coerce").astype("Int64")
    df["Document ID"] = df["Document ID"].astype(str).str.strip()
    agg = df.groupby(["System Name", "Document ID", "Segment ID", "Source Language", "Target Language"], dropna=False).agg({"zscore": "mean"}).reset_index()

    logs: List[Dict[str, object]] = []
    for xml_file in sorted(repo_root.rglob("xml/newstest2021.en-*.all.xml")):
        src_map, ref_map, hyp_map = parse_xml21_like(xml_file)
        pair = xml_file.name.split(".")[1]
        src_lang, tgt_lang = pair.split("-")
        if (src_lang, tgt_lang) not in TARGET_PAIRS:
            continue
        df_lang = agg[(agg["Source Language"].map(lang_map) == src_lang) & (agg["Target Language"].map(lang_map) == tgt_lang)].copy()
        for system in df_lang["System Name"].dropna().astype(str).str.strip().unique():
            df_sys = df_lang[df_lang["System Name"].astype(str).str.strip() == system].copy()
            rows = []
            for _, row in df_sys.iterrows():
                key = (row["Document ID"], int(row["Segment ID"])) if pd.notna(row["Segment ID"]) else None
                if not key:
                    continue
                mt = hyp_map.get(system, {}).get(key, "")
                if not mt and get_system_type(system) == "Human":
                    mt = ref_map.get(key, "")
                if not str(mt).strip():
                    continue
                rows.append({"src": src_map.get(key, ""), "mt": mt, "ref": ref_map.get(key, ""), "score": float(row["zscore"])})
            out_path = write_rows(out_root, 2021, f"{src_lang}-{tgt_lang}", system, rows)
            if out_path:
                logs.append({"year": 2021, "pair": f"{src_lang}-{tgt_lang}", "system": system, "rows": len(rows), "path": str(out_path)})
    return logs


def process_2022(sources: Dict[str, Path], out_root: Path) -> List[Dict[str, object]]:
    repo_root = sources["repo"]
    score_file = find_first_existing(repo_root, ["humaneval/DA+SQM/scores/WMT22.Appraise.DocLvl.20221107.filtered.csv"])
    if not score_file:
        raise FileNotFoundError("Could not find WMT22.Appraise.DocLvl.20221107.filtered.csv in the 2022 repo.")

    src_map_codes = {"eng": "en"}
    trg_map_codes = {"cs": "cs", "ces": "cs", "de": "de", "deu": "de", "hr": "hr", "hrv": "hr", "ja": "ja", "jpn": "ja", "ru": "ru", "rus": "ru", "uk": "uk", "ukr": "uk", "zh": "zh", "zho": "zh"}
    allowed = TARGET_PAIRS

    columns = [
        "username",
        "system",
        "itemId",
        "itemType",
        "srcLang",
        "trgLang",
        "score",
        "docId",
        "isDocScore",
        "startTime",
        "endTime",
    ]
    df = pd.read_csv(score_file, header=None, names=columns)
    df = df[df["isDocScore"] == False].copy()
    df = df[df["srcLang"].isin(src_map_codes)].copy()
    df["srcLang"] = df["srcLang"].map(src_map_codes)
    df = df[df["trgLang"].isin(trg_map_codes)].copy()
    df["trgLang"] = df["trgLang"].map(trg_map_codes)
    df = df[df.apply(lambda r: (r["srcLang"], r["trgLang"]) in allowed, axis=1)].copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["zscore"] = df.groupby("username")["score"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) else 1.0))
    df["itemId"] = pd.to_numeric(df["itemId"], errors="coerce").astype("Int64")
    df["docId"] = df["docId"].astype(str).str.strip()
    df = df.groupby(["system", "docId", "itemId", "srcLang", "trgLang"], dropna=False).agg({"zscore": "mean"}).reset_index()

    logs: List[Dict[str, object]] = []
    for (src_lang, tgt_lang), df_pair in df.groupby(["srcLang", "trgLang"]):
        xml_file = find_first_existing(repo_root, [f"xml/wmttest2022.{src_lang}-{tgt_lang}.all.xml"])
        if not xml_file:
            continue
        src_map, ref_map, hyp_map = parse_xml21_like(xml_file, src_lang=src_lang)
        for system, group in df_pair.groupby("system"):
            rows = []
            for _, row in group.iterrows():
                if pd.isna(row["itemId"]):
                    continue
                key = (row["docId"], int(row["itemId"]) + 1)
                mt = hyp_map.get(str(system), {}).get(key, "")
                if not mt and get_system_type(str(system)) == "Human":
                    mt = ref_map.get(key, "")
                if not str(mt).strip():
                    continue
                rows.append({"src": src_map.get(key, ""), "mt": mt, "ref": ref_map.get(key, ""), "score": float(row["zscore"])})
            out_path = write_rows(out_root, 2022, f"{src_lang}-{tgt_lang}", str(system), rows)
            if out_path:
                logs.append({"year": 2022, "pair": f"{src_lang}-{tgt_lang}", "system": str(system), "rows": len(rows), "path": str(out_path)})
    return logs


def process_2023(sources: Dict[str, Path], out_root: Path) -> List[Dict[str, object]]:
    repo_root = sources["repo"]
    score_file = find_first_existing(repo_root, ["humaneval/DA+SQM/WMT23.scores_all.csv"])
    if not score_file:
        raise FileNotFoundError("Could not find WMT23.scores_all.csv in the 2023 repo.")

    cols = ["UserID", "SystemID", "SegmentID", "Type", "SourceLanguage", "TargetLanguage", "Score", "DocumentID", "IsDocument?", "StartTime", "EndTime"]
    lang_map = {"cs": "ces", "de": "deu", "ru": "rus", "zh": "zho"}

    df = pd.read_csv(score_file, header=None, names=cols)
    df = df[(df["IsDocument?"] == False) & (df["Type"] == "TGT")].copy()
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    grouped = df.groupby("UserID")["Score"]
    std = grouped.transform("std").mask(lambda s: s == 0, 1.0)
    df["zscore"] = (df["Score"] - grouped.transform("mean")) / std
    df["DocumentID"] = df["DocumentID"].astype(str).str.strip()
    df["SegmentID"] = pd.to_numeric(df["SegmentID"], errors="coerce").astype("Int64")

    agg = df.groupby(["SystemID", "DocumentID", "SegmentID"], dropna=False).agg({"zscore": "mean"}).reset_index()
    lang_keys = df[["SystemID", "DocumentID", "SegmentID", "SourceLanguage", "TargetLanguage"]].drop_duplicates()
    merged = agg.merge(lang_keys, on=["SystemID", "DocumentID", "SegmentID"], how="left")

    logs: List[Dict[str, object]] = []
    for xml_file in sorted(repo_root.rglob("xml/wmttest2023.en-*.all.xml")):
        pair = xml_file.name.split(".")[1]
        src_lang, tgt_lang = pair.split("-")
        if (src_lang, tgt_lang) not in TARGET_PAIRS:
            continue
        target_three = lang_map.get(tgt_lang)
        if not target_three:
            continue
        src_map, ref_map, hyp_map = parse_xml21_like(xml_file, src_lang=src_lang)
        df_lang = merged[(merged["SourceLanguage"] == "eng") & (merged["TargetLanguage"] == target_three)].copy()
        for system in df_lang["SystemID"].dropna().astype(str).unique():
            rows = []
            df_sys = df_lang[df_lang["SystemID"].astype(str) == system]
            for _, row in df_sys.iterrows():
                key = (row["DocumentID"], int(row["SegmentID"])) if pd.notna(row["SegmentID"]) else None
                if not key:
                    continue
                mt = hyp_map.get(system, {}).get(key, "")
                if not mt and get_system_type(system) == "Human":
                    mt = ref_map.get(key, "")
                if not str(mt).strip():
                    continue
                rows.append({"src": src_map.get(key, ""), "mt": mt, "ref": ref_map.get(key, ""), "score": float(row["zscore"])})
            out_path = write_rows(out_root, 2023, f"{src_lang}-{tgt_lang}", system, rows)
            if out_path:
                logs.append({"year": 2023, "pair": f"{src_lang}-{tgt_lang}", "system": system, "rows": len(rows), "path": str(out_path)})
    return logs


def parse_xml_2024(xml_path: Path) -> Tuple[Dict[Tuple[str, int], str], Dict[str, Dict[Tuple[str, int], str]], Dict[str, Dict[Tuple[str, int], str]], Dict[Tuple[str, int], str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    src_map: Dict[Tuple[str, int], str] = {}
    hyp_map: Dict[str, Dict[Tuple[str, int], str]] = defaultdict(dict)
    ref_map: Dict[str, Dict[Tuple[str, int], str]] = defaultdict(dict)
    ref_first_map: Dict[Tuple[str, int], str] = {}

    for doc in root.findall(".//doc"):
        doc_id = (doc.attrib.get("id") or "").strip()
        if not doc_id:
            continue
        src_elem = doc.find('./src[@lang="en"]')
        if src_elem is not None:
            for seg in src_elem.findall(".//seg"):
                sid_raw = (seg.attrib.get("id") or "").strip()
                if not sid_raw:
                    continue
                try:
                    seg_id = int(sid_raw)
                except ValueError:
                    continue
                src_map[(doc_id, seg_id)] = norm_text("".join(seg.itertext()))
        ref_elems = doc.findall("./ref")
        if ref_elems:
            for seg in ref_elems[0].findall(".//seg"):
                sid_raw = (seg.attrib.get("id") or "").strip()
                if not sid_raw:
                    continue
                try:
                    seg_id = int(sid_raw)
                except ValueError:
                    continue
                ref_first_map[(doc_id, seg_id)] = norm_text("".join(seg.itertext()))
        for ref in doc.findall(".//ref"):
            translator = (ref.attrib.get("translator") or "").strip()
            if not translator:
                continue
            for seg in ref.findall(".//seg"):
                sid_raw = (seg.attrib.get("id") or "").strip()
                if not sid_raw:
                    continue
                try:
                    seg_id = int(sid_raw)
                except ValueError:
                    continue
                ref_map[translator][(doc_id, seg_id)] = norm_text("".join(seg.itertext()))
        for hyp in doc.findall(".//hyp"):
            system = (hyp.attrib.get("system") or "").strip()
            if not system:
                continue
            for seg in hyp.findall(".//seg"):
                sid_raw = (seg.attrib.get("id") or "").strip()
                if not sid_raw:
                    continue
                try:
                    seg_id = int(sid_raw)
                except ValueError:
                    continue
                hyp_map[system][(doc_id, seg_id)] = norm_text("".join(seg.itertext()))
    return src_map, hyp_map, ref_map, ref_first_map


def build_global_ref_lines_2024(xml_path: Path) -> List[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = [""]
    for doc in root.findall(".//doc"):
        refs = doc.findall("./ref")
        if not refs:
            continue
        segs: List[Tuple[int, str]] = []
        for seg in refs[0].findall(".//seg"):
            sid_raw = (seg.attrib.get("id") or "").strip()
            if not sid_raw:
                continue
            try:
                seg_id = int(sid_raw)
            except ValueError:
                continue
            segs.append((seg_id, norm_text("".join(seg.itertext()))))
        lines.extend(text for _seg_id, text in sorted(segs, key=lambda x: x[0]))
    return lines


def mqm_weight_2024(category: str, severity: str) -> float:
    severity = severity or "No-error"
    if severity in {"No-error", "neutral"} or "Reinterpretation" in (category or ""):
        return 0.0
    if category == "Non-translation!":
        return -25.0
    if category == "Source issue":
        return 0.0
    if severity == "minor":
        if "Fluency/Punctuation" in (category or ""):
            return -0.1
        return -1.0
    if severity in {"major", "critical"}:
        return -5.0
    raise ValueError(f"Unknown MQM severity/category: {severity} / {category}")


def process_2024(sources: Dict[str, Path], out_root: Path) -> List[Dict[str, object]]:
    repo_root = sources["repo"]
    wave_files = sorted(repo_root.rglob("humeval/esa_generalMT2024_wave*.csv"))
    if not wave_files:
        raise FileNotFoundError("Could not find humeval/esa_generalMT2024_wave*.csv in the 2024 repo.")
    mqm_file = find_first_existing(repo_root, ["humeval/mqm_generalMT2024_ende.tsv"])
    if not mqm_file:
        raise FileNotFoundError("Could not find humeval/mqm_generalMT2024_ende.tsv in the 2024 repo.")

    lang_map = {"eng": "en", "ces": "cs", "deu": "de", "rus": "ru", "zho": "zh", "jpn": "ja", "hin": "hi", "ukr": "uk", "isl": "is", "spa": "es"}
    columns = ["user_id", "system_id", "segment_id", "segment_type", "source_lang", "target_lang", "overall", "doc_id", "unk", "error_spans", "start_time", "end_time"]

    pair_ref_lines: Dict[str, List[str]] = {}
    pair_source_lines: Dict[str, List[str]] = {}
    for _src_lang, tgt_lang in sorted(TARGET_PAIRS):
        pair = f"en-{tgt_lang}"
        xml_file = find_first_existing(repo_root, [f"xml/wmttest2024.{pair}.all.xml"])
        src_file = find_first_existing(repo_root, [f"txt-ts/sources/{pair}.txt"])
        if src_file:
            pair_source_lines[pair] = read_text_lines(src_file)
        if xml_file:
            pair_ref_lines[pair] = build_global_ref_lines_2024(xml_file)

    system_cache: Dict[Tuple[str, str], List[str]] = {}

    def get_system_lines(pair: str, system: str) -> List[str]:
        key = (pair, system)
        if key not in system_cache:
            path = find_first_existing(repo_root, [f"txt-ts/system-outputs/{pair}/{system}.txt"])
            system_cache[key] = read_text_lines(path) if path else []
        return system_cache[key]

    ann_rows: List[Dict[str, object]] = []
    for wave_file in wave_files:
        df = pd.read_csv(wave_file, header=None, dtype=str)
        df.columns = columns
        df = df[df["segment_type"] == "TGT"].copy()
        df = df[~df["system_id"].astype(str).str.contains("tutorial", na=False)]
        df = df[~df["doc_id"].astype(str).str.contains("#incomplete", na=False)]
        df = df[~df["doc_id"].astype(str).str.contains("#dup", na=False)]
        df = df[df["doc_id"].astype(str) != "canary"].copy()
        df["source_lang"] = df["source_lang"].map(lang_map)
        df["target_lang"] = df["target_lang"].map(lang_map)
        df = df[df.apply(lambda r: (r["source_lang"], r["target_lang"]) in TARGET_PAIRS, axis=1)].copy()
        df["pair"] = df["source_lang"] + "-" + df["target_lang"]
        df["segment_index"] = pd.to_numeric(df["segment_id"], errors="coerce").astype("Int64")
        df["overall"] = pd.to_numeric(df["overall"], errors="coerce")

        for _, row in df.iterrows():
            if pd.isna(row["segment_index"]):
                continue
            pair = row["pair"]
            seg_idx = int(row["segment_index"])
            src_lines = pair_source_lines.get(pair, [])
            ref_lines = pair_ref_lines.get(pair, [])
            system_id = str(row["system_id"])
            mt_lines = get_system_lines(pair, system_id)
            
            # Fallback to reference lines for human translators
            if not mt_lines and get_system_type(system_id) == "Human":
                mt_lines = ref_lines
            src_text = src_lines[seg_idx] if 0 <= seg_idx < len(src_lines) else ""
            mt_text = mt_lines[seg_idx] if 0 <= seg_idx < len(mt_lines) else ""
            ref_text = ref_lines[seg_idx] if 0 <= seg_idx < len(ref_lines) else ""
            if not str(mt_text).strip():
                continue
            ann_rows.append(
                {
                    "pair": pair,
                    "system": str(row["system_id"]),
                    "annotator": str(row["user_id"]),
                    "score_raw": float(row["overall"]),
                    "seg_index": seg_idx,
                    "src": src_text,
                    "mt": mt_text,
                    "ref": ref_text,
                }
            )

    mqm_df = pd.read_csv(mqm_file, dtype=str, sep="\t", quoting=csv.QUOTE_NONE, quotechar="\x07")
    mqm_df = mqm_df[mqm_df["doc"].astype(str) != "canary"].copy()
    mqm_df["globalSegId"] = pd.to_numeric(mqm_df["globalSegId"], errors="coerce").astype("Int64")
    mqm_df["score_part"] = mqm_df.apply(lambda r: mqm_weight_2024(str(r["category"]), str(r["severity"])), axis=1)
    mqm_df = mqm_df.groupby(["system", "doc", "docSegId", "globalSegId", "rater"], dropna=False, sort=False).agg({"score_part": "sum"}).reset_index()

    pair = "en-de"
    src_lines = pair_source_lines.get(pair, [])
    ref_lines = pair_ref_lines.get(pair, [])
    for _, row in mqm_df.iterrows():
        if pd.isna(row["globalSegId"]):
            continue
        seg_idx = int(row["globalSegId"]) - 1
        if seg_idx < 0:
            continue
            
        system_id = str(row["system"])
        mt_lines = get_system_lines(pair, system_id)
        
        # Fallback to reference lines for human translators
        if not mt_lines and get_system_type(system_id) == "Human":
            mt_lines = ref_lines
        src_text = src_lines[seg_idx] if seg_idx < len(src_lines) else ""
        mt_text = mt_lines[seg_idx] if seg_idx < len(mt_lines) else ""
        ref_text = ref_lines[seg_idx] if seg_idx < len(ref_lines) else ""
        if not str(mt_text).strip():
            continue
        ann_rows.append(
            {
                "pair": pair,
                "system": str(row["system"]),
                "annotator": str(row["rater"]),
                "score_raw": float(row["score_part"]),
                "seg_index": seg_idx,
                "src": src_text,
                "mt": mt_text,
                "ref": ref_text,
            }
        )

    if not ann_rows:
        return []

    ann_df = pd.DataFrame(ann_rows)
    ann_df["score"] = ann_df.groupby("annotator")["score_raw"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) else 1.0))
    ann_df = collapse_segment_duplicates(ann_df, group_cols=["pair", "system", "seg_index"])

    logs: List[Dict[str, object]] = []
    for (pair, system), group in ann_df.groupby(["pair", "system"], sort=False):
        rows = group[["src", "mt", "ref", "score"]].to_dict(orient="records")
        out_path = write_rows(out_root, 2024, pair, system, rows)
        if out_path:
            logs.append({"year": 2024, "pair": pair, "system": system, "rows": len(rows), "path": str(out_path)})
    return logs


PROCESSORS: Dict[int, Callable[[Dict[str, Path], Path], List[Dict[str, object]]]] = {
    2017: lambda sources, out_root: process_2017_2018(2017, sources, out_root),
    2018: lambda sources, out_root: process_2017_2018(2018, sources, out_root),
    2019: process_2019,
    2020: process_2020,
    2021: process_2021,
    2022: process_2022,
    2023: process_2023,
    2024: process_2024,
}


def run_year(year: int, raw_root: Path, out_root: Path, skip_download: bool, force: bool) -> List[Dict[str, object]]:
    print(f"[INFO] Preparing WMT {year}")
    sources = ensure_year_downloads(year, raw_root=raw_root, skip_download=skip_download, force=force)
    logs = PROCESSORS[year](sources, out_root)
    print(f"[INFO] WMT {year}: wrote {len(logs)} system files")
    return logs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download, extract, and process WMT 2017-2024 into datasets/WMT/{year}/{pair}/{system}.csv")
    parser.add_argument("--years", nargs="+", type=int, default=list(ALLOWED_YEARS), help="Years to process, e.g. --years 2017 2021 2024")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT, help=f"Where raw downloads/extractions are stored. Default: {DEFAULT_RAW_ROOT}")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help=f"Where final CSVs are written. Default: {DEFAULT_OUT_ROOT}")
    parser.add_argument("--skip-download", action="store_true", help="Use already-downloaded archives or local cache directories; do not hit the network.")
    parser.add_argument("--force-download", action="store_true", help="Redownload archives even if the files already exist.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional path for a JSON manifest of generated files.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    years = []
    for year in args.years:
        if year not in ALLOWED_YEARS:
            raise ValueError(f"Unsupported year: {year}")
        years.append(year)

    mkdir(args.raw_root)
    mkdir(args.out_root)

    all_logs: List[Dict[str, object]] = []
    for year in years:
        year_logs = run_year(year, raw_root=args.raw_root, out_root=args.out_root, skip_download=args.skip_download, force=args.force_download)
        all_logs.extend(year_logs)

    manifest_path = args.manifest or (args.out_root / "manifest.json")
    mkdir(manifest_path.parent)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(all_logs, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Wrote manifest: {manifest_path}")
    print(f"[INFO] Total generated files: {len(all_logs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
