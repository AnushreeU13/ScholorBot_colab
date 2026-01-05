import json
import zipfile
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MAPPING_DIR = PROJECT_ROOT / "datasets" / "KB_raw" / "druglabels" / "mappings"
OUT_DIR = PROJECT_ROOT / "datasets" / "KB_processed" / "druglabels_text"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RXNORM_ZIP = MAPPING_DIR / "rxnorm_mappings.zip"
META_ZIP = MAPPING_DIR / "dm_spl_zip_files_meta_data.zip"

# Keep this list clinically relevant; you can expand later
TARGET_INGREDIENTS = [
    # TB first-line
    "isoniazid",
    "rifampin",
    "rifampicin",
    "pyrazinamide",
    "ethambutol",
    # CAP common antibiotics
    "amoxicillin",
    "amoxicillin clavulanate",
    "azithromycin",
    "doxycycline",
    "levofloxacin",
    "moxifloxacin",
    "ceftriaxone",
]

def _read_pipe_delimited_from_zip(zip_path: Path, inner_filename_hint: str):
    """
    Read the first file in zip whose name contains 'inner_filename_hint'
    and return lines (decoded as utf-8 with fallback).
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        candidates = [n for n in z.namelist() if inner_filename_hint in n.lower()]
        if not candidates:
            candidates = [z.namelist()[0]]
        name = candidates[0]
        raw = z.read(name)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="ignore")

    return text.splitlines()

def build_setid_from_rxnorm():
    """
    Build a set of SPL_SETID by matching keywords in RXSTRING.
    rxnorm_mappings format:
    SPL_SETID|SPL_VERSION|RXCUI|RXSTRING|RXTTY
    """
    lines = _read_pipe_delimited_from_zip(RXNORM_ZIP, "rxnorm")
    header = lines[0].strip().lower()
    assert "rxstring" in header, f"Unexpected header: {lines[0]}"

    targets = [t.lower() for t in TARGET_INGREDIENTS]
    setids = set()

    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) < 5:
            continue
        spl_setid = parts[0].strip().lower()
        rxstring = parts[3].strip().lower()

        # Simple deterministic keyword match (reproducible)
        if any(t in rxstring for t in targets):
            setids.add(spl_setid)

    return setids

def build_setid_to_zipmeta():
    """
    Build mapping SETID -> {zip_name, upload_date, spl_version, title}
    meta format:
    SETID|ZIP_FILE_NAME|UPLOAD_DATE|SPL_VERSION|TITLE
    """
    lines = _read_pipe_delimited_from_zip(META_ZIP, "meta")
    header = lines[0].strip().lower()
    assert "zip_file_name" in header, f"Unexpected header: {lines[0]}"

    m = {}
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) < 5:
            continue

        setid = parts[0].strip().lower()
        zip_name = parts[1].strip()
        upload_date = parts[2].strip()
        spl_version = parts[3].strip()
        title = parts[4].strip()

        if setid and zip_name:
            m[setid] = {
                "zip_name": zip_name,
                "upload_date": upload_date,     # DailyMed UPLOAD_DATE
                "spl_version": spl_version,
                "title": title,
            }
    return m

def main():
    setids = build_setid_from_rxnorm()
    print(f"[INFO] Candidate setids from rxnorm match: {len(setids)}")

    setid2meta = build_setid_to_zipmeta()

    inner_paths = []
    meta_rows = []
    missing = 0

    # Convert setid -> inner zip path used in your outer zip structure
    # Outer zip uses: prescription/<ZIP_FILE_NAME>
    for s in sorted(setids):
        if s not in setid2meta:
            missing += 1
            continue

        zip_name = setid2meta[s]["zip_name"]
        inner_zip_path = f"prescription/{zip_name}"
        inner_paths.append(inner_zip_path)

        meta_rows.append({
            "setid": s,
            "inner_zip_path": inner_zip_path,
            "zip_file_name": zip_name,
            "upload_date": setid2meta[s].get("upload_date"),
            "spl_version": setid2meta[s].get("spl_version"),
            "title": setid2meta[s].get("title"),
        })

    print(f"[INFO] Mapped to inner zip filenames: {len(inner_paths)}")
    print(f"[WARN] Missing setid in zip metadata: {missing}")

    out_txt = OUT_DIR / "spl_target_inner_zip_paths.txt"
    out_setid = OUT_DIR / "spl_target_setids.txt"
    out_meta = OUT_DIR / "spl_setid_zipmeta.json"

    out_txt.write_text("\n".join(inner_paths), encoding="utf-8")
    out_setid.write_text("\n".join(sorted(setids)), encoding="utf-8")
    out_meta.write_text(json.dumps(meta_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Wrote whitelist paths: {out_txt}")
    print(f"[DONE] Wrote setids: {out_setid}")
    print(f"[DONE] Wrote zip meta JSON: {out_meta}")

if __name__ == "__main__":
    main()
