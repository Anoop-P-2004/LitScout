import os
import time
import requests
from typing import List, Dict, Optional
from pathlib import Path
import re

# ---------------- CONFIG ---------------- #

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "Extracted Papers"
OUTPUT_DIR.mkdir(exist_ok=True)

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper"
UNPAYWALL_API = "https://api.unpaywall.org/v2"

REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_REQUESTS = 1.0  # polite usage

# Optional but recommended (Unpaywall requires an email string)
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", None)

# ---------------- UTILITIES ---------------- #

def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in " _-()" else "_" for c in name)[:180]

def extract_arxiv_id(text: str):
    if not text:
        return None
    match = re.search(r"arxiv\.org/(abs|pdf)/(\d+\.\d+)", text.lower())
    return match.group(2) if match else None

def download_pdf(url: str, filepath: Path) -> bool:
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()

        if "application/pdf" not in r.headers.get("Content-Type", ""):
            return False

        with open(filepath, "wb") as f:
            f.write(r.content)

        return True
    except Exception:
        return False


# ---------------- FREE RETRIEVAL METHODS ---------------- #

def semantic_scholar_oa_pdf(paper_id: str) -> Optional[str]:
    try:
        r = requests.get(
            f"{SEMANTIC_SCHOLAR_API}/{paper_id}",
            params={"fields": "openAccessPdf"},
            timeout=REQUEST_TIMEOUT
        )
        r.raise_for_status()
        data = r.json()
        pdf = data.get("openAccessPdf")
        if pdf and pdf.get("url"):
            return pdf["url"]
    except Exception:
        pass
    return None


def arxiv_pdf(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def unpaywall_pdf(doi: str) -> Optional[str]:
    if not UNPAYWALL_EMAIL:
        return None

    try:
        r = requests.get(
            f"{UNPAYWALL_API}/{doi}",
            params={"email": UNPAYWALL_EMAIL},
            timeout=REQUEST_TIMEOUT
        )
        r.raise_for_status()
        data = r.json()
        loc = data.get("best_oa_location")
        if loc and loc.get("url_for_pdf"):
            return loc["url_for_pdf"]
    except Exception:
        pass
    return None


# ---------------- MAIN EXTRACTION LOGIC ---------------- #

def fetch_single_paper(paper: Dict) -> Dict[str, str]:
    title = paper.get("title", "unknown")
    paper_id = paper.get("paperId")
    external_ids = paper.get("externalIds", {})

    filename = sanitize_filename(title) + ".pdf"
    filepath = OUTPUT_DIR / filename

    if filepath.exists():
        return {"title": title, "status": "already_exists"}

    # 1. Semantic Scholar OA
    if paper_id:
        url = semantic_scholar_oa_pdf(paper_id)
        if url and download_pdf(url, filepath):
            return {"title": title, "status": "downloaded", "source": "Semantic Scholar OA"}

    # 2. arXiv
    arxiv_id = external_ids.get("ArXiv")
    if not arxiv_id:
        arxiv_id = extract_arxiv_id(
            paper.get("url", "") + " " +
            str(paper.get("openAccessPdf", {}).get("disclaimer", ""))
        )

    if arxiv_id:
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        if download_pdf(url, filepath):
            return {"title": title, "status": "downloaded", "source": "arXiv"}

    # 3. Unpaywall (DOI)
    doi = external_ids.get("DOI")
    if doi:
        url = unpaywall_pdf(doi)
        if url and download_pdf(url, filepath):
            return {"title": title, "status": "downloaded", "source": "Unpaywall"}

    return {"title": title, "status": "failed"}


def fetch_papers(screened_papers: List[Dict]) -> Dict[str, int]:
    print("\n=== FETCHING PAPERS ===")
    print(f"Target directory: {OUTPUT_DIR.resolve()}")
    print(f"Total papers: {len(screened_papers)}\n")

    downloaded = 0
    skipped = 0
    failed = 0

    for idx, paper in enumerate(screened_papers, 1):
        result = fetch_single_paper(paper)
        title = result["title"]

        if result["status"] == "downloaded":
            downloaded += 1
            print(f"[{idx}] ✔ Downloaded: {title} ({result['source']})")

        elif result["status"] == "already_exists":
            skipped += 1
            print(f"[{idx}] ↺ Already exists: {title}")

        else:
            failed += 1
            print(f"[{idx}] ✖ Failed: {title}")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print("\n=== EXTRACTION SUMMARY ===")
    print(f"Downloaded: {downloaded}")
    print(f"Already existed: {skipped}")
    print(f"Failed: {failed}")
    print(f"Saved in: {OUTPUT_DIR.resolve()}")

    return {
        "total": len(screened_papers),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed
    }


# ---------------- ENTRY POINT ---------------- #

if __name__ == "__main__":
    raise RuntimeError(
        "This module is not standalone.\n"
        "Import and call fetch_papers(screened_papers) from your pipeline."
    )
