"""Download openly licensed images from Wikimedia Commons, with a licence log.

Only files whose licence is Creative Commons or public domain are kept. Every file is
written to data/sources.csv with its author, licence and page URL.

    python scripts/fetch_commons.py --out raw/murals_full --domain B --dataset mural \
        --search "kerala mural painting" --category "Mural paintings in Kerala" --limit 400
"""

import argparse
import csv
import html
import os
import re
import time
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
UA = "KeralaGAN/0.1 (student research project; https://github.com/abhinavrenjithg-sys/keralagan)"
OK_LICENCE = re.compile(r"^(cc[- ]?by|cc[- ]?by[- ]?sa|cc0|cc[- ]?zero|public domain|pd\b|pd-)", re.I)
OK_MIME = {"image/jpeg", "image/png", "image/webp"}


def clean(text):
    """Strip HTML from Commons metadata fields."""
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()[:200]


def api(session, **params):
    params.update(format="json", formatversion=2)
    for attempt in range(4):
        r = session.get(API, params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def iter_pages(session, generator_params, width):
    params = dict(prop="imageinfo", iiprop="url|mime|size|extmetadata", iiurlwidth=width, **generator_params)
    while True:
        data = api(session, action="query", **params)
        for page in data.get("query", {}).get("pages", []):
            yield page
        if "continue" not in data:
            break
        params.update(data["continue"])


def subcategories(session, category, depth):
    seen, frontier = {category}, [category]
    for _ in range(depth):
        nxt = []
        for cat in frontier:
            data = api(session, action="query", list="categorymembers", cmtitle=f"Category:{cat}",
                       cmtype="subcat", cmlimit=500)
            for m in data.get("query", {}).get("categorymembers", []):
                name = m["title"].split(":", 1)[1]
                if name not in seen:
                    seen.add(name); nxt.append(name)
        frontier = nxt
    return sorted(seen)


def candidates(session, args):
    gens = []
    for cat in args.category:
        for c in subcategories(session, cat, args.depth):
            gens.append(dict(generator="categorymembers", gcmtitle=f"Category:{c}", gcmtype="file", gcmlimit=200))
    for q in args.search:
        gens.append(dict(generator="search", gsrsearch=f"filetype:bitmap {q}", gsrnamespace=6, gsrlimit=200))
    for g in gens:
        try:
            yield from iter_pages(session, g, args.width)
        except requests.HTTPError as e:
            print(f"  skipped {g}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", required=True, help="dataset name for sources.csv, e.g. mural")
    ap.add_argument("--domain", required=True, choices=["A", "B"])
    ap.add_argument("--category", action="append", default=[], help="Commons category (repeatable)")
    ap.add_argument("--search", action="append", default=[], help="Commons file search (repeatable)")
    ap.add_argument("--exclude", action="append", default=[], help="skip titles containing this text (repeatable)")
    ap.add_argument("--depth", type=int, default=1, help="how deep to follow subcategories")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--width", type=int, default=1024, help="download a thumbnail this wide")
    ap.add_argument("--min_side", type=int, default=400)
    ap.add_argument("--sources", default="data/sources.csv")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    s = requests.Session(); s.headers["User-Agent"] = UA
    new_file = not os.path.exists(args.sources) or os.path.getsize(args.sources) == 0
    log_f = open(args.sources, "a", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    if new_file:
        log.writerow(["dataset", "domain", "file_or_folder", "source_url", "author", "licence", "notes"])

    seen, kept, skipped = set(), 0, {"licence": 0, "small": 0, "type": 0, "excluded": 0, "error": 0}
    excl = [e.lower() for e in args.exclude]
    for page in candidates(s, args):
        if kept >= args.limit:
            break
        title = page.get("title", "")
        if title in seen or not page.get("imageinfo"):
            continue
        seen.add(title)
        if any(e in title.lower() for e in excl):
            skipped["excluded"] += 1; continue
        info = page["imageinfo"][0]
        meta = info.get("extmetadata", {})
        licence = clean(meta.get("LicenseShortName", {}).get("value"))
        if not OK_LICENCE.match(licence):
            skipped["licence"] += 1; continue
        if info.get("mime") not in OK_MIME:
            skipped["type"] += 1; continue
        if min(info.get("width", 0), info.get("height", 0)) < args.min_side:
            skipped["small"] += 1; continue
        url = info.get("thumburl") or info["url"]
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", title.split(":", 1)[1])[:90]
        name = os.path.splitext(name)[0] + ".jpg"
        try:
            r = s.get(url, timeout=60); r.raise_for_status()
            (out / name).write_bytes(r.content)
        except requests.RequestException:
            skipped["error"] += 1; continue
        author = clean(meta.get("Artist", {}).get("value")) or "unknown"
        log.writerow([args.dataset, args.domain, f"{out}/{name}", info.get("descriptionurl", ""), author, licence,
                      "Wikimedia Commons"])
        kept += 1
        if kept % 25 == 0:
            print(f"  {kept} downloaded"); log_f.flush()
        time.sleep(0.2)  # be polite to Commons
    log_f.close()
    print(f"Kept {kept} images in {out}; skipped {skipped}")


if __name__ == "__main__":
    main()
