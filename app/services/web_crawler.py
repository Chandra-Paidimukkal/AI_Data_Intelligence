"""
web_crawler.py — Deep web crawler + schema-based data extractor.

Crawls websites (single page, URL list, or deep crawl) and extracts
structured data matching a given schema — no AI API key required.

Uses:
  - httpx for async HTTP fetching
  - BeautifulSoup for HTML parsing
  - Python heuristic extractor for field matching
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any
from urllib.parse import urljoin, urlparse

from loguru import logger

# Max concurrent requests
_MAX_CONCURRENT = 5
# Request timeout seconds
_TIMEOUT = 15
# Max pages per deep crawl
_MAX_PAGES = 100
# Max text chars per page sent to extractor
_MAX_TEXT = 15_000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ── Entry points ──────────────────────────────────────────────────────────────

async def crawl_url_list(
    urls: list[str],
    schema: dict,
    max_depth: int = 1,
    on_progress=None,
) -> list[dict]:
    """
    Crawl a list of URLs and extract schema fields from each.
    If max_depth > 1, follows internal links up to that depth.
    Returns list of result dicts with source_url + extracted fields.
    """
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
    all_results = []

    for i, url in enumerate(urls):
        if on_progress:
            on_progress(i, len(urls), url)
        try:
            if max_depth <= 1:
                result = await _scrape_single(url, schema, semaphore)
                if result:
                    all_results.append(result)
            else:
                results = await _deep_crawl(url, schema, max_depth, semaphore)
                all_results.extend(results)
        except Exception as e:
            logger.warning(f"Failed to crawl {url}: {e}")
            all_results.append({
                "source_url": url,
                "error": str(e),
                **{f["name"]: None for f in schema.get("fields", [])},
            })

    return all_results


async def crawl_single_url(url: str, schema: dict, max_depth: int = 1) -> list[dict]:
    """Crawl a single URL (with optional deep crawl) and extract schema fields."""
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
    if max_depth <= 1:
        result = await _scrape_single(url, schema, semaphore)
        return [result] if result else []
    else:
        return await _deep_crawl(url, schema, max_depth, semaphore)


# ── Single page scraper ───────────────────────────────────────────────────────

async def _scrape_single(url: str, schema: dict, semaphore: asyncio.Semaphore) -> dict | None:
    """Fetch one URL and extract schema fields from its content."""
    async with semaphore:
        html, final_url = await _fetch_html(url)
        if not html:
            return None

        parsed = _parse_html(html, final_url)
        extracted = _extract_from_parsed(parsed, schema)
        extracted["source_url"] = final_url
        extracted["page_title"] = parsed.get("title", "")
        return extracted


# ── Deep crawler ──────────────────────────────────────────────────────────────

async def _deep_crawl(
    start_url: str,
    schema: dict,
    max_depth: int,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """
    Crawl a site starting from start_url, following internal links
    up to max_depth levels deep. Extracts schema fields from every page.
    """
    base_domain = urlparse(start_url).netloc
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(start_url, 0)]
    all_results: list[dict] = []

    while queue and len(visited) < _MAX_PAGES:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        logger.info(f"Crawling [{depth}/{max_depth}]: {url}")

        async with semaphore:
            html, final_url = await _fetch_html(url)
        if not html:
            continue

        parsed = _parse_html(html, final_url)
        extracted = _extract_from_parsed(parsed, schema)
        extracted["source_url"] = final_url
        extracted["page_title"] = parsed.get("title", "")

        # Only add if at least one field was extracted
        non_null = sum(1 for k, v in extracted.items()
                       if k not in ("source_url", "page_title", "error") and v is not None)
        if non_null > 0:
            all_results.append(extracted)

        # Follow internal links if not at max depth
        if depth < max_depth:
            for link in parsed.get("links", []):
                link_domain = urlparse(link).netloc
                if link_domain == base_domain and link not in visited:
                    queue.append((link, depth + 1))

        # Small delay to be polite
        await asyncio.sleep(0.3)

    logger.info(f"Deep crawl complete: {len(visited)} pages, {len(all_results)} results")
    return all_results


# ── HTTP fetcher ──────────────────────────────────────────────────────────────

async def _fetch_html(url: str) -> tuple[str, str]:
    """Fetch HTML from a URL. Returns (html, final_url) or ('', url) on error."""
    try:
        import httpx
        async with httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers=HEADERS,
            follow_redirects=True,
            verify=False,  # some sites have cert issues
        ) as client:
            r = await client.get(url)
            if r.status_code == 200:
                return r.text, str(r.url)
            logger.warning(f"HTTP {r.status_code} for {url}")
            return "", url
    except Exception as e:
        logger.warning(f"Fetch error {url}: {e}")
        return "", url


# ── HTML parser ───────────────────────────────────────────────────────────────

def _parse_html(html: str, base_url: str) -> dict:
    """
    Parse HTML and extract:
    - title
    - clean text content
    - tables (as list of {headers, rows})
    - internal links
    - key-value pairs from definition lists, spec tables
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Fallback: basic regex extraction
        return _parse_html_regex(html, base_url)

    soup = BeautifulSoup(html, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "advertisement", "iframe", "noscript"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title else ""

    # Extract tables
    tables = []
    for tbl in soup.find_all("table"):
        rows_raw = []
        for tr in tbl.find_all("tr"):
            cells = [td.get_text(separator=" ", strip=True)
                     for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows_raw.append(cells)
        if len(rows_raw) >= 2:
            tables.append({
                "headers": rows_raw[0],
                "rows": rows_raw[1:],
            })

    # Extract definition lists (common for specs)
    kv_pairs = []
    for dl in soup.find_all("dl"):
        terms = dl.find_all("dt")
        defs = dl.find_all("dd")
        for dt, dd in zip(terms, defs):
            key = dt.get_text(strip=True)
            val = dd.get_text(strip=True)
            if key and val:
                kv_pairs.append({"key": key, "value": val})

    # Extract spec-like divs (label: value patterns)
    for elem in soup.find_all(class_=re.compile(r"spec|attribute|property|detail|feature", re.I)):
        text = elem.get_text(separator="\n", strip=True)
        for line in text.split("\n"):
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2 and len(parts[0]) < 60:
                    kv_pairs.append({"key": parts[0].strip(), "value": parts[1].strip()})

    # Clean text
    main_content = soup.find("main") or soup.find(id=re.compile(r"content|main|product", re.I)) or soup.body
    if main_content:
        text = main_content.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = text[:_MAX_TEXT]

    # Extract internal links
    links = []
    base_domain = urlparse(base_url).netloc
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        if parsed.netloc == base_domain and parsed.scheme in ("http", "https"):
            # Skip non-content links
            if not any(ext in parsed.path.lower() for ext in
                       [".jpg", ".png", ".gif", ".pdf", ".zip", ".css", ".js"]):
                links.append(full_url.split("#")[0])  # remove anchors

    return {
        "title": title,
        "text": text,
        "tables": tables,
        "kv_pairs": kv_pairs,
        "links": list(set(links))[:50],  # deduplicate, cap at 50
    }


def _parse_html_regex(html: str, base_url: str) -> dict:
    """Fallback HTML parser using regex (when BeautifulSoup not available)."""
    # Strip tags
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()[:_MAX_TEXT]

    title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = title_m.group(1).strip() if title_m else ""

    return {"title": title, "text": text, "tables": [], "kv_pairs": [], "links": []}


# ── Schema field extractor ────────────────────────────────────────────────────

def _extract_from_parsed(parsed: dict, schema: dict) -> dict:
    """
    Extract schema fields from parsed page content.
    Uses heuristic matching — no AI needed.
    """
    text = parsed.get("text", "")
    tables = parsed.get("tables", [])
    kv_pairs = parsed.get("kv_pairs", [])
    fields = schema.get("fields", [])

    result = {}

    for field in fields:
        fname = field["name"]
        ftype = field.get("type", "string")
        desc = field.get("description", "")

        # Build search terms from field name + description
        search_terms = _get_search_terms(fname, desc)

        value = None

        # 1. Try KV pairs first (most reliable)
        value = _search_kv(kv_pairs, search_terms, ftype)

        # 2. Try tables
        if value is None:
            value = _search_tables(tables, search_terms, ftype)

        # 3. Try text patterns
        if value is None:
            value = _search_text(text, search_terms, ftype)

        result[fname] = value

    return result


def _get_search_terms(field_name: str, description: str) -> list[str]:
    """Generate search terms from field name and description."""
    terms = []

    # Split camelCase and snake_case
    name_words = re.sub(r"([A-Z])", r" \1", field_name).strip()
    name_words = name_words.replace("_", " ").lower()
    terms.append(name_words)

    # Add individual words (skip short ones)
    for word in name_words.split():
        if len(word) > 3:
            terms.append(word)

    # Extract key terms from description
    if description:
        # Look for quoted examples
        examples = re.findall(r"'([^']+)'|\"([^\"]+)\"", description)
        for ex in examples:
            term = ex[0] or ex[1]
            if term:
                terms.append(term.lower())

        # Add first few words of description
        desc_words = description.lower().split()[:5]
        terms.extend(desc_words)

    return list(dict.fromkeys(terms))  # deduplicate preserving order


def _search_kv(kv_pairs: list, terms: list[str], ftype: str) -> Any:
    """Search key-value pairs for matching field."""
    for kv in kv_pairs:
        key = kv["key"].lower()
        for term in terms:
            if term in key or key in term:
                return _coerce_value(kv["value"], ftype)
    return None


def _search_tables(tables: list, terms: list[str], ftype: str) -> Any:
    """Search tables for matching column header and extract value."""
    for table in tables:
        headers = [str(h).lower() for h in table.get("headers", [])]
        rows = table.get("rows", [])

        # Check if any header matches our search terms
        for col_idx, header in enumerate(headers):
            for term in terms:
                if term in header or header in term:
                    # Found matching column — get first non-empty value
                    for row in rows:
                        if col_idx < len(row) and row[col_idx]:
                            return _coerce_value(row[col_idx], ftype)

        # Also check if table is transposed (first column = field names)
        if rows:
            for row in rows:
                if row and len(row) >= 2:
                    row_label = str(row[0]).lower()
                    for term in terms:
                        if term in row_label or row_label in term:
                            return _coerce_value(row[1], ftype)

    return None


def _search_text(text: str, terms: list[str], ftype: str) -> Any:
    """Search plain text for field value using pattern matching."""
    lines = text.split("\n")

    for line in lines:
        line_lower = line.lower()
        for term in terms:
            if term in line_lower:
                # Try to extract value after colon or equals
                m = re.search(r"[:=]\s*(.+?)(?:\n|$)", line)
                if m:
                    val = m.group(1).strip()
                    if val and len(val) < 200:
                        return _coerce_value(val, ftype)

                # For numeric fields, extract first number from line
                if ftype in ("number", "integer"):
                    nums = re.findall(r"-?\d+\.?\d*", line)
                    if nums:
                        return _coerce_value(nums[0], ftype)

    return None


def _coerce_value(value: str, ftype: str) -> Any:
    """Convert extracted string value to the correct type."""
    if not value or str(value).strip() in ("", "-", "N/A", "n/a", "TBD", "—"):
        return None

    value = str(value).strip()

    if ftype in ("number", "integer", "currency"):
        # Extract numeric part
        m = re.search(r"-?\d[\d,]*\.?\d*", value.replace(",", ""))
        if m:
            try:
                num_str = m.group(0).replace(",", "")
                return int(float(num_str)) if ftype == "integer" else float(num_str)
            except ValueError:
                pass
        return None

    if ftype == "boolean":
        return value.lower() in ("yes", "true", "1", "on", "enabled")

    # String — clean up
    value = re.sub(r"\s+", " ", value).strip()
    return value if value else None
