"""
explore_arxiv.py

Run this directly to see what the arXiv API response looks like
after parsing, displayed as clean JSON.

Usage:
    python explore_arxiv.py              # uses local sample XML
    python explore_arxiv.py --live       # hits real arXiv API (if available)
"""

import json
import sys
import xml.etree.ElementTree as ET

# The arXiv API uses Atom feed format with this namespace.
# Every tag is prefixed with it — this is the fiddly part.
ATOM_NS = "http://www.w3.org/2005/Atom"


def parse_arxiv_xml(xml_string: str) -> list[dict]:
    root = ET.fromstring(xml_string)
    papers = []

    for entry in root.findall(f"{{{ATOM_NS}}}entry"):

        # --- ID ---
        # Full URL like "http://arxiv.org/abs/2401.12345v1"
        # We strip to just "2401.12345"
        raw_id = entry.find(f"{{{ATOM_NS}}}id").text.strip()
        arxiv_id = raw_id.split("/abs/")[-1].split("v")[0]

        # --- Title ---
        title = entry.find(f"{{{ATOM_NS}}}title").text.strip()

        # --- Abstract (called "summary" in Atom) ---
        abstract = entry.find(f"{{{ATOM_NS}}}summary").text.strip()

        # --- Authors ---
        authors = [
            author.find(f"{{{ATOM_NS}}}name").text.strip()
            for author in entry.findall(f"{{{ATOM_NS}}}author")
        ]

        # --- Year ---
        published = entry.find(f"{{{ATOM_NS}}}published").text.strip()
        year = int(published[:4])

        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": year,
            }
        )

    return papers


def from_local_file(path: str = "tests/sample_arxiv.xml") -> list[dict]:
    with open(path, "r") as f:
        return parse_arxiv_xml(f.read())


async def from_live_api(
    query: str = "CO2 Electroreduction", max_results: int = 2
) -> list[dict]:
    import httpx

    url = f"https://export.arxiv.org/api/query?search_query=ti:{query}&max_results={max_results}"
    headers = {"User-Agent": "papertrail/0.1 (learning project)"}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
    return parse_arxiv_xml(response.text)


if __name__ == "__main__":
    if "--live" in sys.argv:
        import asyncio

        papers = asyncio.run(from_live_api())
    else:
        papers = from_local_file()

    print(json.dumps(papers, indent=2))
    print(f"\n--- {len(papers)} paper(s) parsed ---")
    print(f"\nFields available: {list(papers[0].keys()) if papers else 'none'}")
    print(f"\nAbstract preview (first 200 chars):\n{papers[0]['abstract'][:200]}...")
