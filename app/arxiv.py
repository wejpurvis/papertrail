"""
Utility module for fetching papers from arXiv. The arXiv API returns XML. Will then parse it to retun extract title, abstract, authors, and year.
"""

import httpx
import xml.etree.ElementTree as ET


async def fetch_arxiv_papers(query: str, max_results: int = 5) -> list[dict]:
    try:
        # Try to fetch from arXiv API
        headers = {"User-Agent": "papertrail/0.1 (learning project)"}
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://export.arxiv.org/api/query",
                params={"search_query": f"all:{query}", "max_results": max_results},
                headers=headers,
            )
            response.raise_for_status()
        return parse_arxiv_xml(response.text)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            # fall back to sample data during development
            return parse_arxiv_xml(open("tests/sample_arxiv.xml").read())
        raise


def parse_arxiv_xml(xml_string: str) -> list[dict]:
    # Parse arxiv XML into tree and extract paper info
    root = ET.fromstring(xml_string)
    papers = []

    # arXiv API uses Atom feed format, so each paper is an <entry> under the root
    ATOM_NS = "http://www.w3.org/2005/Atom"

    for entry in root.findall(f"{{{ATOM_NS}}}entry"):
        # For each entry we want to extract the ID, title, abstract (summary), authors, and published year.

        # Extract arXiv ID from the <id> tag (strip to just the number)
        # Full URL looks like <id>https://arxiv.org/api/KfM71HeO60fj1NkojYDkVkxKaw8</id>
        # Want just "KfM71HeO60fj1NkojYDkVkxKaw8"
        raw_id = entry.find(f"{{{ATOM_NS}}}id").text.strip()
        arxiv_id = raw_id.split("/abs/")[-1].split("v")[0]

        # Extract title from <title> tag
        title = entry.find(f"{{{ATOM_NS}}}title").text.strip()

        # Extract abstract from <summary> tag
        abstract = entry.find(f"{{{ATOM_NS}}}summary").text.strip()

        # Authors are in <author><name> tags, need to extract all of them
        authors = [
            author.find(f"{{{ATOM_NS}}}name").text.strip()
            for author in entry.findall(f"{{{ATOM_NS}}}author")
        ]

        # Extract published year from <published> tag (format is like "2024-01-15T12:34:56Z")
        published = entry.find(f"{{{ATOM_NS}}}published").text.strip()
        year = int(published[:4])

        # Append extracted info as a dict to our papers list
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
