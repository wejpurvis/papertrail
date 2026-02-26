"""
How to find the format of an unknown API response.
Run this before writing any parsing code.
"""

import asyncio
import xml.etree.ElementTree as ET
import httpx

URL = "https://export.arxiv.org/api/query?search_query=ti:CO2+electroreduction&max_results=1"


async def main():
    async with httpx.AsyncClient() as client:
        response = await client.get(URL)

    raw = response.text

    # Step 1: print raw so you can see exactly what the API returns
    print("=== RAW RESPONSE (first 1000 chars) ===")
    print(raw[:1000])

    # Step 2: parse and find the namespace from the root tag
    # root.tag looks like "{http://www.w3.org/2005/Atom}feed"
    root = ET.fromstring(raw)
    print("\n=== ROOT TAG ===")
    print(root.tag)
    # namespace is the part inside {}
    namespace = root.tag.split("}")[0].strip("{")
    print(f"Namespace to use in your parser: '{namespace}'")

    # Step 3: show all unique tag names one level down
    print("\n=== CHILDREN OF ROOT ===")
    for child in root:
        tag = child.tag.split("}")[-1]
        print(f"  <{tag}>: {(child.text or '').strip()[:80]}")

        # Get the first entry and show title, abstract, authors, and year
        if tag == "entry":
            title = child.find(f"{{{namespace}}}title").text.strip()
            abstract = child.find(f"{{{namespace}}}summary").text.strip()
            published = child.find(f"{{{namespace}}}published").text.strip()
            year = int(published[:4])
            authors = [
                author.find(f"{{{namespace}}}name").text.strip()
                for author in child.findall(f"{{{namespace}}}author")
            ]

            print("\n  === EXTRACTED FIELDS ===")
            print(f"  Title: {title}")
            print(f"  Abstract: {abstract[:80]}...")
            print(f"  Year: {year}")
            print(f"  Authors: {', '.join(authors)}")


if __name__ == "__main__":
    asyncio.run(main())
