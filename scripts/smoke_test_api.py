"""Quick smoke test for Semantic Scholar API."""

import requests

BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,year,citationCount,influentialCitationCount,abstract,externalIds,s2FieldsOfStudy,venue"

# 1. Single paper lookup — "Attention Is All You Need"
print("--- Single paper lookup ---")
r = requests.get(f"{BASE}/paper/ArXiv:1706.03762", params={"fields": FIELDS}, timeout=30)
r.raise_for_status()
p = r.json()
print(f"Title: {p['title']}")
print(f"Year: {p['year']}, Venue: {p['venue']}")
print(f"Citations: {p['citationCount']}, Influential: {p['influentialCitationCount']}")
print(f"ArXiv ID: {p.get('externalIds', {}).get('ArXiv')}")
print(f"Abstract: {p.get('abstract', '')[:150]}...")
print()

# 2. Bulk search — NLP papers 2018-2020
print("--- Bulk search: NLP 2018-2020 ---")
r2 = requests.get(
    f"{BASE}/paper/search/bulk",
    params={
        "query": "natural language processing transformer",
        "fields": FIELDS,
        "limit": 10,
        "year": "2018-2020",
        "fieldsOfStudy": "Computer Science",
    },
    timeout=30,
)
r2.raise_for_status()
d = r2.json()
print(f"Total available: {d['total']}")
print(f"Returned: {len(d.get('data', []))}")
print(f"Has pagination token: {'token' in d}")
print()
for item in d.get("data", [])[:5]:
    has_abs = "+" if item.get("abstract") else "-"
    arxiv = item.get("externalIds", {}).get("ArXiv", "")
    print(f"  [{item['citationCount']:>6}] ({item['year']}) [abs:{has_abs}] [arx:{arxiv or 'none':>12}] {item['title'][:70]}")

# 3. Check a few known breakthrough papers
print()
print("--- Known breakthrough papers ---")
breakthroughs = [
    "ArXiv:1301.3781",  # word2vec
    "ArXiv:1810.04805",  # BERT
    "ArXiv:2005.14165",  # GPT-3
    "ArXiv:1409.0473",  # Attention (Bahdanau)
    "ArXiv:1406.2661",  # GANs
]
for pid in breakthroughs:
    import time
    time.sleep(1)
    r3 = requests.get(f"{BASE}/paper/{pid}", params={"fields": "title,year,citationCount,influentialCitationCount"}, timeout=30)
    if r3.status_code == 200:
        p3 = r3.json()
        print(f"  [{p3['citationCount']:>6} / {p3['influentialCitationCount']:>5} infl] ({p3['year']}) {p3['title']}")
    else:
        print(f"  FAIL {pid}: {r3.status_code}")
