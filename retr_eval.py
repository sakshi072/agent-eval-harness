import glob, json, os, re, sys, time, urllib.request

GOLD, LABEL = sys.argv[1], sys.argv[2]
DOMAIN = os.environ.get("DOMAIN", "langchain-docs")
KS = (3, 5, 10)
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"<\d-hop>", " ", s)).strip().lower()
files = {os.path.basename(p): norm(open(p).read()) for p in glob.glob("raw_docs_v2/*.md")}

def source_pages(contexts):
    pages = set()
    for c in contexts:
        t = norm(c)
        for i in range(0, max(1, len(t) - 60), 150):
            w = t[i:i + 60]
            owners = [f for f, body in files.items() if w in body]
            if len(owners) == 1:
                pages.add(owners[0])
    return pages

def page_of(url):
    path = url.split("docs.langchain.com/", 1)[1]
    return "langchain-docs_" + re.sub(r"[^A-Za-z0-9._-]+", "_", path) + ".md"

def search(q, k):
    req = urllib.request.Request(
        "http://127.0.0.1:8000/search",
        data=json.dumps({"query": q, "domain_name": DOMAIN, "top_k": k}).encode(),
        headers={"Authorization": "Bearer " + os.environ["TOKEN"], "Content-Type": "application/json"},
    )
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=300))
    pages = [page_of(s["file_url"]) for s in d.get("sources", []) if s.get("file_url")]
    return pages, time.time() - t0

g = json.load(open(GOLD))
g = g if isinstance(g, list) else g["samples"]
rows, skipped = [], []
for s in g:
    want = source_pages(s["reference_contexts"])
    if not want:
        skipped.append(s["user_input"]); continue
    row = {"q": s["user_input"], "want": sorted(want)}
    for k in KS:
        got, secs = search(s["user_input"], k)
        row[str(k)] = {"got": got, "secs": secs}
    rows.append(row)

n = len(rows)
print(f"[{LABEL}] {n} questions scored, {len(skipped)} skipped (no distinctive reference text)")
for k in KS:
    hit = sum(bool(set(r['want']) & set(r[str(k)]['got'])) for r in rows)
    full = sum(set(r['want']) <= set(r[str(k)]['got']) for r in rows)
    secs = sum(r[str(k)]['secs'] for r in rows) / n
    print(f"top_k={k:<2} any-page {hit}/{n}   all-pages {full}/{n}   avg {secs:.1f}s")
mrr = sum(next((1 / (i + 1) for i, p in enumerate(r['10']['got']) if p in r['want']), 0) for r in rows) / n
print(f"MRR@10 {mrr:.2f}")
print("no page found at top_k=3:")
for r in rows:
    if not set(r["want"]) & set(r["3"]["got"]):
        print("  -", r["q"][:90])
json.dump(rows, open(f"retr_{LABEL}.json", "w"), indent=1)
