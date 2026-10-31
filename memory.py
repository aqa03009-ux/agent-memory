#!/usr/bin/env python3
"""
memory.py — a local-first memory vault for AI agents.

Design philosophy: the embedding model is NEVER kept warm. Every command:
  1. starts -> 2. loads the model (only when needed) -> 3. does the work -> 4. exits.
RAM/VRAM is released immediately. Vectors are cached in SQLite, so the model is
not opened again unless something actually changed.

Commands:
  reindex [--dim 256] [--with-images]    scan the vault, cache embeddings
  search "query" [--top 3] [--full]      token-cheap summary (default)
  show <path>                            print a full note
  list [--tag setup]                   list notes
  add --title .. --tags .. --file .. [--to setups|projects|daily]
  graph                                  write the wikilink+tag graph to relations.md
  assets                                 inventory media under assets/
  status                                 index/model status and resource usage
  last [--n 1]                           newest session handoff notes
  open [--n 5]                           unclaimed session handoff notes
  claim <path>                           mark a handoff note as claimed (claim-once)
  mcp                                    stdio MCP server (JSON-RPC 2.0)
"""
import argparse, os, re, sqlite3, sys, hashlib, time, json, math
from pathlib import Path

VAULT = Path(os.environ.get("AGENT_MEMORY_VAULT", Path.home() / "memory"))
DB = Path(os.environ.get("AGENT_MEMORY_DB", VAULT / ".vectors" / "memory.db"))
MODEL_ID = os.environ.get("AGENT_MEMORY_MODEL", "Qwen/Qwen3-Embedding-0.6B")
# Qwen3-Embedding expects an instruction on the query side (asymmetric retrieval).
# If you switch to a model without instructions, override or clear this prefix.
QUERY_INSTRUCT = os.environ.get(
    "AGENT_MEMORY_QUERY_INSTRUCT",
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
)
DIM = int(os.environ.get("AGENT_MEMORY_DIM", "256"))
AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov"}

# IMPORTANT: must be set before torch is imported.
# During reindex, per-layer projection temporaries allocate gigabytes; with
# fragmentation the "reserved but unallocated" pool builds up and we hit OOM.
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Reindex resilience (do not die while another job is holding the GPU):
EMB_CHUNK = int(os.environ.get("AGENT_MEMORY_CHUNK", "16"))   # documents per slice
EMB_BATCH = int(os.environ.get("AGENT_MEMORY_BATCH", "8"))    # batch inside a slice (halved on OOM)

# ---------- DB ----------
FTS_OK = True
def db_conn():
    global FTS_OK
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB), timeout=30)
    c.execute("PRAGMA busy_timeout=30000")
    c.execute("""CREATE TABLE IF NOT EXISTS docs
      (id TEXT PRIMARY KEY, title TEXT, path TEXT, tags TEXT, content TEXT, mtime REAL, kind TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS vectors
      (doc_id TEXT PRIMARY KEY, dim INTEGER, vec BLOB)""")
    c.execute("""CREATE TABLE IF NOT EXISTS links
      (src TEXT, dst TEXT, type TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS meta
      (key TEXT PRIMARY KEY, value TEXT)""")
    # FTS5 keyword index: lets most searches run without touching the model.
    # If the local sqlite3 lacks FTS5, the vector/keyword fallback still works.
    try:
        c.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                      title, tags, content, tokenize='unicode61')""")
    except sqlite3.OperationalError as e:
        FTS_OK = False
        print(f"[memory] FTS5 unavailable ({e}) — vector search only", file=sys.stderr)
    return c

def read_md(p: Path):
    t = p.read_text(encoding="utf-8", errors="ignore")
    m_title = re.search(r"^#\s+(.+)", t, re.M)
    title = m_title.group(1).strip() if m_title else p.stem
    m_tags = re.search(r"^tags:\s*(.+)", t, re.M | re.I)
    tags = m_tags.group(1).strip() if m_tags else ""
    links = re.findall(r"\[\[([^\]]+)\]\]", t)
    return title, tags, t[:8000], links

# ---------- MODEL (lazy) ----------
_model_cache = None
def get_model():
    """Loaded only on demand. When the process exits, RAM/VRAM is released.
    NOTES (measured 2026-10):
    - ROCm + default dtype (bf16) produced NaN vectors on this machine.
    - float32 is clean; CPU is the default to avoid burning ~2.5 GB VRAM on
      every invocation just to fall back.
    - Force a device with AGENT_MEMORY_DEVICE=cpu|auto.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        want = os.environ.get("AGENT_MEMORY_DEVICE", "cpu")
        devs = ["cpu"] if want == "cpu" else (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
        t0 = time.time()
        for dev in devs:
            try:
                print(f"[memory] loading model: {MODEL_ID} ({dev}, float32)...", file=sys.stderr)
                m = SentenceTransformer(MODEL_ID, device=dev,
                    model_kwargs={"dtype": torch.float32})
                import numpy as _np
                probe = m.encode(["memory probe"], normalize_embeddings=True, show_progress_bar=False)
                if bool(_np.isnan(probe).any()):
                    print(f"[memory] {dev} produced NaN, trying the next device.", file=sys.stderr)
                    del m
                    continue
                _model_cache = m
                print(f"[memory] model ready ({dev}, {time.time()-t0:.1f}s). "
                      "It will be released when the process exits.", file=sys.stderr)
                return _model_cache
            except Exception as e:
                print(f"[memory] {dev} failed ({e}), trying the next device.", file=sys.stderr)
                continue
        print("[memory] no embedding model available, keyword fallback", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[memory] no embedding model available, keyword fallback: {e}", file=sys.stderr)
        return None

def _is_oom(e) -> bool:
    return ("out of memory" in str(e).lower()) or ("OutOfMemory" in type(e).__name__)


def _encode_resilient(model, texts):
    """Embed in slices; on OOM halve the batch and retry the same slice.
    Why: when another GPU job (training or inference) holds the device, a single large
    encode call OOMs on the per-layer projection temporaries and the whole
    reindex used to die. Now it keeps shrinking the batch and finishes."""
    import numpy as _np
    try:
        import torch
    except Exception:
        torch = None
    out, bs, i = [], EMB_BATCH, 0
    while i < len(texts):
        chunk = texts[i:i + EMB_CHUNK]
        try:
            v = model.encode(chunk, normalize_embeddings=True,
                             show_progress_bar=False, batch_size=bs)
        except Exception as e:
            if _is_oom(e) and bs > 1:
                bs = max(1, bs // 2)
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[memory] GPU OOM — retrying with batch {bs} "
                      f"(documents {i}-{i+len(chunk)})", file=sys.stderr)
                continue
            raise
        out.append(_np.asarray(v, dtype=_np.float32))
        i += EMB_CHUNK
    return _np.vstack(out) if out else _np.zeros((0, DIM), dtype=_np.float32)


def embed_texts(model, texts, dim=DIM, is_query=False):
    if model is None:
        return None
    texts = [_neutralize_mm_tokens(t) for t in texts]   # <|audio|> etc. used to break reindex
    # Qwen3-Embedding: instruction on the query, none on documents (asymmetric retrieval).
    if is_query and "qwen" in MODEL_ID.lower():
        texts = [QUERY_INSTRUCT + t for t in texts]
    vecs = _encode_resilient(model, texts)
    import numpy as _np
    vecs = _np.asarray(vecs, dtype=_np.float32)
    if vecs.size and vecs.shape[1] > dim:  # MRL truncation + re-normalization (official recipe)
        vecs = vecs[:, :dim]
        vecs = vecs / (_np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    return vecs

def embed_image(model, img_path: Path, dim=DIM):
    """Image embedding (full vision encoder). Same space as text -> cross-modal search."""
    from PIL import Image
    import numpy as _np
    img = Image.open(img_path).convert("RGB")
    v = model.encode([img], normalize_embeddings=True, show_progress_bar=False)
    v = _np.asarray(v, dtype=_np.float32)
    if v.shape[1] > dim:
        v = v[:, :dim]
        v = v / (_np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v[0]


# --- media (audio/video) embeddings — optional, requires a multimodal model ---
_MM_TOKEN_RE = re.compile(r"<\|(image|video|audio)\|>")

def _neutralize_mm_tokens(t):
    """The processor treats `<|audio|>`/`<|image|>`/`<|video|>` as special tokens;
    if they appear in plain text the encoder raises ValueError and reindex dies."""
    return _MM_TOKEN_RE.sub(lambda m: f"< {m.group(1)} >", t)

def _l2_trunc(vecs, dim):
    """MRL: keep the first `dim` dimensions, then re-normalize (official recipe)."""
    import numpy as _np
    v = _np.asarray(vecs, dtype=_np.float32).reshape(1, -1)
    if v.shape[1] > dim:
        v = v[:, :dim]
    return v / (_np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

def _encode_checked(model, inp, tries=3):
    """On ROCm the audio path occasionally returns an all-zero vector. That is a
    silent failure (the record becomes unsearchable), so retry on zero/NaN."""
    import numpy as _np
    for _ in range(tries):
        v = _np.asarray(model.encode(inp, normalize_embeddings=True,
                                     show_progress_bar=False), dtype=_np.float32).reshape(1, -1)
        if bool(_np.isfinite(v).all()) and float(_np.linalg.norm(v)) > 1e-6:
            return v
    raise RuntimeError(f"embedding returned zero/NaN ({tries} attempts)")

def _decode_audio(path, sr=16000):
    """torchcodec does not work on this machine (its CUDA build wants libnvrtc.so.13).
    The model card expects mono 16 kHz, so decode with soundfile + librosa."""
    import soundfile as sf, librosa
    a, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr0 != sr:
        a = librosa.resample(a, orig_sr=sr0, target_sr=sr)
    return {"array": a.astype("float32"), "sampling_rate": sr}

def embed_audio(model, path, dim=DIM):
    """Audio embedding (audio encoder). Same space as text -> cross-modal search."""
    v = _encode_checked(model, {"audio": _decode_audio(path)})
    return _l2_trunc(v, dim)[0]

def _video_frames(path, fps=1, max_frames=32):
    """Model-card default: 1 fps, at most 32 frames. Uses ffmpeg instead of torchcodec."""
    import subprocess, tempfile, glob
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path),
                        "-vf", f"fps={fps},scale=448:-2", "-frames:v", str(max_frames),
                        f"{td}/f_%04d.png"], check=True, capture_output=True)
        return [Image.open(f).convert("RGB") for f in sorted(glob.glob(f"{td}/f_*.png"))]

def embed_video(model, path, dim=DIM, fps=1, max_frames=32):
    """Video = a 1 fps frame sequence fed to the vision encoder (no separate video tower)."""
    frames = _video_frames(path, fps, max_frames)
    if not frames:
        raise RuntimeError(f"no frames extracted: {path}")
    v = _encode_checked(model, {"video": frames})
    return _l2_trunc(v, dim)[0]

# ---------- COMMANDS ----------
IGNORE_FILE = VAULT / ".memoryignore"

def load_ignores():
    """`.memoryignore`: one glob or directory prefix per line ('#' for comments).
    Raw corpora (e.g. YouTube transcripts) are not memory and stay out of the index."""
    pats = []
    if IGNORE_FILE.exists():
        for line in IGNORE_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                pats.append(line.rstrip("/"))
    return pats

def is_ignored(rel, pats):
    import fnmatch
    for p in pats:
        if rel == p or rel.startswith(p + "/") or fnmatch.fnmatch(rel, p):
            return True
    return False

def _ignored_count():
    all_md = [p for p in VAULT.rglob("*.md")
              if ".obsidian" not in str(p) and ".vectors" not in str(p) and "/.trash/" not in str(p)]
    return max(0, len(all_md) - len(iter_vault_mds()))

def iter_vault_mds():
    pats = load_ignores()
    files = sorted(VAULT.rglob("*.md"))
    out = []
    for f in files:
        s = str(f)
        if ".obsidian" in s or ".vectors" in s or "/.trash/" in s:
            continue
        if pats and is_ignored(str(f.relative_to(VAULT)), pats):
            continue
        out.append(f)
    return out

def _acquire_lock():
    """One reindex at a time (plugins can trigger a background reindex)."""
    import fcntl
    DB.parent.mkdir(parents=True, exist_ok=True)
    f = open(DB.parent / "reindex.lock", "w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except BlockingIOError:
        f.close()
        return None

def cmd_reindex(args):
    """INCREMENTAL reindex: only new/changed files are embedded.
    If nothing changed the MODEL IS NEVER OPENED (0 VRAM, 0 RAM, ~instant).
    The first version re-embedded all 453 documents on every call and could OOM
    a 16 GB machine while another job was running. Use --force for a full rebuild."""
    lock = _acquire_lock()
    if lock is None:
        print("[memory] another reindex is running — skipping this one.", file=sys.stderr)
        return
    try:
        return _reindex_locked(args)
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            print("[memory] database is locked (another process) — try again.", file=sys.stderr)
        else:
            raise
    finally:
        try:
            lock.close()
        except Exception:
            pass

def _reindex_locked(args):
    files = iter_vault_mds()
    con = db_conn()
    force = getattr(args, "force", False)

    # If the model or dimension changed, old vectors live in a different space.
    # Keep model identity in the DB and force a full rebuild when it changes.
    row = con.execute("SELECT value FROM meta WHERE key='model'").fetchone()
    stored_model = row[0] if row else None
    row = con.execute("SELECT value FROM meta WHERE key='dim'").fetchone()
    stored_dim = row[0] if row else None
    if stored_model != MODEL_ID or stored_dim != str(args.dim):
        if stored_model is not None:
            print(f"[memory] model/dim changed ({stored_model}/{stored_dim} -> "
                  f"{MODEL_ID}/{args.dim}); full reindex", file=sys.stderr)
        force = True

    old = {p: (mid, mt) for mid, p, mt in con.execute("SELECT id, path, mtime FROM docs")}
    have_vec = {r[0] for r in con.execute("SELECT doc_id FROM vectors")}

    con.execute("DELETE FROM links")   # the link graph is rebuilt from scratch: cheap, no model
    seen, todo_texts, todo_ids, kept = set(), [], [], 0
    changed, added = [], []
    for f in files:
        rel = str(f.relative_to(VAULT))
        seen.add(rel)
        doc_id = hashlib.sha1(rel.encode()).hexdigest()[:16]
        mt = f.stat().st_mtime
        title, tags, content, links = read_md(f)
        for lk in links:
            con.execute("INSERT INTO links VALUES (?,?,?)", (rel, lk, "wikilink"))
        for tg in re.findall(r"#(\w+)", tags):
            con.execute("INSERT INTO links VALUES (?,?,?)", (rel, tg, "tag"))
        if (not force) and rel in old and abs(old[rel][1] - mt) < 1e-6 and doc_id in have_vec:
            kept += 1
            continue
        kind = rel.split("/")[0] if "/" in rel else "root"
        con.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?,?)",
            (doc_id, title, rel, tags, content, mt, kind))
        todo_texts.append(f"{title}\n{tags}\n{content[:4000]}")
        todo_ids.append(doc_id)
        (changed if rel in old else added).append(rel)

    gone = 0
    for rel, (mid, _) in old.items():          # rows for files removed from the vault
        if rel not in seen:
            con.execute("DELETE FROM docs WHERE id=?", (mid,))
            con.execute("DELETE FROM vectors WHERE doc_id=?", (mid,))
            gone += 1

    media_jobs, media_ok = [], 0
    if (getattr(args, "with_audio", False) or getattr(args, "with_video", False)) and "qwen" in MODEL_ID.lower():
        print("[memory] WARNING: the Qwen text model does not support audio/video embeddings; "
              "run a separate reindex with AGENT_MEMORY_MODEL=google/embeddinggemma-2 for media. "
              "Media skipped in this pass.", file=sys.stderr)
        args.with_audio = False
        args.with_video = False
    specs = []
    if getattr(args, "with_images", False):
        specs.append(("images", {".jpg", ".jpeg", ".png", ".webp"}, "img_", embed_image, "[image]"))
    if getattr(args, "with_audio", False):
        specs.append(("audio", AUDIO_EXT, "aud_", embed_audio, "[audio]"))
    if getattr(args, "with_video", False):
        specs.append(("videos", VIDEO_EXT, "vid_", embed_video, "[video]"))
    for sub, exts, prefix, fn, label in specs:
        d = VAULT / "assets" / sub
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if not f.is_file() or f.suffix.lower() not in exts:
                continue
            rel = str(f.relative_to(VAULT))
            aid = prefix + hashlib.sha1(rel.encode()).hexdigest()[:12]
            if (not force) and aid in have_vec and aid in old and abs(old[aid][1] - f.stat().st_mtime) < 1e-6:
                media_ok += 1
                continue
            media_jobs.append((f, aid, rel, fn, label))

    if not todo_texts and not media_jobs:
        ensure_fts(con)          # repair FTS if it drifted, without opening the model
        con.commit(); con.close()
        print(f"[memory] reindex: nothing changed ({kept} documents up to date) — "
              f"model NOT opened, zero resources used.", file=sys.stderr)
        return

    model = get_model()
    print(f"[memory] {len(files)} md scanned | {kept} current | "
          f"{len(added)} new | {len(changed)} changed | {gone} deleted | "
          f"{media_ok + len(media_jobs)} media", file=sys.stderr)
    vecs = embed_texts(model, todo_texts, dim=args.dim) if todo_texts else None
    if vecs is not None:
        for doc_id, v in zip(todo_ids, vecs):
            con.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?)",
                (doc_id, int(v.shape[0]), v.astype("<f4").tobytes()))
    elif todo_texts:
        for doc_id in todo_ids:          # do not keep a stale vector: drop it and re-embed later
            con.execute("DELETE FROM vectors WHERE doc_id=?", (doc_id,))
        print("[memory] WARNING: no model — changed documents could not be re-embedded, "
              "old vectors dropped (keyword fallback applies; the next reindex will embed them).", file=sys.stderr)

    if media_jobs and model is not None:
        for f, aid, rel, fn, label in media_jobs:
            try:
                v = fn(model, f, dim=args.dim)
                con.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?,?)",
                    (aid, f"{label} {f.name}", rel, f"#{label.strip('[]')} #asset",
                     f"{label} asset: {f.name} ({f.stat().st_size // 1024}KB)",
                     f.stat().st_mtime, "asset"))
                con.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?)",
                    (aid, int(v.shape[0]), v.astype("<f4").tobytes()))
            except Exception as e:
                print(f"[memory] skipped {label} {f.name}: {e}", file=sys.stderr)

    con.execute("INSERT OR REPLACE INTO meta VALUES ('model', ?)", (MODEL_ID,))
    con.execute("INSERT OR REPLACE INTO meta VALUES ('dim', ?)", (str(args.dim),))
    con.commit()
    n_fts = rebuild_fts(con)     # refresh the full-text index (no model needed)
    total = con.execute("SELECT count(*) FROM docs").fetchone()[0]
    embedded = len(todo_ids) if vecs is not None else 0
    extra = f" + {len(media_jobs)} media" if (media_jobs and model is not None) else ""
    print(f"[memory] reindex complete: {total} documents "
          f"({embedded} text{extra} embedded this pass), "
          f"fts={n_fts} "
          f"dim={args.dim if vecs is not None else 'keyword'} -> {DB}", file=sys.stderr)
    print("[memory] the model leaves with the process: VRAM is free again.", file=sys.stderr)
    con.close()

# ---------- FTS5 + RRF ----------
STOPWORDS = {
    # english
    "the", "and", "for", "how", "what", "why", "when", "where", "which", "is",
    "are", "was", "were", "with", "this", "that", "from", "you", "your", "not",
    "can", "should", "would", "there", "their", "have", "has", "get", "use",
    # turkish
    "ne", "nasıl", "neden", "niçin", "için", "ile", "bir", "bu", "ve", "veya",
    "var", "yok", "mı", "mi", "mu", "mü", "kaç", "hangi", "kim", "daha", "çok",
    "gibi", "kadar", "sonra", "önce", "ise", "ama", "her", "tüm", "şey", "şeyi",
}

def _norm(s):
    """Lowercase + diacritic folding (matches FTS unicode61 behavior)."""
    import unicodedata
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))

def query_tokens(query):
    """Informative tokens (stopwords removed)."""
    toks = [t.lower() for t in re.findall(r"\w+", query, re.UNICODE) if len(t) > 1]
    toks = [t for t in toks if t not in STOPWORDS]
    return list(dict.fromkeys(toks))[:12]

def rebuild_fts(con):
    """Rebuild the full-text index from the docs table (NO model, ~ms)."""
    if not FTS_OK:
        return 0
    con.execute("DELETE FROM docs_fts")
    con.execute("INSERT INTO docs_fts(rowid,title,tags,content) "
                "SELECT rowid,title,tags,content FROM docs")
    con.commit()
    return con.execute("SELECT count(*) FROM docs_fts").fetchone()[0]

def ensure_fts(con):
    """Rebuild FTS when it drifted out of sync with docs (self-healing)."""
    if not FTS_OK:
        return
    try:
        n_docs = con.execute("SELECT count(*) FROM docs").fetchone()[0]
        n_fts = con.execute("SELECT count(*) FROM docs_fts").fetchone()[0]
    except sqlite3.OperationalError:
        return
    if n_fts != n_docs:
        rebuild_fts(con)

def fts_search(con, query, limit=60):
    """Keyword search -> [(doc_id, rank, bm25)] (rank 1 = best).
    No prefix matching for short tokens (<4 chars): 'cap' should not match 'capable'.
    Prefix matching is on for longer tokens: 'youtube' -> "youtube's", 'python' -> 'python3'."""
    if not FTS_OK:
        return []
    toks = query_tokens(query)
    if not toks:
        return []
    parts = []
    for t in toks:
        if len(t) >= 4:
            parts.append(f'"{t}"*')     # prefix
        else:
            parts.append(f'"{t}"')      # exact word
    match = " OR ".join(parts)
    try:
        rows = con.execute(
            "SELECT rowid, bm25(docs_fts) FROM docs_fts WHERE docs_fts MATCH ? "
            "ORDER BY bm25(docs_fts) LIMIT ?", (match, limit)).fetchall()
    except sqlite3.OperationalError as e:
        print(f"[memory] FTS query error: {e}", file=sys.stderr)
        return []
    out = []
    for rowid, score in rows:
        r = con.execute("SELECT id FROM docs WHERE rowid=?", (rowid,)).fetchone()
        if r:
            out.append((r[0], len(out) + 1, score))
    return out

def fts_confidence(con, hits, query, by_id, topn=8):
    """How trustworthy is the keyword stream on its own?
    IDF-weighted coverage: what share of the query's informative tokens does the
    best document contain? 0.0-1.0. At >=0.55 keyword search alone is enough
    (no need to open the model)."""
    toks = query_tokens(query)
    if not toks or not hits:
        return 0.0, {}
    n_docs = con.execute("SELECT count(*) FROM docs").fetchone()[0] or 1
    idf = {}
    for t in toks:
        try:
            df = con.execute("SELECT count(*) FROM docs_fts WHERE docs_fts MATCH ?",
                             (f'"{t}"',)).fetchone()[0]
        except sqlite3.OperationalError:
            df = 0
        idf[t] = 0.0 if df == 0 else math.log(1 + n_docs / df)
    total = sum(idf.values())
    if total <= 0:
        return 0.0, {}
    best, best_cov = 0.0, {}
    for doc_id, _rank, _score in hits[:topn]:
        row = by_id.get(doc_id)
        if not row:
            continue
        hay = _norm(f"{row[1]} {row[3]} {row[4]}")
        matched = {t: idf[t] for t in toks if _norm(t) in hay}
        cov = sum(matched.values()) / total
        if cov > best:
            best, best_cov = cov, matched
    return best, best_cov

def rrf_fuse(vec_ids, fts_ids, k=60):
    """Reciprocal Rank Fusion: merge the vector and FTS rankings into one list.
    score(d) = sum(1 / (k + rank_i(d))) — ranks start at 1 inside each stream."""
    scores = {}
    for ids in (vec_ids, fts_ids):
        for i, doc_id in enumerate(ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + i + 1)
    return sorted(((v, k2) for k2, v in scores.items()), key=lambda x: -x[0])

def related_of(con, path):
    rows = list(con.execute("SELECT dst,type FROM links WHERE src=? LIMIT 8", (path,)))
    return [f"{t}:{d}" for d, t in rows]

def do_search(query, top=3, full=False, semantic=False, fts_only=False):
    """Hybrid search: FTS5 (keywords) + vectors (meaning) -> RRF fusion.
    Default 'lite' mode: if the keyword stream is confident the model is NEVER
    opened (~0.01 s). When keyword confidence is very low, semantic search opens
    the model. Returns (lines, info_line)."""
    t0 = time.time()
    con = db_conn()
    rows = list(con.execute("SELECT id,title,path,tags,content,kind FROM docs"))
    if not rows:
        return ["[memory] empty — run `memory.py reindex` first."], ""
    by_id = {r[0]: r for r in rows}
    vec_rows = {r[0]: r[1] for r in con.execute("SELECT doc_id, vec FROM vectors")}

    # --- stream 1: FTS5 (keywords) ---
    ensure_fts(con)
    fts_hits = fts_search(con, query)
    fts_ids = [d for d, _r, _s in fts_hits]
    conf, matched = fts_confidence(con, fts_hits, query, by_id)

    # --- stream 2: vectors (meaning) ---
    mode = "fts"
    if semantic:
        mode = "hybrid"
    elif not fts_only and (not fts_ids or conf < 0.25):
        mode = "hybrid"          # keyword confidence very low -> open the semantic stream
    vec_ids = []
    if mode == "hybrid":
        if not vec_rows:
            mode = "fts"
        else:
            model = get_model()
            if model:
                import numpy as _np
                qv = embed_texts(model, [query], dim=DIM, is_query=True)[0]
                scored = []
                for doc_id, vec in vec_rows.items():
                    d = len(vec) // 4
                    if d == 0:
                        continue
                    v = _np.frombuffer(vec, dtype="<f4", count=d)
                    dd = min(d, len(qv))
                    if dd == 0:
                        continue
                    scored.append((float(v[:dd] @ qv[:dd]), doc_id))
                scored.sort(key=lambda x: -x[0])
                vec_ids = [d for _, d in scored]
            else:
                mode = "fts"

    # --- fusion ---
    fused = rrf_fuse(vec_ids, fts_ids)
    dt = time.time() - t0
    info = (f"[memory] search: mode={mode} | fts={len(fts_ids)} (confidence={conf:.2f}) "
            f"vector={len(vec_ids)} | {dt:.2f}s"
            + (" (model loaded)" if mode == "hybrid" else " (model not opened)"))
    if not fused:
        return ([f"[memory] no match: '{query}'",
                 "[memory] try: --semantic (meaning), --top N, or fewer words"], info)
    out = []
    for score, doc_id in fused[:top]:
        row = by_id.get(doc_id)
        if not row:
            continue
        _id, title, path, tags, content, kind = row
        rel = related_of(con, path)
        out.append("")
        out.append(f"=== {title} [{path}] (rrf={score:.4f}) ===")
        out.append(f"kind={kind} | tags={tags} | links={', '.join(rel) if rel else '-'}")
        if full:
            out.append(content[:4000])
        else:
            heads = re.findall(r"^##\s+(.+)", content, re.M)[:5]
            first = (content[:400].strip().replace("\n", " "))
            out.append(first + "...")
            if heads:
                out.append("sections: " + " | ".join(heads))
        out.append(f'--> full note: memory.py show "{path}"')
    out.append("")
    out.append("[memory] summary mode = fewer tokens. Use --full or show for detail.")
    con.close()
    return out, info

def cmd_search(args):
    lines, info = do_search(args.query, args.top, args.full, args.semantic, args.fts_only)
    for l in lines:
        print(l)
    if info:
        print(info, file=sys.stderr)

def cmd_show(args):
    p = VAULT / args.path
    print(p.read_text(encoding="utf-8", errors="ignore")[:8000])

def cmd_list(args):
    con = db_conn()
    rows = list(con.execute("SELECT title,path,tags,kind FROM docs"))
    if args.tag:
        rows = [r for r in rows if args.tag.lower() in (r[2] or "").lower()]
    for title, path, tags, kind in sorted(rows):
        print(f"- [{kind}] {title} [{path}] {tags}")
    con.close()

def cmd_add(args):
    src = Path(args.file)
    if not src.exists():
        print(f"[memory] file not found: {src}")
        return
    dst_dir = VAULT / args.to
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    body = src.read_text(encoding="utf-8", errors="ignore")
    if not body.startswith("#"):
        body = f"# {args.title}\n\ntags: {args.tags}\n\n" + body
    dst.write_text(body, encoding="utf-8")
    print(f"[memory] written: {dst}\nnext: memory.py reindex")

def cmd_last(args):
    """Show the newest session handoff notes (first call of a new session)."""
    d = VAULT / "sessions"
    files = sorted(d.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True) if d.exists() else []
    if not files:
        print("[memory] sessions/ is empty — no handoff notes yet.")
        return
    for f in files[:args.n]:
        import datetime
        mt = datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"=== {f.name}  ({mt}) ===")
        print(f.read_text(encoding="utf-8", errors="ignore")[:args.chars])
        print()

# ---------- MCP server ----------
class SimpleNS:
    """Minimal argparse.Namespace-like container."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


MCP_TOOLS = [
    {"name": "memory_search",
     "description": "Search the canonical memory vault. Hybrid keyword (FTS5) + vector. "
                    "Returns a token-cheap summary by default; use top/full/semantic as needed.",
     "inputSchema": {"type": "object",
        "properties": {"query": {"type": "string", "description": "search query (any language)"},
                       "top": {"type": "integer", "default": 3, "description": "number of results"},
                       "full": {"type": "boolean", "default": False, "description": "return full text"},
                       "semantic": {"type": "boolean", "default": False, "description": "force semantic search (slower)"},
                       "fts": {"type": "boolean", "default": False, "description": "keyword search only (fast)"}},
        "required": ["query"]}},
    {"name": "memory_show",
     "description": "Return the full contents of a note (path relative to the vault root).",
     "inputSchema": {"type": "object",
        "properties": {"path": {"type": "string", "description": "e.g. setups/example.md"}},
        "required": ["path"]}},
    {"name": "memory_list",
     "description": "List every note in the vault, optionally filtered by tag.",
     "inputSchema": {"type": "object",
        "properties": {"tag": {"type": "string", "description": "tag filter (empty = all)"}}}},
    {"name": "memory_add",
     "description": "Write a new persistent note and refresh the index (writes the file + reindex).",
     "inputSchema": {"type": "object",
        "properties": {"title": {"type": "string"}, "tags": {"type": "string"},
                       "body": {"type": "string", "description": "note body (markdown)"},
                       "to": {"type": "string", "default": "daily",
                              "description": "folder: daily | setups | decisions | projects | sessions"}},
        "required": ["title", "body"]}},
    {"name": "memory_last",
     "description": "Newest session handoff note (use it as the first call of a new session).",
     "inputSchema": {"type": "object", "properties": {"n": {"type": "integer", "default": 1}}}},
    {"name": "memory_open",
     "description": "List open (unclaimed) session handoff notes: what work is still waiting?",
     "inputSchema": {"type": "object", "properties": {"n": {"type": "integer", "default": 5}}}},
    {"name": "memory_claim",
     "description": "Mark a handoff note as claimed (claim-once: two agents never take the same job).",
     "inputSchema": {"type": "object",
        "properties": {"path": {"type": "string", "description": "e.g. sessions/2026-01-01-task.md"}},
        "required": ["path"]}},
    {"name": "memory_status",
     "description": "Vault/index/model status (note count, FTS, vectors, model info).",
     "inputSchema": {"type": "object", "properties": {}}},
]

def _mcp_call(name, a):
    if name == "memory_search":
        lines, info = do_search(a.get("query", ""), int(a.get("top", 3)),
                                bool(a.get("full", False)), bool(a.get("semantic", False)),
                                bool(a.get("fts", False)))
        return (info + "\n" if info else "") + "\n".join(lines)
    if name == "memory_show":
        p = VAULT / a.get("path", "")
        if not p.exists():
            raise FileNotFoundError(f"not found: {p}")
        return p.read_text(encoding="utf-8", errors="ignore")[:12000]
    if name == "memory_list":
        con = db_conn()
        rows = list(con.execute("SELECT title,path,tags,kind FROM docs"))
        con.close()
        tag = (a.get("tag") or "").lower()
        out = [f"- [{k}] {t} [{pth}] {tg}" for t, pth, tg, k in sorted(rows)
               if not tag or tag in (tg or "").lower()]
        return "\n".join(out) or "(empty)"
    if name == "memory_add":
        to = a.get("to", "daily") or "daily"
        title = a.get("title", "note")
        body = a.get("body", "")
        if not body.strip():
            raise ValueError("body is empty")
        dst_dir = VAULT / to
        dst_dir.mkdir(parents=True, exist_ok=True)
        import re as _re
        slug = _re.sub(r"[^\w\-]+", "-", title.lower()).strip("-")[:50] or "note"
        dst = dst_dir / f"{slug}.md"
        n = 2
        while dst.exists():
            dst = dst_dir / f"{slug}-{n}.md"
            n += 1
        text = f"# {title}\n\ntags: {a.get('tags','')}\n\n{body}"
        dst.write_text(text, encoding="utf-8")
        subprocess_run = __import__("subprocess").run(
            [sys.executable, str(Path(__file__).resolve()), "reindex"],
            capture_output=True, text=True, timeout=600)
        return f"written: {dst}\n" + subprocess_run.stderr[-500:]
    if name == "memory_last":
        d = VAULT / "sessions"
        files = sorted(d.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True) if d.exists() else []
        if not files:
            return "(sessions/ is empty)"
        out = []
        for f in files[:int(a.get("n", 1))]:
            out.append(f"=== {f.name} ===\n" + f.read_text(encoding="utf-8", errors="ignore")[:6000])
        return "\n\n".join(out)
    if name == "memory_open":
        import io, contextlib as _ctx
        buf = io.StringIO()
        with _ctx.redirect_stdout(buf):
            cmd_open(SimpleNS(n=int(a.get("n", 5))))
        return buf.getvalue()
    if name == "memory_claim":
        import io, contextlib as _ctx
        buf = io.StringIO()
        with _ctx.redirect_stdout(buf):
            cmd_claim(SimpleNS(path=a.get("path", "")))
        return buf.getvalue()
    if name == "memory_status":
        con = db_conn()
        n_docs = con.execute("SELECT count(*) FROM docs").fetchone()[0]
        n_vec = con.execute("SELECT count(*) FROM vectors").fetchone()[0]
        n_fts = con.execute("SELECT count(*) FROM docs_fts").fetchone()[0] if FTS_OK else 0
        con.close()
        return (f"vault: {VAULT} ({len(iter_vault_mds())} md)\n"
                f"index: docs={n_docs} vectors={n_vec} fts={n_fts}\n"
                f"model: {MODEL_ID} (lazy-load, opened only when needed)")
    raise ValueError(f"unknown tool: {name}")

def _is_claimed(head):
    """Accept both the English and the original Turkish claim marker."""
    return bool(re.search(r"^(status:\s*claimed|durum:\s*devralindi)", head, re.M | re.I))

def cmd_open(args):
    """List open (unclaimed) session handoff notes — the new session's to-do list."""
    d = VAULT / "sessions"
    files = sorted(d.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True) if d.exists() else []
    open_notes, closed = [], 0
    for f in files:
        try:
            head = f.read_text(encoding="utf-8", errors="ignore")[:2500]
        except Exception:
            continue
        if _is_claimed(head):
            closed += 1
            continue
        m = re.search(r"^#\s+(.+)$", head, re.M)
        title = m.group(1).strip() if m else f.stem
        m2 = re.search(r"^(?:recorded|kayit):\s*(.+)$", head, re.M)
        m3 = re.search(r"^session:\s*(.+)$", head, re.M)
        open_notes.append((f, title, (m2.group(1).strip() if m2 else ""), (m3.group(1).strip() if m3 else "")))
    if not open_notes:
        print(f"[memory] no open handoff notes ({closed} closed).")
        return
    print(f"[memory] OPEN handoff notes ({len(open_notes)} open, {closed} closed):")
    for f, title, recorded, ses in open_notes[:args.n]:
        print(f"\n--- [{f.name}]")
        print(f"    {title[:110]}")
        print(f"    recorded: {recorded}  session: {ses}")
    print("\n-> to take one: memory.py claim sessions/<file>")

def cmd_claim(args):
    """Mark a handoff note as claimed (claim-once: two agents never take the same job)."""
    p = VAULT / args.path
    if not p.exists():
        print(f"[memory] file not found: {p}")
        return
    body = p.read_text(encoding="utf-8", errors="ignore")
    if _is_claimed(body):
        print(f"[memory] already claimed: {args.path}")
        return
    stamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")
    if re.search(r"^(status|durum):", body, re.M | re.I):
        body = re.sub(r"^(status|durum):.*$", f"status: claimed ({stamp})", body, count=1, flags=re.M)
    else:
        line = f"status: claimed ({stamp})"
        m = re.search(r"^## ", body, re.M)
        if m:
            body = body[:m.start()] + line + "\n\n" + body[m.start():]
        else:
            body = body.rstrip() + "\n\n" + line + "\n"
    p.write_text(body, encoding="utf-8")
    print(f"[memory] claimed: {args.path} ({stamp})")

def cmd_mcp(args):
    """stdio MCP server (JSON-RPC 2.0, one message per line)."""
    PROTOCOLS = ["2025-06-18", "2025-03-26", "2024-11-05"]
    def send(obj):
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        method = msg.get("method")
        mid = msg.get("id")
        if not method:
            continue
        if method == "initialize":
            pv = (msg.get("params") or {}).get("protocolVersion")
            proto = pv if pv in PROTOCOLS else PROTOCOLS[-1]
            send({"jsonrpc": "2.0", "id": mid,
                  "result": {"protocolVersion": proto,
                             "capabilities": {"tools": {"listChanged": False}},
                             "serverInfo": {"name": "memory-vault", "version": "2.1"}}})
        elif method.startswith("notifications/"):
            continue
        elif method == "ping":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": MCP_TOOLS}})
        elif method == "tools/call":
            p = msg.get("params") or {}
            try:
                txt = _mcp_call(p.get("name", ""), p.get("arguments") or {})
                send({"jsonrpc": "2.0", "id": mid,
                      "result": {"content": [{"type": "text", "text": txt}]}})
            except Exception as e:
                send({"jsonrpc": "2.0", "id": mid,
                      "result": {"content": [{"type": "text", "text": f"Error: {e}"}],
                                 "isError": True}})
        elif method in ("resources/list", "resources/templates/list"):
            key = "resources" if method.endswith("list") and "templates" not in method else "resourceTemplates"
            send({"jsonrpc": "2.0", "id": mid, "result": {key: []}})
        elif method == "prompts/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"prompts": []}})
        elif mid is not None:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": f"Unknown method: {method}"}})

def cmd_graph(args):
    con = db_conn()
    out = ["# Link graph (generated)", ""]
    for (src,) in con.execute("SELECT DISTINCT src FROM links"):
        outs = list(con.execute("SELECT dst,type FROM links WHERE src=?", (src,)))
        out.append(f"## {src}")
        for d, t in outs:
            out.append(f"- ({t}) → {d}")
        out.append("")
    (VAULT / "relations.md").write_text("\n".join(out), encoding="utf-8")
    print(f"[memory] wrote relations.md ({len(out)} lines). Obsidian also renders these links.")
    con.close()

def cmd_assets(args):
    for sub in ("images", "videos", "audio"):
        d = VAULT / "assets" / sub
        files = sorted(d.iterdir()) if d.exists() else []
        print(f"\n## {sub} ({len(files)}) → {d}")
        for f in files[:30]:
            sz = f.stat().st_size // 1024 if f.is_file() else 0
            print(f"  - {f.name} ({sz}KB)")
        if sub in ("videos", "audio"):
            print("  note: use `reindex --with-audio/--with-video` for real audio/video embeddings.")

def cmd_status(args):
    con = db_conn()
    n_docs = con.execute("SELECT count(*) FROM docs").fetchone()[0]
    n_vec = con.execute("SELECT count(*) FROM vectors").fetchone()[0]
    n_fts = con.execute("SELECT count(*) FROM docs_fts").fetchone()[0] if FTS_OK else 0
    n_links = con.execute("SELECT count(*) FROM links").fetchone()[0]
    con.close()
    print(f"vault: {VAULT} ({len(iter_vault_mds())} md"
          + (f", {_ignored_count()} ignored" if _ignored_count() else "") + ")")
    print(f"db: {DB} ({DB.stat().st_size//1024 if DB.exists() else 0}KB)")
    print(f"index: docs={n_docs} vectors={n_vec} fts={n_fts} links={n_links}")
    print(f"model: {MODEL_ID} (lazy-load, only opened while a command runs)")
    print("lifecycle: the model is loaded per command; the process exits -> VRAM/RAM is freed.")
    print("search modes:")
    print("  default     -> FTS5 first; if keywords are enough the model is NOT opened (~0.2 s),")
    print("                 if keyword confidence is low the semantic stream opens the model.")
    print("  --semantic  -> semantic + keyword RRF fusion (model loaded)")
    print("  --fts       -> keyword only (model never opened)")
    print("git: markdown is the source of truth, .vectors/ is derived (git-ignored).")
    try:
        import torch
        print(f"torch: {torch.__version__}, cuda={torch.cuda.is_available()}")
    except Exception as e:
        print(f"torch not available: {e}")

def main():
    ap = argparse.ArgumentParser(description="Local-first memory vault for AI agents")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reindex")
    r.add_argument("--dim", type=int, default=DIM)
    r.add_argument("--with-images", action="store_true")
    r.add_argument("--with-audio", action="store_true", help="embed assets/audio/* with a real audio encoder (experimental)")
    r.add_argument("--with-video", action="store_true", help="embed assets/videos/* with 1 fps frames (VRAM ~5GB, experimental)")
    r.add_argument("--force", action="store_true", help="ignore the cache and re-embed the whole vault")
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("--top", type=int, default=3); s.add_argument("--full", action="store_true")
    s.add_argument("--semantic", action="store_true", help="force the semantic stream (loads the model)")
    s.add_argument("--fts", dest="fts_only", action="store_true", help="keyword search only (never opens the model)")
    l = sub.add_parser("list"); l.add_argument("--tag", default="")
    a = sub.add_parser("add"); a.add_argument("--title", required=True); a.add_argument("--tags", default="")
    a.add_argument("--file", required=True); a.add_argument("--to", default="daily")
    sh = sub.add_parser("show"); sh.add_argument("path")
    la = sub.add_parser("last", help="newest session handoff notes"); la.add_argument("--n", type=int, default=1); la.add_argument("--chars", type=int, default=3000)
    sub.add_parser("graph"); sub.add_parser("assets"); sub.add_parser("status")
    sub.add_parser("mcp", help="stdio MCP server (tools: memory_search/show/list/add/last/status)")
    op = sub.add_parser("open", help="open (unclaimed) handoff notes"); op.add_argument("--n", type=int, default=5)
    cl = sub.add_parser("claim", help="claim a handoff note"); cl.add_argument("path")
    args = ap.parse_args()
    {"reindex": cmd_reindex, "search": cmd_search, "list": cmd_list,
     "add": cmd_add, "show": cmd_show, "graph": lambda a: cmd_graph(a),
     "assets": lambda a: cmd_assets(a), "status": lambda a: cmd_status(a),
     "last": cmd_last, "mcp": cmd_mcp, "open": cmd_open, "claim": cmd_claim}[args.cmd](args)

if __name__ == "__main__":
    main()
