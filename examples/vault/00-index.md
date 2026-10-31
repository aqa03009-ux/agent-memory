# Memory — index

> Entry point for the vault. Every new session reads this file first, then
> searches with `memory.py search "<task keywords>"`.

## Rules

1. **Summary first, detail later** — `search` returns summaries; `show` returns
   the full note.
2. **Markdown is the source of truth** — the SQLite index is disposable.
3. **Raw corpora are not memory** — transcripts and logs go into `.memoryignore`.

## Map

| Folder | Purpose |
|---|---|
| `setups/` | how working things are configured |
| `decisions/` | why something was chosen |
| `projects/` | one note per project |
| `sessions/` | handoff notes between sessions |

## Related

- [[profile]]
- [[setups/example-setup]]
- [[decisions/example-decision]]
