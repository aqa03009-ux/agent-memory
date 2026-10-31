# Finish the index migration

status: open
recorded: 2026-01-01 10:00
session: example-session-id

## What the user asked

Move the old index to the hybrid FTS5 + vector layout.

## Where we stopped

- [x] FTS table created
- [x] RRF fusion working
- [ ] Verify the confidence gate on a real vault
- [ ] Delete the legacy `.vectors/old.db`

## Next step

Run `./memory.py reindex --force` against `examples/vault` and check
`./memory.py search "confidence gate" --semantic`.
