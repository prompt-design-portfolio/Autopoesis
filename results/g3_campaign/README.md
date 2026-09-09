# G3 campaign — raw per-seed results

Thirteen seeds, both alignments, produced by `scripts/g3_seeds.py` against engine version `G4-k`.
One file per `(seed, alignment)`; each is a `G3Result.as_dict()` and reloads with
`G3Result.from_dict`.

These are copied here from `var/`, which is gitignored, because **they are the evidence for every
number in `docs/G3_WRITEUP.md` and were nearly lost to a container restart.** The database files
that made `var/` too large to track (135 MB of SQLite) are not evidence and are not copied.

Read them with `python -m civitas_g campaign --dir results/g3_campaign`.
