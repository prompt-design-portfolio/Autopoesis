# G4 campaign — raw per-seed results

* `m1_k7_seed*.json` — mechanic 1, `hold="fail"` (`prep_value` 1.0 → 1.5), five seeds.
* `m1b_k7hold-value_seed*.json` — G4-D1, `hold="value"` (`prep_fail` 0.25 → 0.167), four seeds.
* `diag_econ1p5_seed0_aligned.json` — the economics diagnostic at K=5.
* `noise_seed0.json` — eight replicates per arm over B's RNG seed, the measured noise floor.

Both K=7 sets describe **the same mapping space, the same chance hit rate, and an EV of exactly
zero.** They differ only in which economic term absorbs the EV constraint, and the content effect
changes sign between them — see `docs/G4_WRITEUP.md` §2.8.
