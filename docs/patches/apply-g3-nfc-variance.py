"""Engine version G3-nfc-variance: log the sum of squares alongside the sum.

WHY. G3's coherence check asks whether two arms' `nfc_mean` agree. "Agree" needs a scale, and the
only scale currently available is the effect itself -- which is the defect in
`docs/G3_WRITEUP.md` §5.6: the bar shrinks with the effect, so a seed with no effect cannot pass.
The principled scale is the sampling noise of the statistic, and that needs the second moment. The
engine logs `nfc_sum` and `n_nfc` and not `nfc_sumsq`, so the standard error of `nfc_mean` is not
recoverable from any run ever made. A1.8 forbids Civitas adding measurement of its own; a versioned
engine change is the sanctioned route.

TRAJECTORY-NEUTRAL BY CONSTRUCTION. Two accumulators and two log fields. Nothing is read back, no
branch depends on them, and no draw is added, removed or resized -- so the RNG stream is untouched
and the equivalence check is bit-identity against G4-k, not an argument.
"""
import pathlib

p = pathlib.Path("sim_v3_13.py")
s = p.read_text()

# ---- 1. the counters exist ------------------------------------------------------------------
old = "             nfc_n=0, nfc_sum=0, nfc_cens=0,"
new = "             nfc_n=0, nfc_sum=0, nfc_sumsq=0, nfc_cens=0,"
assert s.count(old) == 1, "counter init"
s = s.replace(old, new)

old = '''                         "nfc_n", "nfc_sum", "nfc_cens")}'''
new = '''                         "nfc_n", "nfc_sum", "nfc_sumsq", "nfc_cens")}'''
assert s.count(old) == 1, "counter copy list"
s = s.replace(old, new)

# ---- 2. accumulate, on the same lines that accumulate the sum -------------------------------
old = '''                        W["nfc_n"] += 1; W["nfc_sum"] += a.nfc_at'''
new = '''                        W["nfc_n"] += 1; W["nfc_sum"] += a.nfc_at
                        W["nfc_sumsq"] += a.nfc_at * a.nfc_at'''
assert s.count(old) == 1, "W accumulate"
s = s.replace(old, new)

old = '''                            WF["nfc_n"] += 1; WF["nfc_sum"] += a.nfc_at'''
new = '''                            WF["nfc_n"] += 1; WF["nfc_sum"] += a.nfc_at
                            WF["nfc_sumsq"] += a.nfc_at * a.nfc_at'''
assert s.count(old) == 1, "WF accumulate"
s = s.replace(old, new)

# ---- 3. into the log ------------------------------------------------------------------------
old = '''                n_nfc=W["nfc_n"], nfc_sum=W["nfc_sum"], n_nfc_cens=W["nfc_cens"],'''
new = '''                n_nfc=W["nfc_n"], nfc_sum=W["nfc_sum"], nfc_sumsq=W["nfc_sumsq"],
                n_nfc_cens=W["nfc_cens"],'''
assert s.count(old) == 1, "log W"
s = s.replace(old, new)

old = '''                f_n_nfc=WF["nfc_n"], f_nfc_sum=WF["nfc_sum"], f_n_nfc_cens=WF["nfc_cens"],'''
new = '''                f_n_nfc=WF["nfc_n"], f_nfc_sum=WF["nfc_sum"], f_nfc_sumsq=WF["nfc_sumsq"],
                f_n_nfc_cens=WF["nfc_cens"],'''
assert s.count(old) == 1, "log WF"
s = s.replace(old, new)

p.write_text(s)
print("G3-nfc-variance applied")
