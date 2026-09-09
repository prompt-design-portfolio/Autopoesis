"""Engine version G5-policy: the external-policy hook (G5-D7).

Applied as a script rather than by hand so the change is exactly the change the spec describes,
and so it can be re-applied on a clean tree if anything goes wrong.
"""
import pathlib, sys

p = pathlib.Path("sim_v3_13.py")
s = p.read_text()

# ---- 1. Agent.act takes the policy ---------------------------------------------------------
old = "    def act(self, obs, cfg, rng, chain_on=True):"
new = "    def act(self, obs, cfg, rng, chain_on=True, external_policy=None):"
assert s.count(old) == 1, "act signature"
s = s.replace(old, new)

# ---- 2. the override, AFTER the logits are drawn -------------------------------------------
old = """        a = int(np.argmax(logits))
        if plastic:"""
new = """        a = int(np.argmax(logits))
        if external_policy is not None:
            # G5-D7.  The logits above are computed first, action-noise draw included, so the
            # RNG stream is identical whether or not a policy is in force -- G3-mapping's
            # discipline: draw, then overwrite.  `None` means this step is the network's, which
            # is how a policy is applied to a sampled subset without a second mechanism.  The
            # trace below then follows the action ACTUALLY taken, which is why the override
            # happens here rather than at the call site.
            proposed = external_policy(self.lineage, obs, chain_on, logits)
            if proposed is not None:
                a = int(proposed)
                if not 0 <= a < cfg.n_actions or (not chain_on and a >= PREP0):
                    raise ValueError(
                        f"an external policy returned action {a}; legal actions are "
                        f"0..{(cfg.n_actions if chain_on else PREP0) - 1} here.  Refused rather "
                        f"than clipped: a clipped action would be recorded as a choice the "
                        f"policy did not make.")
        if plastic:"""
assert s.count(old) == 1, "act override site"
s = s.replace(old, new)

# ---- 3. the call site ----------------------------------------------------------------------
old = "            action = a.act(obs, cfg, rng, world.chain_on)"
new = "            action = a.act(obs, cfg, rng, world.chain_on, external_policy)"
assert s.count(old) == 1, "act call site"
s = s.replace(old, new)

# ---- 4. run() takes it ---------------------------------------------------------------------
old = """def run(cfg, verbose=True, init_genomes=None, phases=None, init_store=None,
        init_mapping=None):"""
new = """def run(cfg, verbose=True, init_genomes=None, phases=None, init_store=None,
        init_mapping=None, external_policy=None):"""
assert s.count(old) == 1, "run signature"
s = s.replace(old, new)

p.write_text(s)
print("G5-policy applied")
