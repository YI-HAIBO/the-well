# -*- coding: utf-8 -*-
import io, os, re, numpy as np
base = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38"
rows = []
for s in [0, 1, 2]:
    txt = io.open(os.path.join(base, "e1_seed%d.txt" % s), encoding="utf-8", errors="replace").read()
    m = re.search(r"summary json: (\{.*\})", txt)
    if not m:
        rows.append((s, None))
        continue
    import ast
    d = ast.literal_eval(m.group(1))
    rows.append((s, d))
out = []
for s, d in rows:
    if d:
        out.append("seed %d: vrmse_ours=%.4f vrmse_mean=%.4f rel_l2=%.4f go=%s" %
                   (s, d["vrmse_ours"], d["vrmse_mean"], d["rel_l2"], d["go"]))
    else:
        out.append("seed %d: PARSE FAIL" % s)
vs = [d["vrmse_ours"] for _, d in rows if d]
if len(vs) >= 2:
    out.append("VRMSE ours: mean=%.4f std=%.4f  (n=%d)" % (np.mean(vs), np.std(vs, ddof=1), len(vs)))
io.open(os.path.join(base, "e1_seeds_summary.txt"), "w", encoding="utf-8").write("\n".join(out))
