# Does a third Metrica game (Sample Game 3, EPTS) help the dead-ball classifier generalise?
# Out-of-game AUC / P / R of D for each test game, trained on 1 other game vs on 2 other games.
import os, sys, time, json, warnings
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.environ.get("MTAH_EXP_DIR") or os.path.join(REPO, "solution", "cache", "experiments")
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("MTAH_CACHE_DIR", os.path.join(REPO, "solution", "cache"))
sys.path[:0] = [REPO, os.path.join(REPO, "solution"), os.path.join(REPO, "evaluation")]
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score
from possession import calibrate, deadball

t = time.time()
P = calibrate.prepare(games=(1, 2, 3), seed=0, dead_model=False)
S = deadball.training_set(P)
print("prepared %.0fs" % (time.time() - t), [s[0].shape for s in S], flush=True)
res = []
for i in range(3):
    Xt, yt = S[i]
    others = [j for j in range(3) if j != i]
    for train in [[j] for j in others] + [others]:
        X = np.concatenate([S[j][0] for j in train]); y = np.concatenate([S[j][1] for j in train])
        z = deadball.predict_logit(deadball.fit(X, y), Xt)
        pr = z > 0; tp = (pr & (yt == 1)).sum()
        r = {"test": i + 1, "train": [j + 1 for j in train], "auc": roc_auc_score(yt, z),
             "P": tp / max(pr.sum(), 1), "R": tp / yt.sum(), "prior": float(yt.mean())}
        res.append(r); print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
json.dump(res, open(os.path.join(REPO, "solution", "continuity", "results", "db_game3.json"), "w"), indent=1)
