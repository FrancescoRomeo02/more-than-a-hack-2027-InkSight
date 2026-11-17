# Stress test of the dead-ball classifier against ATD-like PLAYER tracking (Metrica only).
# Perturbation of the test game: Gaussian position jitter (sigma, metres, per frame) and players
# missing in long blocks (fraction drop). Variants: player-speed half-width h (2 = current,
# 12 ~ 1 s) x training on clean games vs on clean + perturbed copies. Leave-one-game-out AUC.
# Stages (each fits in a short command, results cached): prep | feats H | fit H
import os, sys, json, pickle, warnings
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.environ.get("MTAH_EXP_DIR") or os.path.join(REPO, "solution", "cache", "experiments")
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("MTAH_CACHE_DIR", os.path.join(REPO, "solution", "cache"))
sys.path[:0] = [REPO, os.path.join(REPO, "solution"), os.path.join(REPO, "evaluation")]
warnings.filterwarnings("ignore")
import numpy as np
from possession import calibrate, deadball, features
from possession.features import runs

LEVELS = {"clean": (0.0, 0.0), "mid": (0.15, 0.10), "hard": (0.30, 0.20)}
RES = os.path.join(REPO, "solution", "continuity", "results", "robust_stress.jsonl")


def perturb(G, sigma, drop, rng, median_s=10.0):
    P = G["players"].copy()
    if sigma > 0:
        P += rng.normal(0, sigma, P.shape).astype(np.float32)
    if drop > 0:
        on = ~np.isnan(P[:, :, 0])
        miss = np.zeros(on.shape, bool)
        n, target, got = len(P), drop * on.sum(), 0
        while got < target:
            k = int(rng.integers(P.shape[1])); ln = int(rng.lognormal(np.log(median_s * 25), 0.8))
            a = int(rng.integers(0, max(n - ln, 1)))
            got += int((on[a:a + ln, k] & ~miss[a:a + ln, k]).sum()); miss[a:a + ln, k] = True
        P[miss] = np.nan
    return dict(G, players=P)


stage = sys.argv[1]
if stage == "prep":
    P = calibrate.prepare(games=(1, 2, 3), seed=0, dead_model=False)
    pickle.dump([(G, gt) for G, f, gt, *_ in P], open(os.path.join(OUT, "stress_P.pkl"), "wb"))
elif stage == "feats":
    h = int(sys.argv[2]); lv = sys.argv[3]
    features.PLAYER_SPEED_H = h
    P = pickle.load(open(os.path.join(OUT, "stress_P.pkl"), "rb"))
    for gi, (G, gt) in enumerate(P):
        Gp = perturb(G, *LEVELS[lv], np.random.default_rng([gi, h, list(LEVELS).index(lv)]))
        X = deadball.frame_features(features.compute(Gp), Gp)
        keep = gt != "X"
        np.savez_compressed(os.path.join(OUT, f"stress_X_h{h}_{lv}_g{gi}.npz"), X=X[keep], y=(gt[keep] == "D"))
elif stage == "fit":
    from sklearn.metrics import roc_auc_score
    h = int(sys.argv[2])
    L = lambda lv, g: np.load(os.path.join(OUT, f"stress_X_h{h}_{lv}_g{g}.npz"))
    done = set()
    if os.path.exists(RES):
        for l in open(RES):
            r = json.loads(l); done.add((r["h"], r["train"], r["test_game"]))
    for test in range(3):
        tr = [g for g in range(3) if g != test]
        for train in ("clean", "aug"):
            if (h, train, test + 1) in done:
                continue
            lvs = ["clean"] if train == "clean" else list(LEVELS)
            X = np.concatenate([L(lv, g)["X"] for lv in lvs for g in tr])
            y = np.concatenate([L(lv, g)["y"] for lv in lvs for g in tr])
            m = deadball.fit(X, y, subsample=4 * len(lvs))
            for lv in LEVELS:
                d = L(lv, test); z = deadball.predict_logit(m, d["X"]); yt = d["y"]
                pr = z > 0; tp = (pr & yt).sum()
                r = {"h": h, "train": train, "test_game": test + 1, "test_level": lv,
                     "auc": round(roc_auc_score(yt, z), 4), "P": round(tp / max(pr.sum(), 1), 3), "R": round(tp / yt.sum(), 3)}
                print(json.dumps(r), flush=True)
                with open(RES, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
