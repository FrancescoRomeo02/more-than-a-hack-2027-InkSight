# Paired spellF1 (Metrica games 1-3): OLD dead-ball setup (player speed h=2, training on clean games)
# vs NEW (h=12 + training augmented with ATD-like players), each scored on the clean games and on
# games with ATD-like players at the "hard" level (jitter 0.3 m, 20% of player positions missing).
# Stages: prep ARM (old_clean|old_hard|new_clean|new_hard) ; score N   (resumable, short commands)
import os, sys, json, pickle, warnings
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.environ.get("MTAH_EXP_DIR") or os.path.join(REPO, "solution", "cache", "experiments")
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("MTAH_CACHE_DIR", os.path.join(REPO, "solution", "cache"))
sys.path[:0] = [REPO, os.path.join(REPO, "solution"), os.path.join(REPO, "evaluation")]
warnings.filterwarnings("ignore")
import numpy as np
from possession import calibrate, deadball, features, metrica

ARMS = ("old_clean", "old_hard", "new_clean", "new_hard")
RES = os.path.join(REPO, "solution", "continuity", "results", "robust_paired.jsonl")
stage = sys.argv[1]

if stage == "prep":
    arm = sys.argv[2]; setup, test = arm.split("_")
    if setup == "old":
        features.PLAYER_SPEED_H = 2; deadball.AUG_LEVELS = (); deadball.SUBSAMPLE = 4
    P = calibrate.prepare(seed=0, dead_model=(test == "clean"))
    if test == "hard":                       # score games with perturbed players, models from the others
        sets = deadball.training_set(P)
        Q = []
        for i, (G, f, gt, gl, dl, t) in enumerate(P):
            X = np.concatenate([s[0] for j, s in enumerate(sets) if j != i])
            y = np.concatenate([s[1] for j, s in enumerate(sets) if j != i])
            Gp = dict(G, players=metrica.atd_like_players(G["players"], np.random.default_rng([7, i]), 0.30, 0.20))
            fp = features.compute(Gp)
            Q.append((Gp, fp, gt, gl, deadball.dead_logit(fp, Gp, deadball.fit(X, y, subsample=deadball.SUBSAMPLE)), t))
        P = Q
    slim = [({"ball": G["ball"], "n_frames": G["n_frames"]}, f, gt, gl, dl, t) for G, f, gt, gl, dl, t in P]
    pickle.dump({"P": slim, "h": features.PLAYER_SPEED_H}, open(os.path.join(OUT, f"rp_{arm}.pkl"), "wb"))

elif stage == "score":
    from multiprocessing import Pool
    N = int(sys.argv[2])
    D = {a: pickle.load(open(os.path.join(OUT, f"rp_{a}.pkl"), "rb")) for a in ARMS}
    done = {json.loads(l)["trial"] for l in open(RES)} if os.path.exists(RES) else set()

    def run(i):
        p, d = calibrate.sample(np.random.default_rng([99, i]))
        cfg = calibrate.make_cfg(d["switch"], d["min_pos"], d["min_dead"])
        r = {"trial": i}
        for a in ARMS:
            features.PLAYER_SPEED_H = D[a]["h"]
            r[a] = calibrate.score(D[a]["P"], p, cfg)[1]
        return r
    with Pool(3) as pool, open(RES, "a") as fh:
        for r in pool.imap_unordered(run, [i for i in range(N) if i not in done]):
            fh.write(json.dumps(r) + "\n"); fh.flush()
