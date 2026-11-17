# Paired: spellF1 on Games 1-2 with the dead-ball classifier trained on ONE other game (old setup)
# vs on TWO other games (with Sample Game 3). Same atd_like ball and synthetic tags in both arms.
import os, sys, json, warnings
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.environ.setdefault("MTAH_CACHE_DIR", os.path.join(REPO, "solution", "cache"))
sys.path[:0] = [REPO, os.path.join(REPO, "solution"), os.path.join(REPO, "evaluation")]
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
from possession import calibrate, deadball

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
OUTF = os.path.join(REPO, "solution", "continuity", "results", "db_game3_paired.jsonl")
A = B = None


def init():
    global A, B
    P3 = calibrate.prepare(games=(1, 2, 3), seed=0)            # dl: trained on the 2 other games
    S = deadball.training_set(P3[:2])
    one = [deadball.dead_logit(P3[i][1], P3[i][0], deadball.fit(*S[1 - i])) for i in range(2)]
    A = [(G, f, gt, gl, one[i], t) for i, (G, f, gt, gl, dl, t) in enumerate(P3[:2])]
    B = P3[:2]


def run(i):
    p, d = calibrate.sample(np.random.default_rng([77, i]))
    cfg = calibrate.make_cfg(d["switch"], d["min_pos"], d["min_dead"])
    return {"trial": i, "one": calibrate.score(A, p, cfg)[1], "two": calibrate.score(B, p, cfg)[1], "k_dead": p["k_dead"]}


if __name__ == "__main__":
    done = {json.loads(l)["trial"] for l in open(OUTF)} if os.path.exists(OUTF) else set()
    init()                                   # once, in the parent: workers share it via fork
    print("prepared", flush=True)
    with Pool(3) as pool, open(OUTF, "a") as fh:
        for r in pool.imap_unordered(run, [i for i in range(N) if i not in done]):
            fh.write(json.dumps(r) + "\n"); fh.flush()
