"""Metrica Sports open sample data (OTHER games, public, CC-licensed) used to calibrate
and validate the pipeline. Labels are derived ALGORITHMICALLY from the published event
data -- no human labelling by us, and nothing about the challenge game.

Source: https://github.com/metrica-sports/sample-data  (Sample_Game_1, Sample_Game_2)
"""
import os
import urllib.request

import numpy as np
import pandas as pd

from .data import CACHE_DIR, HOME, AWAY

URL = ("https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
       "Sample_Game_{g}/Sample_Game_{g}_{f}.csv")
FILES = ("RawTrackingData_Home_Team", "RawTrackingData_Away_Team", "RawEventsData")
PITCH = (105.0, 68.0)
DIR = os.path.join(CACHE_DIR, "metrica")


# Sample Game 3 ships in the FIFA EPTS format (tracking txt + metadata xml + events json)
EPTS_GAMES = (3,)
EPTS_FILES = ("tracking.txt", "metadata.xml", "events.json")
EPTS_URL = ("https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
            "Sample_Game_{g}/Sample_Game_{g}_{f}")


def download(games=(1, 2, 3)):
    os.makedirs(DIR, exist_ok=True)
    for g in games:
        for f in (EPTS_FILES if g in EPTS_GAMES else FILES):
            if g in EPTS_GAMES:
                path = os.path.join(DIR, f"Sample_Game_{g}_{f}")
                if not os.path.exists(path):
                    urllib.request.urlretrieve(EPTS_URL.format(g=g, f=f), path)
                continue
            path = os.path.join(DIR, f"Sample_Game_{g}_{f}.csv")
            if not os.path.exists(path):
                urllib.request.urlretrieve(URL.format(g=g, f=f), path)


def _tracking(path):
    raw = pd.read_csv(path, skiprows=2)
    cols = list(raw.columns)
    xy = []
    for i, c in enumerate(cols):
        if c.startswith("Player"):
            xy.append(raw.iloc[:, [i, i + 1]].to_numpy(np.float32))
    bi = cols.index("Ball")
    ball = raw.iloc[:, [bi, bi + 1]].to_numpy(np.float32)
    return raw["Period"].to_numpy(), np.stack(xy, 1), ball


def _load_epts(g):
    """Sample Game 3 (EPTS): same dict as the CSV games. Player columns = every player of the
    squad (substitutes NaN while off the pitch), mapped per frame range through the
    DataFormatSpecifications of the metadata; events converted to the CSV columns."""
    import json
    import xml.etree.ElementTree as ET
    p = lambda f: os.path.join(DIR, f"Sample_Game_{g}_{f}")
    root = ET.parse(p("metadata.xml")).getroot()
    home_id = root.find(".//Score").get("idLocalTeam")
    players = [(pl.get("id"), pl.get("teamId")) for pl in root.iter("Player")]
    col = {pid: i for i, (pid, _) in enumerate(players)}
    team = np.array([HOME if t == home_id else AWAY for _, t in players], np.int8)
    chan = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    gp = {e.findtext("Name"): e.findtext("Value") for e in root.iter("ProviderParameter")}
    halves = [(int(gp["first_half_start"]), int(gp["first_half_end"])),
              (int(gp["second_half_start"]), int(gp["second_half_end"]))]
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        order = [col[chan[r]] for r in refs[0::2]]            # x channel of each slot
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))

    cache = p("parsed.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        frames, xy, ball = z["frames"], z["xy"], z["ball"]
    else:
        fr_l, xy_l, ball_l = [], [], []
        with open(p("tracking.txt")) as fh:
            for line in fh:
                fr, pl, b = line.rstrip("\n").split(":")
                fr_l.append(int(fr))
                xy_l.append(np.array(pl.replace(";", ",").split(","), np.float32))
                ball_l.append(np.array(b.split(","), np.float32))
        frames = np.array(fr_l)
        width = max(len(v) for v in xy_l)
        xy = np.full((len(xy_l), width), np.nan, np.float32)
        for i, v in enumerate(xy_l):
            xy[i, :len(v)] = v
        ball = np.stack(ball_l)
        np.savez_compressed(cache, frames=frames, xy=xy, ball=ball)

    first, last = halves[0][0], halves[-1][1]
    n = last - first + 1
    out = np.full((n, len(players), 2), np.nan, np.float32)
    bl = np.full((n, 2), np.nan, np.float32)
    idx = frames - first
    keep = (idx >= 0) & (idx < n)
    bl[idx[keep]] = ball[keep]
    for a, b, order in specs:
        sel = keep & (frames >= a) & (frames <= b)
        slots = xy[sel].reshape(sel.sum(), -1, 2)
        for k, c in enumerate(order):
            out[idx[sel], c] = slots[:, k]
    per = np.zeros(n, np.int64)
    for k, (a, b) in enumerate(halves, 1):
        per[a - first:b - first + 1] = k
    scale = np.array(PITCH, np.float32)

    ev = json.load(open(p("events.json")))["data"]
    rows = []
    for e in ev:
        st = e.get("subtypes")
        st = [] if st is None else (st if isinstance(st, list) else [st])
        rows.append({"Team": "Home" if e["team"]["id"] == home_id else "Away",
                     "Type": e["type"]["name"],
                     "Subtype": "-".join(x["name"] for x in st) if st else np.nan,
                     "Period": e["period"],
                     "Start Frame": e["start"]["frame"] - first + 1,     # 1-based like the CSV
                     "End Frame": (e["end"]["frame"] - first + 1) if e["end"]["frame"] else 0,
                     "Start X": e["start"]["x"], "Start Y": e["start"]["y"],
                     "End X": e["end"]["x"], "End Y": e["end"]["y"]})
    return {"ball": bl * scale, "players": out * scale, "team": team, "n_frames": n,
            "period": per, "events": pd.DataFrame(rows)}


def load_game(g):
    download((g,))
    if g in EPTS_GAMES:
        return _load_epts(g)
    p = lambda f: os.path.join(DIR, f"Sample_Game_{g}_{f}.csv")
    per, ph, ball = _tracking(p(FILES[0]))
    _, pa, _ = _tracking(p(FILES[1]))
    players = np.concatenate([ph, pa], 1)
    team = np.array([HOME] * ph.shape[1] + [AWAY] * pa.shape[1], np.int8)
    scale = np.array(PITCH, np.float32)
    ev = pd.read_csv(p(FILES[2]))
    return {"ball": ball * scale, "players": players * scale, "team": team,
            "n_frames": len(per), "period": per, "events": ev}


def _is_dead_trigger(row):
    t, s = row["Type"], str(row["Subtype"]).upper()
    if t == "BALL OUT" or t == "FAULT RECEIVED":
        return True
    if t == "SHOT" and ("OUT" in s or "GOAL" in s):
        return True
    if t == "BALL LOST" and ("OFFSIDE" in s or "END HALF" in s):
        return True
    return False


CONTROL = {"PASS", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST"}


def labels_from_events(game):
    """Per-frame H/A/D (X before the first kickoff of each period / after its end).

    Rules: control events set the team from their start frame; a dead trigger (ball out,
    foul, goal, offside) sets D from its frame until the next restart (set piece or goal
    kick), which sets the team again.
    """
    n, per = game["n_frames"], game["period"]
    ev = game["events"].sort_values("Start Frame").reset_index(drop=True)
    lab = np.full(n, "X", dtype="<U1")
    for period in np.unique(per):
        e = ev[ev["Period"] == period]
        frames = np.where(per == period)[0]
        f0, f1 = frames[0], frames[-1] + 1
        state, cursor = None, None
        for _, r in e.iterrows():
            fr = int(r["Start Frame"]) - 1
            team = "H" if r["Team"] == "Home" else "A"
            restart = r["Type"] == "SET PIECE" or "GOAL KICK" in str(r["Subtype"]).upper()
            if state is None:
                if r["Type"] in CONTROL:
                    state, cursor = team, fr
                continue
            if state == "D" and not (restart or r["Type"] in CONTROL):
                continue
            if r["Type"] in CONTROL and (team != state):
                lab[cursor:fr] = state
                state, cursor = team, fr
            if _is_dead_trigger(r):
                end = int(r["End Frame"]) - 1 if r["Type"] in ("BALL OUT", "SHOT") and r["End Frame"] > 0 else fr
                end = max(end, cursor)
                lab[cursor:end] = state
                state, cursor = "D", end
        last = int(e["End Frame"].max())
        lab[cursor:min(last, f1)] = state
    return lab


def fill_native_gaps(ball, period):
    """Linear interpolation of the ball inside each period. In the Metrica samples the ball is
    missing almost only at dead ball (~95% of D frames vs ~10% of live frames), while in the
    automatic tracking a stoppage is often still tracked: filling first and then cutting random
    gaps (atd_like_ball) removes the 'missing = dead' shortcut. During a stoppage the filled ball
    drifts slowly from where it went out to where play restarts, roughly a ball being carried
    back."""
    b = ball.copy()
    for p in np.unique(period):
        sl = np.where(period == p)[0]
        seg = b[sl]
        ok = ~np.isnan(seg[:, 0])
        if ok.sum() < 2:
            continue
        t = np.arange(len(sl))
        for k in range(2):
            seg[:, k] = np.interp(t, t[ok], seg[ok, k])   # edges held constant
        b[sl] = seg
    return b


def atd_like_ball(ball, period, rng, target=0.41, median_gap=268):
    """Ball as in the automatic tracking: native gaps filled, then long random gaps cut."""
    return drop_ball_like_atd(fill_native_gaps(ball, period), rng, target, median_gap)


def atd_like_players(players, rng, sigma=0.15, drop=0.10, median_s=10.0):
    """Players as an automatic tracker might give them: Gaussian position jitter (sigma metres,
    per frame) and a fraction `drop` of the on-pitch positions missing in long blocks (lognormal,
    median `median_s` seconds). Used only to AUGMENT the dead-ball training set: the real levels
    in the ATD are unknown, so the classifier is trained on several (deadball.AUG_LEVELS)."""
    P = players.copy()
    if sigma > 0:
        P += rng.normal(0, sigma, P.shape).astype(np.float32)
    if drop > 0:
        on = ~np.isnan(P[:, :, 0])
        miss = np.zeros(on.shape, bool)
        n, target, got = len(P), drop * on.sum(), 0
        while got < target:
            k = int(rng.integers(P.shape[1]))
            ln = int(rng.lognormal(np.log(median_s * 25), 0.8))
            a = int(rng.integers(0, max(n - ln, 1)))
            got += int((on[a:a + ln, k] & ~miss[a:a + ln, k]).sum())
            miss[a:a + ln, k] = True
        P[miss] = np.nan
    return P


def drop_ball_like_atd(ball, rng, target=0.41, median_gap=268):
    """Remove the ball in long blocks so the clean sample data resembles the automatic
    tracking (in BAR-ATH the ball is missing ~41% of graded frames, in few long gaps)."""
    b = ball.copy()
    n = len(b)
    missing = np.isnan(b[:, 0])
    while missing.mean() < target:
        length = int(rng.lognormal(np.log(median_gap), 0.6))
        start = int(rng.integers(0, max(n - length, 1)))
        missing[start:start + length] = True
    b[missing] = np.nan
    return b


# --- synthetic Smart Tagging ---------------------------------------------------------------------
# Metrica has no Smart Tagging, so set-piece tags are synthesised from its events to calibrate the
# set-piece anchors (setpieces.py). They mimic the real tagging: a few-second window around the
# restart, a Type, the taker's team (N/A for throw-ins, as in the real file) and, above all, MANY
# MISSES. Keep rates = restarts tagged in a whole match of the Smart Tagging (DATA.md: throw-in 26,
# corner 12, goal kick 10, free kick 5, kickoffs all) / typical counts per match (~45, ~10, ~15,
# ~25). A few false positives are added at random live moments.

SP_TYPE = {"THROW IN": "Throw-in", "CORNER KICK": "Corner Kick", "FREE KICK": "Free Kick",
           "PENALTY": "Free Kick", "GOAL KICK": "Goal Kick"}
SP_KEEP = {"Throw-in": 0.55, "Corner Kick": 0.9, "Goal Kick": 0.65, "Free Kick": 0.25,
           "Kick Off (After Goal)": 1.0, "Kick Off (Start)": 1.0}


def synthetic_tags(game, rng, fps=25, false_pos_per_half=2):
    ev = game["events"].sort_values("Start Frame")
    n = game["n_frames"]
    goals = [int(r["Start Frame"]) - 1 for _, r in ev.iterrows()
             if r["Type"] == "SHOT" and "GOAL" in str(r["Subtype"]).upper()]
    tags = []

    def window(r):
        a = r - int(rng.uniform(1.0, 4.0) * fps)
        return a, a + int(rng.uniform(3.0, 7.0) * fps)

    for _, e in ev.iterrows():
        st = str(e["Subtype"]).upper()
        r = int(e["Start Frame"]) - 1
        typ = None
        if e["Type"] == "SET PIECE":
            base = st.split("-")[0]
            if base == "KICK OFF":
                typ = ("Kick Off (After Goal)" if any(0 < r - g < 180 * fps for g in goals)
                       else "Kick Off (Start)")
            else:
                typ = SP_TYPE.get(base)
        elif "GOAL KICK" in st and e["Type"] in ("PASS", "BALL LOST"):
            typ = "Goal Kick"
        if typ is None or rng.random() > SP_KEEP[typ]:
            continue
        a, b = window(r)
        team = "N/A" if typ == "Throw-in" else e["Team"]
        tags.append({"code": "SET PIECES", "start_frame": a, "end_frame": b,
                     "tags": {"Type": typ, "Team": team}})
    for g in goals:
        tags.append({"code": "GOALS", "start_frame": g - int(rng.uniform(5, 15) * fps),
                     "end_frame": g + int(rng.uniform(0, 3) * fps), "tags": {"Team": "N/A"}})
    for p in np.unique(game["period"]):
        fr = np.where(game["period"] == p)[0]
        for _ in range(false_pos_per_half):
            r = int(rng.integers(fr[0] + 60 * fps, fr[-1] - 60 * fps))
            a, b = window(r)
            tags.append({"code": "SET PIECES", "start_frame": a, "end_frame": b,
                         "tags": {"Type": str(rng.choice(list(SP_KEEP))), "Team": "N/A"}})
    return tags


def true_restarts(game, fps=25):
    """[(frame, team)] of the restarts in the events (for checking the restart detector only)."""
    out = []
    for _, e in game["events"].iterrows():
        if e["Type"] == "SET PIECE" or ("GOAL KICK" in str(e["Subtype"]).upper()
                                         and e["Type"] in ("PASS", "BALL LOST")):
            out.append((int(e["Start Frame"]) - 1, HOME if e["Team"] == "Home" else AWAY))
    return out
