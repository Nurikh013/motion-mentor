#!/usr/bin/env python3
"""
Motion Mentor - prototype by team Butterfly Effect
===================================================
One click (run.bat, or `python main.py`) does everything:

  1. ReferenceAgent   loads the master reference motion-capture file and splits
                      the 4.7 s clip into 3 tasks.
  2. TraineeSimulator creates 5 randomised trainee recordings (prototype data)
                      whose overall accuracy lands on the configured targets.
  3. EvaluatorAgent   re-reads every trainee CSV from disk and scores it against
                      the master reference: accuracy + coherence per task and
                      per body region, timing offset, cognitive-control index.
  4. CoachAgent       turns the scores into suggestions (Teaching / Learning /
                      Assignment) to raise accuracy.
  5. The results are written to data/*.csv and to dashboard.html, which opens
     in the browser.

Install NumPy with the requirements file before running the pipeline.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import webbrowser
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    raise SystemExit("Install dependencies with: python -m pip install -r requirements.txt") from error

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
REFERENCE_CSV = ROOT / "hip_hop_benchmark_features.csv"
TEMPLATE = ROOT / "dashboard_template.html"
DASHBOARD = ROOT / "dashboard.html"
DATA = ROOT / "data"
PROFILE_DIR = DATA / "profiles"

MODULE_NAME = "Hip-Hop Benchmark"

# A joint that is TOLERANCE away from the reference (in body-normalised units)
# scores 0 % for that frame; a perfect match scores 100 %.
TOLERANCE = 0.12

# Simulated trainees for the prototype: target = required overall accuracy (%).
PROFILES = [
    {"id": "P01", "name": "Aria Conti",    "target": 58.30, "seed": 101},
    {"id": "P02", "name": "Marco Bellini", "target": 65.50, "seed": 202},
    {"id": "P03", "name": "Sofia Ricci",   "target": 90.89, "seed": 303},
    {"id": "P04", "name": "Kenji Mori",    "target": 80.62, "seed": 404},
    {"id": "P05", "name": "Lina Haddad",   "target": 76.55, "seed": 505},
]

# The clip is split into three equal tasks.
TASKS = [
    {"name": "Task 1", "title": "Warm-up groove",   "activity": "Upper-body rhythm and arm coordination"},
    {"name": "Task 2", "title": "Core sequence",    "activity": "Full-body choreography"},
    {"name": "Task 3", "title": "Finish and balance", "activity": "Footwork, balance and recovery"},
]

# Five body regions that get their own accuracy + coherence entry.
REGIONS = {
    "Left hand":  ["left_shoulder", "left_arm", "left_fore_arm", "left_hand"],
    "Right hand": ["right_shoulder", "right_arm", "right_fore_arm", "right_hand"],
    "Left leg":   ["left_up_leg", "left_leg", "left_foot", "left_toe_base"],
    "Right leg":  ["right_up_leg", "right_leg", "right_foot", "right_toe_base"],
    "Head":       ["head"],
}

# Joints further down a limb wander more than joints near the body.
CHAIN_GAIN = {"shoulder": 0.5, "fore_arm": 1.1, "arm": 0.8, "hand": 1.4,
              "up_leg": 0.5, "toe_base": 1.2, "leg": 0.8, "foot": 1.1, "head": 0.8}


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #
def chain_gain(joint: str) -> float:
    for key in sorted(CHAIN_GAIN, key=len, reverse=True):
        if joint.endswith(key):
            return CHAIN_GAIN[key]
    return 1.0


def gaussian_smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth along axis 0 with a Gaussian kernel (reflect padding)."""
    half = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    k /= k.sum()
    return np.apply_along_axis(
        lambda v: np.convolve(np.pad(v, half, mode="reflect"), k, mode="valid"), 0, a)


def smooth_noise(rng: np.random.Generator, shape, sigma: float) -> np.ndarray:
    """Slow, human-like wobble with unit standard deviation."""
    n = gaussian_smooth(rng.standard_normal(shape), sigma)
    return n / n.std(axis=0, keepdims=True)


def speed_of(pos: np.ndarray, dt: float) -> np.ndarray:
    """Joint speed (frames x joints) from positions (frames x joints x 3)."""
    return np.linalg.norm(np.gradient(pos, dt, axis=0), axis=2)


def correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlation; NaN where a signal is flat."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, (a * b).sum(axis=0) / den, np.nan)


# --------------------------------------------------------------------------- #
# Agent 1 - keeps the master reference
# --------------------------------------------------------------------------- #
class ReferenceAgent:
    """Owns the master reference recording of a VR training module."""

    def __init__(self, path: Path):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.header = rows[0]
        table = np.array(rows[1:], dtype=float)
        self.frames = table[:, 0].astype(int)
        self.time = table[:, 1]
        self.joints = [h[:-len("_x_norm")] for h in self.header if h.endswith("_x_norm")]
        self.pos = self.positions(table)
        self.dt = float(self.time[1] - self.time[0])
        self.fps = int(round(1.0 / self.dt))
        self.n_frames = len(table)
        # the hips are the normalised root (always 0), so they are not scored
        self.scored = [j for j, n in enumerate(self.joints) if n != "hips"]
        self.tasks = self._split_tasks()

    def positions(self, table: np.ndarray) -> np.ndarray:
        pos = np.zeros((len(table), len(self.joints), 3))
        for j, name in enumerate(self.joints):
            for k, axis in enumerate("xyz"):
                pos[:, j, k] = table[:, self.header.index(f"{name}_{axis}_norm")]
        return pos

    def _split_tasks(self):
        tasks = []
        for meta, chunk in zip(TASKS, np.array_split(np.arange(self.n_frames), len(TASKS))):
            f0, f1 = int(chunk[0]), int(chunk[-1]) + 1
            tasks.append({**meta, "f0": f0, "f1": f1,
                          "start": float(self.time[f0]),
                          "end": float(self.time[min(f1, self.n_frames - 1)])})
        return tasks

    # -- CSV in the same layout as the reference ---------------------------- #
    def write_recording(self, path: Path, pos: np.ndarray) -> None:
        spd = speed_of(pos, self.dt)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.header)
            for t in range(self.n_frames):
                row = [int(self.frames[t]), f"{self.time[t]:.8f}"]
                for j in range(len(self.joints)):
                    row += [f"{v:.8f}" for v in pos[t, j]] + [f"{spd[t, j]:.8f}"]
                w.writerow(row)

    def read_recording(self, path: Path) -> np.ndarray:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        return self.positions(np.array(rows[1:], dtype=float))


# --------------------------------------------------------------------------- #
# Prototype data - randomised trainees derived from the reference
# --------------------------------------------------------------------------- #
class TraineeSimulator:
    """Builds one trainee: the reference plus timing lag and smooth wobble.

    Every trainee gets random strengths and weaknesses per body region and per
    task. A single `scale` knob then sets how big the deviation is; `fit`
    searches for the scale that gives the requested overall accuracy.
    """
    BASE = 0.03

    def __init__(self, ref: ReferenceAgent, seed: int):
        rng = np.random.default_rng(seed)
        self.ref = ref
        T, J = ref.pos.shape[:2]

        gain = np.zeros(J)
        for names in REGIONS.values():
            region_gain = rng.uniform(0.6, 1.5)
            for n in names:
                gain[ref.joints.index(n)] = region_gain * chain_gain(n)

        task_gain = np.ones(T)
        for task in ref.tasks:
            task_gain[task["f0"]:task["f1"]] = rng.uniform(0.7, 1.4)
        task_gain = gaussian_smooth(task_gain, 4.0)

        self.noise = (smooth_noise(rng, (T, J, 3), 6.0)
                      * gain[None, :, None] * task_gain[:, None, None])

        base_lag = rng.uniform(0.3, 2.0)                       # frames behind
        self.lag = np.zeros((T, J))
        for names in REGIONS.values():
            lag = base_lag + rng.uniform(-0.3, 0.6) + 0.8 * smooth_noise(rng, (T,), 20.0)
            for n in names:
                self.lag[:, ref.joints.index(n)] = lag

    def generate(self, scale: float) -> np.ndarray:
        ref = self.ref
        T = ref.n_frames
        f = np.arange(T, dtype=float)
        pos = np.zeros_like(ref.pos)
        for j in ref.scored:
            src = np.clip(f - scale * self.lag[:, j], 0, T - 1)
            for k in range(3):
                pos[:, j, k] = np.interp(src, f, ref.pos[:, j, k])
        pos += scale * self.BASE * self.noise
        return np.round(pos, 8)                                # what the CSV stores

    def fit(self, target: float, evaluator: "EvaluatorAgent") -> np.ndarray:
        lo, hi = 0.0, 1.0
        while evaluator.overall(self.generate(hi)) > target:
            hi *= 2.0
        best = None
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            pos = self.generate(mid)
            acc = evaluator.overall(pos)
            if best is None or abs(acc - target) < abs(best[0] - target):
                best = (acc, pos)
            if acc > target:
                lo = mid
            else:
                hi = mid
        return best[1]


# --------------------------------------------------------------------------- #
# Agent 2 - compares a recording with the master reference
# --------------------------------------------------------------------------- #
class EvaluatorAgent:
    def __init__(self, ref: ReferenceAgent):
        self.ref = ref
        self.ref_speed = speed_of(ref.pos, ref.dt)

    def scores(self, pos: np.ndarray) -> np.ndarray:
        """Accuracy (0-100) for every frame and joint."""
        err = np.linalg.norm(pos - self.ref.pos, axis=2)
        return 100.0 * np.clip(1.0 - err / TOLERANCE, 0.0, 1.0)

    def overall(self, pos: np.ndarray) -> float:
        return float(self.scores(pos)[:, self.ref.scored].mean())

    def _coherence(self, spd, frames: slice, joints) -> float:
        """How well the speed profile follows the reference (0-100)."""
        c = correlation(self.ref_speed[frames][:, joints], spd[frames][:, joints])
        valid = c[np.isfinite(c)]
        return float(100.0 * np.clip(valid.mean(), 0.0, 1.0)) if valid.size else 0.0

    def _lag_ms(self, spd) -> float:
        a = self.ref_speed[:, self.ref.scored].mean(axis=1)
        b = spd[:, self.ref.scored].mean(axis=1)
        T, best = len(a), (-2.0, 0)
        max_lag = min(10, T - 2)
        lags = sorted(range(-max_lag, max_lag + 1), key=abs)
        for lag in lags:
            x, y = (a[:T - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[:T + lag])
            c = float(correlation(x[:, None], y[:, None])[0])
            if c > best[0]:
                best = (c, lag)
        return best[1] * self.ref.dt * 1000.0                   # + = behind

    def evaluate(self, pos: np.ndarray) -> dict:
        ref = self.ref
        sc = self.scores(pos)
        spd = speed_of(pos, ref.dt)
        everything = slice(0, ref.n_frames)

        tasks = []
        for t in ref.tasks:
            fr = slice(t["f0"], t["f1"])
            tasks.append({**t, "frames": t["f1"] - t["f0"],
                          "accuracy": float(sc[fr][:, ref.scored].mean()),
                          "coherence": self._coherence(spd, fr, ref.scored)})

        regions = []
        for name, joints in REGIONS.items():
            idx = [ref.joints.index(n) for n in joints]
            regions.append({"name": name, "n_joints": len(idx),
                            "accuracy": float(sc[:, idx].mean()),
                            "coherence": self._coherence(spd, everything, idx)})

        accuracy = float(sc[:, ref.scored].mean())
        coherence = self._coherence(spd, everything, ref.scored)
        task_acc = [t["accuracy"] for t in tasks]
        consistency = max(0.0, 100.0 - (max(task_acc) - min(task_acc)))
        cci = 0.5 * accuracy + 0.3 * coherence + 0.2 * consistency
        tier = ("Expert" if accuracy >= 85 else "Proficient" if accuracy >= 75
                else "Developing" if accuracy >= 60 else "Needs support")
        return {"accuracy": accuracy, "coherence": coherence, "consistency": consistency,
                "cci": cci, "tier": tier, "lag_ms": self._lag_ms(spd),
                "tasks": tasks, "regions": regions,
                "frame_acc": sc[:, ref.scored].mean(axis=1),
                "speed": spd[:, ref.scored].mean(axis=1)}


# --------------------------------------------------------------------------- #
# Agent 3 - turns scores into suggestions
# --------------------------------------------------------------------------- #
class CoachAgent:
    """Rule-based coach. Every suggestion is tagged with the hackathon theme:
    Teaching (instructor), Learning (trainee) or Assignment (next task)."""

    def __init__(self, ref: ReferenceAgent):
        self.n_scored = len(ref.scored)
        self.n_frames = ref.n_frames

    def advise(self, r: dict) -> list[dict]:
        out = []
        regs, tasks = r["regions"], r["tasks"]
        weak = min(regs, key=lambda x: x["accuracy"])
        strong = max(regs, key=lambda x: x["accuracy"])
        worst = min(tasks, key=lambda x: x["accuracy"])
        loose = min(regs, key=lambda x: x["coherence"])

        median = float(np.median([x["accuracy"] for x in regs]))
        gain = max(0.0, (median - weak["accuracy"]) * weak["n_joints"] / self.n_scored)
        out.append({"tag": "Learning", "title": f"Mirror drill: {weak['name'].lower()}",
                    "text": (f"{weak['name']} is the weakest region at {weak['accuracy']:.1f}% "
                             f"(strongest: {strong['name'].lower()}, {strong['accuracy']:.1f}%). "
                             f"Replay {worst['name']} at half speed with the reference ghost "
                             f"overlaid and trace the {weak['name'].lower()} path."),
                    "gain": gain})

        others = float(np.mean([t["accuracy"] for t in tasks if t is not worst]))
        gain = max(0.0, (others - worst["accuracy"]) * worst["frames"] / self.n_frames)
        out.append({"tag": "Assignment", "title": f"Repeat {worst['name']}: {worst['title'].lower()}",
                    "text": (f"{worst['name']} ({worst['start']:.2f}-{worst['end']:.2f} s) scores "
                             f"{worst['accuracy']:.1f}%, the lowest of the three. Assign it as "
                             f"3 focused repetitions with a pass mark of {min(95, math.ceil(others))}%."),
                    "gain": gain})

        lag = r["lag_ms"]
        if abs(lag) >= 30:
            where = "behind" if lag > 0 else "ahead of"
            text = (f"The trainee moves about {abs(lag):.0f} ms {where} the reference and "
                    f"{loose['name'].lower()} coherence is {loose['coherence']:.1f}%. "
                    f"Count the beat aloud or use a metronome before the next attempt.")
        else:
            text = (f"Timing is on the beat; the gap is in the shape of the movement. "
                    f"{loose['name']} coherence is {loose['coherence']:.1f}%: demonstrate that "
                    f"movement at full range and let the trainee copy it.")
        out.append({"tag": "Teaching", "title": "Instructor cue: timing and rhythm",
                    "text": text, "gain": None})

        by = {x["name"]: x["accuracy"] for x in regs}
        for a, b, what in (("Left hand", "Right hand", "hands"), ("Left leg", "Right leg", "legs")):
            if abs(by[a] - by[b]) >= 5:
                lo, hi = (a, b) if by[a] < by[b] else (b, a)
                out.append({"tag": "Learning", "title": f"Balance the {what}",
                            "text": (f"{lo} trails {hi.lower()} by {abs(by[a] - by[b]):.1f} points. "
                                     f"Add single-side repetitions for the {lo.lower()} before "
                                     f"the full sequence."),
                            "gain": None})

        if r["accuracy"] >= 85:
            out.append({"tag": "Teaching", "title": "Use as a peer demonstration",
                        "text": ("This recording is close to the master reference. Show it to "
                                 "the cohort as a second example and unlock the next module."),
                        "gain": None})
        return out

    def cohort(self, results: list[dict]) -> list[dict]:
        region_avg = {name: float(np.mean([next(x for x in r["regions"] if x["name"] == name)["accuracy"]
                                           for r in results])) for name in REGIONS}
        task_avg = [float(np.mean([r["tasks"][i]["accuracy"] for r in results])) for i in range(len(TASKS))]
        weak_region = min(region_avg, key=region_avg.get)
        hard = int(np.argmin(task_avg))
        low = min(results, key=lambda r: r["accuracy"])
        top = max(results, key=lambda r: r["accuracy"])
        return [
            {"tag": "Teaching", "title": f"Re-teach the {weak_region.lower()} movement",
             "text": (f"Across the cohort the {weak_region.lower()} is the weakest region "
                      f"({region_avg[weak_region]:.1f}% on average). Open the next session with "
                      f"a slow demonstration of it."), "gain": None},
            {"tag": "Assignment", "title": f"Set {TASKS[hard]['name']} as homework",
             "text": (f"{TASKS[hard]['name']} ({TASKS[hard]['title'].lower()}) is the hardest part "
                      f"of the clip, {task_avg[hard]:.1f}% on average. Assign it to everyone "
                      f"below 75%."), "gain": None},
            {"tag": "Learning", "title": "Pair up for peer learning",
             "text": (f"Pair {low['name']} ({low['accuracy']:.2f}%) with {top['name']} "
                      f"({top['accuracy']:.2f}%) for a side-by-side replay session."), "gain": None},
        ]


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def flat(pos: np.ndarray) -> list:
    return [round(float(v), 4) for v in pos.reshape(-1)]


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def main(open_browser: bool = True) -> None:
    print("=" * 62)
    print(" Motion Mentor  -  team Butterfly Effect  -  prototype run")
    print("=" * 62)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    reference = ReferenceAgent(REFERENCE_CSV)
    evaluator = EvaluatorAgent(reference)
    coach = CoachAgent(reference)
    print(f"[ReferenceAgent] {REFERENCE_CSV.name}: {reference.n_frames} frames, "
          f"{reference.time[-1]:.2f} s, {len(reference.joints)} joints, {len(reference.tasks)} tasks")

    results = []
    for p in PROFILES:
        path = PROFILE_DIR / f"{p['id']}_{slug(p['name'])}.csv"
        pos = TraineeSimulator(reference, p["seed"]).fit(p["target"], evaluator)
        reference.write_recording(path, pos)
        print(f"[TraineeSimulator] wrote {path.relative_to(ROOT)}")

        recorded = reference.read_recording(path)             # score what is on disk
        r = evaluator.evaluate(recorded)
        r.update(id=p["id"], name=p["name"], target=p["target"], file=path.name, pos=recorded)
        r["suggestions"] = coach.advise(r)
        results.append(r)

    print("\n[EvaluatorAgent] results")
    print(f"  {'id':<4} {'name':<15} {'accuracy':>9} {'target':>8} {'coherence':>10} {'control':>8}  tier")
    for r in results:
        print(f"  {r['id']:<4} {r['name']:<15} {r['accuracy']:>8.2f}% {r['target']:>7.2f}% "
              f"{r['coherence']:>9.1f}% {r['cci']:>8.1f}  {r['tier']}")
        assert round(r["accuracy"], 2) == r["target"], "accuracy target missed"
    cohort_avg = float(np.mean([r["accuracy"] for r in results]))
    print(f"  cohort average accuracy: {cohort_avg:.2f}%")

    # ---- CSV outputs ------------------------------------------------------ #
    def write(name, header, rows):
        with open(DATA / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    write("summary.csv",
          ["id", "name", "recording", "overall_accuracy", "coherence", "consistency",
           "cognitive_control_index", "tier", "timing_offset_ms", "weakest_region", "weakest_task"],
          [[r["id"], r["name"], r["file"], f"{r['accuracy']:.2f}", f"{r['coherence']:.2f}",
            f"{r['consistency']:.2f}", f"{r['cci']:.2f}", r["tier"], f"{r['lag_ms']:.0f}",
            min(r["regions"], key=lambda x: x["accuracy"])["name"],
            min(r["tasks"], key=lambda x: x["accuracy"])["name"]] for r in results])
    write("task_scores.csv",
          ["id", "name", "task", "activity", "start_s", "end_s", "frames", "accuracy", "coherence"],
          [[r["id"], r["name"], t["name"], t["title"], f"{t['start']:.3f}", f"{t['end']:.3f}",
            t["frames"], f"{t['accuracy']:.2f}", f"{t['coherence']:.2f}"]
           for r in results for t in r["tasks"]])
    write("region_scores.csv",
          ["id", "name", "region", "accuracy", "coherence"],
          [[r["id"], r["name"], g["name"], f"{g['accuracy']:.2f}", f"{g['coherence']:.2f}"]
           for r in results for g in r["regions"]])
    write("suggestions.csv",
          ["id", "name", "theme", "suggestion", "detail", "projected_gain_points"],
          [[r["id"], r["name"], s["tag"], s["title"], s["text"],
            "" if s["gain"] is None else f"{s['gain']:.2f}"]
           for r in results for s in r["suggestions"]])
    print(f"\n[CoachAgent] {sum(len(r['suggestions']) for r in results)} suggestions written")

    # ---- dashboard -------------------------------------------------------- #
    payload = {
        "meta": {"module": MODULE_NAME, "fps": reference.fps, "frames": reference.n_frames,
                 "duration": float(reference.time[-1]), "joints": reference.joints,
                 "tol": TOLERANCE,
                 "tasks": [{k: t[k] for k in ("name", "title", "activity", "f0", "f1", "start", "end")}
                           for t in reference.tasks]},
        "ref": flat(reference.pos),
        "ref_speed": [round(float(v), 4) for v in evaluator.ref_speed[:, reference.scored].mean(axis=1)],
        "cohort": {"avg_accuracy": round(cohort_avg, 2),
                   "avg_coherence": round(float(np.mean([r["coherence"] for r in results])), 1),
                   "suggestions": coach.cohort(results)},
        "profiles": [{
            "id": r["id"], "name": r["name"], "file": r["file"],
            "accuracy": round(r["accuracy"], 2), "coherence": round(r["coherence"], 1),
            "consistency": round(r["consistency"], 1), "cci": round(r["cci"], 1),
            "tier": r["tier"], "lag_ms": round(r["lag_ms"]),
            "tasks": [{"accuracy": round(t["accuracy"], 1), "coherence": round(t["coherence"], 1)}
                      for t in r["tasks"]],
            "regions": [{"name": g["name"], "accuracy": round(g["accuracy"], 1),
                         "coherence": round(g["coherence"], 1)} for g in r["regions"]],
            "frame_acc": [round(float(v), 1) for v in r["frame_acc"]],
            "speed": [round(float(v), 4) for v in r["speed"]],
            "pos": flat(r["pos"]),
            "suggestions": [{**s, "gain": None if s["gain"] is None else round(s["gain"], 1)}
                            for s in r["suggestions"]],
        } for r in results],
    }
    data_json = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", data_json.replace("<", "\\u003c"))
    DASHBOARD.write_text(html, encoding="utf-8")
    print(f"[Dashboard] wrote {DASHBOARD.name}")

    if open_browser:
        webbrowser.open(DASHBOARD.as_uri())
        print("Opening the dashboard in your browser ...")
    print("Done.")


if __name__ == "__main__":
    main(open_browser="--no-open" not in sys.argv)
