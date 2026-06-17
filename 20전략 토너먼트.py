import csv
import os
import random
import time
from dataclasses import dataclass

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from numba import njit

# =========================
# 설정
# =========================
N_PER_STRAT = 50
STRATEGIES = [
    "TFT",
    "TIDEMAN_CHIERUZZI",
    "NYDEGGER",
    "GROFMAN",
    "SHUBIK",
    "GRUDGER",
    "DAVIS",
    "GRAASKAMP",
    "DOWNING",
    "FELD",
    "JOSS",
    "TULLOCK",
    "RANDOM",
    "ALLC",
    "ALLD",
    "ZD_EXTORTIONER",
    "TESTER",
    "PAVLOV",
    "TF2T",
    "PROBER",
]
N_INIT = N_PER_STRAT * len(STRATEGIES)
STRAT_FILENAME = "20strat_numba_v1_no_last2D"
R_INTERACT = 0.05
ROUNDS_PER_PAIR = 12
GENS = 300
ALPHA = 0.005
BETA = 0.001
P_EXP = 2.0
Q_EXP = 2.0
MOVE_SUBSTEPS = 3
MAX_STEP = 0.010
DIST_EPS = 0.02
BASE_NOISE_SIGMA = 0.0008
TOP_K = 20
BOTTOM_K = 20

# 10회 반복
SEED_LIST = list(range(1, 11))

COLOR_LIST = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
    "#e6550d",
    "#31a354",
    "#756bb1",
    "#636363",
]
COLOR_MAP_STRAT = {s: COLOR_LIST[i] for i, s in enumerate(STRATEGIES)}

# 출력 폴더: 스크립트와 같은 위치의 output/ 아래에 생성 (포터블).
#   원래 작성 환경에서는 ~/Desktop 에 저장했으나, 저장소 배포용으로 경로를 상대화함.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output", f"{STRAT_FILENAME}_10runs")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# Agent
# =========================
@dataclass
class Agent:
    x: float
    strategy: str
    score: float = 0.0


@njit(fastmath=True)
def reflect_01_numba(v):
    if v < 0.0:
        return -v
    if v > 1.0:
        return 2.0 - v
    return v


@njit(fastmath=True)
def payoff(a_i, a_j):
    # 1 = C, 0 = D
    if a_i == 1 and a_j == 1:
        return 3.0, 3.0
    if a_i == 1 and a_j == 0:
        return 0.0, 5.0
    if a_i == 0 and a_j == 1:
        return 5.0, 0.0
    return 1.0, 1.0


@njit(fastmath=True)
def nydegger_defects(A):
    vals = np.array([1, 6, 7, 17, 22, 23, 26, 29, 30, 31, 33, 38, 39, 45, 49, 54, 55, 58, 61])
    for v in vals:
        if A == v:
            return True
    return False


@njit(fastmath=True)
def count_ones(arr, n):
    c = 0
    for i in range(n):
        c += arr[i]
    return c


@njit(fastmath=True)
def count_zeros(arr, n):
    return n - count_ones(arr, n)


@njit(fastmath=True)
def coop_rate_recent(arr, start, end):
    if end <= start:
        return 0.5
    s = 0
    for i in range(start, end):
        s += arr[i]
    return s / (end - start)


@njit(fastmath=True)
def looks_random_simple(hist, n):
    if n < 8:
        return False
    coop = count_ones(hist, n)
    rate = coop / n
    if rate < 0.35 or rate > 0.65:
        return False
    switches = 0
    for i in range(1, n):
        if hist[i] != hist[i - 1]:
            switches += 1
    sw_rate = switches / (n - 1)
    return 0.35 <= sw_rate <= 0.75


@njit(fastmath=True)
def choose_action(
    strategy, my_hist, opp_hist, t, rounds, score_me, score_opp, mode0, mode1, mode2, mode3
):
    # 1 = C, 0 = D

    # TFT
    if strategy == 0:
        if t == 0:
            return 1
        return opp_hist[t - 1]

    # Tideman & Chieruzzi
    # 수정점: "마지막 2턴 자동 배신" 제거
    if strategy == 1:
        # mode0: punish remaining
        # mode1: punishment level
        # mode2: opponent defection run length
        # mode3: last fresh start turn
        if mode0 == -2 or mode0 == -1:
            return 1
        if t == 0:
            return 1
        if mode0 > 0:
            return 0
        # fresh start condition
        if score_me + 10.0 <= score_opp and t - mode3 >= 20 and rounds - t >= 10 and mode2 == 0:
            opp_def = count_zeros(opp_hist, t)
            expv = 0.5 * t
            sd = np.sqrt(0.25 * t + 1e-12)
            if np.abs(opp_def - expv) >= 3.0 * sd:
                return 1
        return opp_hist[t - 1]

    # Nydegger
    if strategy == 2:
        if t == 0:
            return 1
        if t == 1:
            return opp_hist[0]
        if t == 2:
            if my_hist[0] == 1 and opp_hist[0] == 0 and my_hist[1] == 0 and opp_hist[1] == 1:
                return 0
            return opp_hist[1]
        a1 = 2 * (1 - opp_hist[t - 3]) + (1 - my_hist[t - 3])
        a2 = 2 * (1 - opp_hist[t - 2]) + (1 - my_hist[t - 2])
        a3 = 2 * (1 - opp_hist[t - 1]) + (1 - my_hist[t - 1])
        A = 16 * a1 + 4 * a2 + a3
        return 0 if nydegger_defects(A) else 1

    # Grofman
    if strategy == 3:
        if t == 0:
            return 1
        if my_hist[t - 1] == opp_hist[t - 1]:
            return 1
        return 1 if np.random.random() < (2.0 / 7.0) else 0

    # Shubik
    if strategy == 4:
        if t == 0:
            return 1
        if mode0 > 0:
            return 0
        return 1

    # Grudger
    if strategy == 5:
        if t == 0:
            return 1
        if mode0 == 1:
            return 0
        return 1

    # Davis
    if strategy == 6:
        if t < 10:
            return 1
        if mode0 == 1:
            return 0
        return 1

    # Graaskamp
    if strategy == 7:
        if mode0 == 1:
            return 0
        if t < 50:
            if t == 0:
                return 1
            return opp_hist[t - 1]
        if t == 50:
            return 0
        if t < 56:
            return opp_hist[t - 1]
        if t == 56:
            return 0 if t == mode1 else 1
        if t == mode1:
            return 0
        return 1 if t == 0 else opp_hist[t - 1]

    # Downing
    if strategy == 8:
        if t < 2:
            return 1
        cc_n = 0
        cc_c = 0
        dc_n = 0
        dc_c = 0
        for k in range(1, t):
            if my_hist[k - 1] == 1:
                cc_n += 1
                cc_c += opp_hist[k]
            else:
                dc_n += 1
                dc_c += opp_hist[k]
        alpha = (cc_c + 1.0) / (cc_n + 2.0)
        beta = (dc_c + 1.0) / (dc_n + 2.0)
        val_c = alpha * 3.0 + (1.0 - alpha) * 0.0
        val_d = beta * 5.0 + (1.0 - beta) * 1.0
        return 1 if val_c >= val_d else 0

    # Feld
    if strategy == 9:
        if t == 0:
            return 1
        if opp_hist[t - 1] == 0:
            return 0
        p = 1.0 - 0.5 * min(t, 200) / 200.0
        return 1 if np.random.random() < p else 0

    # Joss
    if strategy == 10:
        if t == 0:
            return 1
        if opp_hist[t - 1] == 0:
            return 0
        return 1 if np.random.random() < 0.9 else 0

    # Tullock
    if strategy == 11:
        if t < 11:
            return 1
        rate = coop_rate_recent(opp_hist, t - 10, t)
        p = max(0.0, min(1.0, rate - 0.1))
        return 1 if np.random.random() < p else 0

    # Random
    if strategy == 12:
        return 1 if np.random.random() < 0.5 else 0

    # ALLC
    if strategy == 13:
        return 1

    # ALLD
    if strategy == 14:
        return 0

    # ZD Extortioner
    if strategy == 15:
        if t == 0:
            return 1
        prev_me = my_hist[t - 1]
        prev_opp = opp_hist[t - 1]
        if prev_me == 1 and prev_opp == 1:
            p = 0.90
        elif prev_me == 1 and prev_opp == 0:
            p = 0.05
        elif prev_me == 0 and prev_opp == 1:
            p = 0.85
        else:
            p = 0.00
        return 1 if np.random.random() < p else 0

    # Tester
    if strategy == 16:
        if t == 0:
            return 0
        if t == 1:
            return 1
        if mode0 == 2:
            return opp_hist[t - 1]
        if mode0 == 1:
            return 0
        return opp_hist[t - 1]

    # Pavlov / WSLS
    if strategy == 17:
        if t == 0:
            return 1
        if my_hist[t - 1] == opp_hist[t - 1]:
            return my_hist[t - 1]
        return 1 - my_hist[t - 1]

    # Tit for two tats
    if strategy == 18:
        if t < 2:
            return 1
        if opp_hist[t - 1] == 0 and opp_hist[t - 2] == 0:
            return 0
        return 1

    # Prober
    if strategy == 19:
        if t == 0:
            return 0
        if t == 1 or t == 2:
            return 1
        if mode0 == 1:
            return 0
        if mode0 == 2:
            return opp_hist[t - 1]
        return opp_hist[t - 1]

    return 1


@njit(fastmath=True)
def update_mode_after_round(
    strategy, my_hist, opp_hist, t, score_me, score_opp, mode0, mode1, mode2, mode3
):
    if strategy == 1:
        # Tideman & Chieruzzi
        if mode0 == -2:
            mode0 = -1
        elif mode0 == -1:
            mode0 = 0
            mode1 = 0
            mode2 = 0
            mode3 = t
            for k in range(t + 1):
                my_hist[k] = 1 if k == t else my_hist[k]
        else:
            opp_def = 1 - opp_hist[t]
            if opp_def == 1:
                if mode2 == 0:
                    mode1 += 1
                    mode0 = mode1
                mode2 += 1
            else:
                mode2 = 0
            if mode0 > 0:
                mode0 -= 1
            if score_me + 10.0 <= score_opp and t - mode3 >= 19 and mode2 == 0:
                opp_d = count_zeros(opp_hist, t + 1)
                expv = 0.5 * (t + 1)
                sd = np.sqrt(0.25 * (t + 1) + 1e-12)
                if np.abs(opp_d - expv) >= 3.0 * sd:
                    mode0 = -2
        return mode0, mode1, mode2, mode3

    if strategy == 4:
        if mode0 > 0:
            mode0 -= 1
            if mode0 == 0:
                mode1 += 1
        else:
            if opp_hist[t] == 0 and my_hist[t] == 1:
                if t == 0 or not (opp_hist[t - 1] == 0 and my_hist[t - 1] == 1):
                    mode0 = max(1, mode1)
        return mode0, mode1, mode2, mode3

    if strategy == 5:
        if opp_hist[t] == 0:
            mode0 = 1
        return mode0, mode1, mode2, mode3

    if strategy == 6:
        if t >= 9 and opp_hist[t] == 0:
            mode0 = 1
        return mode0, mode1, mode2, mode3

    if strategy == 7:
        if t == 55:
            if looks_random_simple(opp_hist, t + 1):
                mode0 = 1
            else:
                gap = 5 + int(np.random.random() * 11.0)
                mode1 = t + gap
        elif t > 55 and mode0 == 0 and t == mode1:
            gap = 5 + int(np.random.random() * 11.0)
            mode1 = t + gap
        return mode0, mode1, mode2, mode3

    if strategy == 16:
        if t == 0:
            if opp_hist[0] == 0:
                mode0 = 2
            else:
                mode0 = 1
        elif t >= 1 and mode0 == 1 and opp_hist[t] == 0:
            mode0 = 2
        return mode0, mode1, mode2, mode3

    if strategy == 19:
        if t == 2 and mode0 == 0:
            if opp_hist[1] == 1 and opp_hist[2] == 1:
                mode0 = 1
            else:
                mode0 = 2
        return mode0, mode1, mode2, mode3

    return mode0, mode1, mode2, mode3


@njit(fastmath=True)
def simulate_match(strategy_i, strategy_j, rounds):
    hist_i = np.zeros(rounds, dtype=np.int8)
    hist_j = np.zeros(rounds, dtype=np.int8)

    score_i = 0.0
    score_j = 0.0
    coop_i = 0
    coop_j = 0

    i_m0 = 0
    i_m1 = 1
    i_m2 = 0
    i_m3 = -100

    j_m0 = 0
    j_m1 = 1
    j_m2 = 0
    j_m3 = -100

    for t in range(rounds):
        ai = choose_action(
            strategy_i, hist_i, hist_j, t, rounds, score_i, score_j, i_m0, i_m1, i_m2, i_m3
        )
        aj = choose_action(
            strategy_j, hist_j, hist_i, t, rounds, score_j, score_i, j_m0, j_m1, j_m2, j_m3
        )

        hist_i[t] = ai
        hist_j[t] = aj

        pi, pj = payoff(ai, aj)
        score_i += pi
        score_j += pj
        coop_i += ai
        coop_j += aj

        i_m0, i_m1, i_m2, i_m3 = update_mode_after_round(
            strategy_i, hist_i, hist_j, t, score_i, score_j, i_m0, i_m1, i_m2, i_m3
        )
        j_m0, j_m1, j_m2, j_m3 = update_mode_after_round(
            strategy_j, hist_j, hist_i, t, score_j, score_i, j_m0, j_m1, j_m2, j_m3
        )

    return score_i, score_j, coop_i / rounds, coop_j / rounds


@njit(fastmath=True)
def play_and_move_numba(
    xs,
    s,
    R_INTERACT,
    ALPHA,
    BETA,
    P_EXP,
    Q_EXP,
    MAX_STEP,
    BASE_NOISE_SIGMA,
    DIST_EPS,
    ROUNDS_PER_PAIR,
    MOVE_SUBSTEPS,
):
    n = xs.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    moves = np.zeros(n, dtype=np.float64)

    dx_full = xs[:, None] - xs[None, :]
    D2 = dx_full * dx_full
    within = (D2 <= R_INTERACT**2) & (D2 > 0)
    i_idx, j_idx = np.nonzero(np.triu(within, 1))

    for k in range(i_idx.shape[0]):
        i = i_idx[k]
        j = j_idx[k]
        dist = np.sqrt(D2[i, j] + DIST_EPS**2)
        dx = dx_full[i, j]

        si, sj, ci_ratio, cj_ratio = simulate_match(s[i], s[j], ROUNDS_PER_PAIR)
        scores[i] += si
        scores[j] += sj

        if cj_ratio >= 0.5:
            moves[i] += ALPHA / (dist**P_EXP) * dx * cj_ratio
        else:
            moves[i] -= BETA / (dist**Q_EXP) * dx * (1.0 - cj_ratio)

        if ci_ratio >= 0.5:
            moves[j] -= ALPHA / (dist**P_EXP) * dx * ci_ratio
        else:
            moves[j] += BETA / (dist**Q_EXP) * dx * (1.0 - ci_ratio)

    deg = within.sum(axis=1).astype(np.float64)
    mask = deg > 0
    moves[mask] /= deg[mask]

    if BASE_NOISE_SIGMA > 0:
        moves += np.random.normal(0.0, BASE_NOISE_SIGMA, n)

    norms = np.abs(moves) + 1e-12
    moves *= np.minimum(1.0, MAX_STEP / norms)
    delta = moves / MOVE_SUBSTEPS

    for _ in range(MOVE_SUBSTEPS):
        xs += delta
        for i in range(n):
            xs[i] = reflect_01_numba(xs[i])

    return scores, xs


def play_and_move(agents):
    xs = np.array([a.x for a in agents], dtype=np.float64)
    strat_to_int = {s: i for i, s in enumerate(STRATEGIES)}
    s = np.array([strat_to_int[a.strategy] for a in agents], dtype=np.int32)
    scores, new_xs = play_and_move_numba(
        xs,
        s,
        R_INTERACT,
        ALPHA,
        BETA,
        P_EXP,
        Q_EXP,
        MAX_STEP,
        BASE_NOISE_SIGMA,
        DIST_EPS,
        ROUNDS_PER_PAIR,
        MOVE_SUBSTEPS,
    )
    for i, a in enumerate(agents):
        a.x = new_xs[i]
        a.score = scores[i]


# =========================
# Death-Birth
# =========================
def select_and_reproduce_DB(agents):
    scores = np.array([a.score for a in agents])
    order = np.argsort(scores)
    bottom_idx = order[:BOTTOM_K]
    top_repro = order[::-1][:TOP_K]
    for k, dead_i in enumerate(bottom_idx):
        parent = agents[top_repro[k % len(top_repro)]]
        child = Agent(random.random(), parent.strategy, 0.0)
        agents[dead_i] = child


# =========================
# 저장 관련
# =========================
def get_counts_path(seed):
    return os.path.join(OUT_DIR, f"counts_{STRAT_FILENAME}_seed{seed}.png")


def get_gif_path(seed):
    return os.path.join(OUT_DIR, f"spatial_evolution_{STRAT_FILENAME}_seed{seed}.gif")


def get_log_path(seed):
    return os.path.join(OUT_DIR, f"agent_log_1D_seed{seed}_{STRAT_FILENAME}.csv")


def clear_seed_outputs(seed):
    for path in [get_counts_path(seed), get_gif_path(seed), get_log_path(seed)]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


# =========================
# counts, GIF
# =========================
def save_counts_plot(counts_per_gen, seed):
    gens = np.arange(len(counts_per_gen))
    fig, ax = plt.subplots(figsize=(12, 6))
    for s in STRATEGIES:
        y = [d.get(s, 0) for d in counts_per_gen]
        ax.plot(gens, y, label=s, linewidth=1.8, color=COLOR_MAP_STRAT[s])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Count")
    ax.set_title(f"Strategy counts — {STRAT_FILENAME} (seed={seed})")
    ax.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(get_counts_path(seed), dpi=150)
    plt.close()


def make_spatial_gif(frames_agents, seed):
    gif_path = get_gif_path(seed)
    frames = []
    bins = np.linspace(0, 1, 51)
    centers = (bins[:-1] + bins[1:]) / 2

    for idx, agents in enumerate(frames_agents):
        if idx % 10 != 0 and idx != len(frames_agents) - 1:
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        for s in STRATEGIES:
            xs = [a.x for a in agents if a.strategy == s]
            h = np.histogram(xs, bins)[0]
            ax.plot(centers, h, color=COLOR_MAP_STRAT[s], lw=1.5, label=s)
        ax.set_xlabel("Position x in [0,1]")
        ax.set_ylabel("Number of agents per bin")
        ax.set_title(f"Gen {idx} ({STRAT_FILENAME}, seed={seed})")
        ax.legend(ncol=4, fontsize=7)
        ax.grid(True, alpha=0.3)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        frames.append(np.asarray(canvas.buffer_rgba()))
        plt.close(fig)

    imageio.mimsave(gif_path, frames, duration=0.8)
    print(f"✅ GIF 완성 → {gif_path}")


# =========================
# 1회 실행
# =========================
def run_one_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    clear_seed_outputs(seed)

    start_time = time.time()

    agents = [
        Agent(random.random(), s_name, 0.0) for s_name in STRATEGIES for _ in range(N_PER_STRAT)
    ]

    counts_per_gen = []
    frames_agents = []
    log_rows = []

    print(
        f"\n[시작] 20 strategies × {N_PER_STRAT} = {N_INIT} agents | seed={seed} | numba accelerated"
    )

    for gen in range(GENS):
        play_and_move(agents)
        select_and_reproduce_DB(agents)

        d = {s: 0 for s in STRATEGIES}
        for a in agents:
            d[a.strategy] += 1
        counts_per_gen.append(d)

        for slot, a in enumerate(agents):
            log_rows.append([gen, slot, a.x, a.strategy, a.score])

        if gen % 20 == 0 or gen == GENS - 1:
            elapsed = time.time() - start_time
            progress = (gen + 1) / GENS * 100
            eta = elapsed / (gen + 1) * (GENS - gen - 1) if gen > 0 else 0.0
            top3 = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:3]
            top3_txt = " | ".join([f"{k}:{v}" for k, v in top3])
            print(
                f"SEED {seed:2d} | Gen {gen:4d}/{GENS} | {progress:5.1f}% | {top3_txt} | 경과 {elapsed/60:.1f}분 | 예상 남은 {eta/60:.1f}분"
            )

        if gen % 10 == 0 or gen == GENS - 1:
            frames_agents.append([Agent(a.x, a.strategy) for a in agents])

    with open(get_log_path(seed), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["gen", "slot", "x", "strategy", "score"])
        writer.writerows(log_rows)

    save_counts_plot(counts_per_gen, seed)
    make_spatial_gif(frames_agents, seed)

    final_counts = counts_per_gen[-1]
    winner = max(final_counts.items(), key=lambda kv: kv[1])[0]
    elapsed_total = time.time() - start_time

    print(f"✅ seed={seed} 완료 | winner={winner} | 총 {elapsed_total/60:.1f}분")

    return {
        "seed": seed,
        "winner": winner,
        "elapsed_sec": elapsed_total,
        **{f"final_{s}": final_counts[s] for s in STRATEGIES},
    }


def save_summary_csv(run_summaries):
    summary_path = os.path.join(OUT_DIR, f"summary_{STRAT_FILENAME}_10runs.csv")
    fieldnames = ["seed", "winner", "elapsed_sec"] + [f"final_{s}" for s in STRATEGIES]

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_summaries)

    print(f"\n✅ 반복실험 요약 저장 → {summary_path}")


def print_overall_summary(run_summaries):
    winner_counts = {}
    for row in run_summaries:
        w = row["winner"]
        winner_counts[w] = winner_counts.get(w, 0) + 1

    print("\n===== 전체 10회 반복실험 요약 =====")
    for strat, cnt in sorted(winner_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{strat:20s}: {cnt}회 우승")

    print("\n평균 최종 개체 수:")
    for s in STRATEGIES:
        mean_cnt = np.mean([row[f"final_{s}"] for row in run_summaries])
        print(f"{s:20s}: {mean_cnt:7.2f}")


# =========================
# 메인
# =========================
if __name__ == "__main__":
    print("메인 실행 시작")
    all_start = time.time()
    run_summaries = []

    for seed in SEED_LIST:
        summary = run_one_seed(seed)
        run_summaries.append(summary)

    save_summary_csv(run_summaries)
    print_overall_summary(run_summaries)

    total_elapsed = time.time() - all_start
    print(f"\n✅ 전체 완료! 총 경과 시간: {total_elapsed/60:.1f}분")
    print(f"저장 폴더: {OUT_DIR}")
