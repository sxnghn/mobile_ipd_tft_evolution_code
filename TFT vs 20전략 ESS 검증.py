#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1차원 이동형 반복 죄수의 딜레마 연구 - TFT의 진화적 안정성(ESS) 검증
=====================================================================

목적
----
저항(reciprocity) 전략 TFT가 "거의 전부 TFT인 집단(99% 또는 95%)"에서,
나머지(1% 또는 5%)를 차지한 단일 침입 전략을 몰아낼 수 있는지를
well-mixed replicator dynamics 로 검증한다.

이는 개요서의 'Replicator Map 과 고정점 구조' 섹션(불안정 고정점 p* = 1/21)
과 동일한 평균장(mean-field) 틀이다.

게임
----
- 결정론적 R라운드 IPD (개요서 기본: R = 12)
- 보수행렬 (T,R,P,S) = (5,3,1,0)
- 검증: TFT-TFT=36, ALLD-ALLD=12, TFT vs ALLD = 11/16  (R=12에서)

ESS 판정
--------
거주 전략 = TFT, 침입 전략 = X.
A(I,J) = I 가 J 와 R라운드 게임에서 얻는 평균 총점.
- 1차 조건: A(TFT,TFT) > A(X,TFT)            -> 강한 ESS (몰아냄)
- 동률이면 2차 조건: A(TFT,X) > A(X,X)        -> ESS (몰아냄)
- A(X,TFT) > A(TFT,TFT)                       -> 침입당함 (TFT 멸종)
- 둘 다 동률                                  -> 중립 드리프트(strict ESS 아님)

replicator map (개요서 식과 동일한 형태, 이산):
    p' = p*pi_T / ( p*pi_T + (1-p)*pi_X )
    pi_T = p*A(TFT,TFT) + (1-p)*A(TFT,X)
    pi_X = p*A(X,TFT)   + (1-p)*A(X,X)
초기비율 p0 = 0.99, 0.95 에서 각각 수렴시켜 TFT 분율이 1로 가는지 본다.

주의: 일부 전략(Tideman&Chieruzzi, Graaskamp, (Revised)Downing, Nydegger
lookup, ZD-Extortioner 계수)은 Axelrod 원정의가 매우 정교하여
'합리적 근사'로 구현했다. 자세한 내용은 각 클래스 주석 참고.
"""

import csv
import os
import random

# ----------------------------------------------------------------------
# 기본 설정
# ----------------------------------------------------------------------
C, D = 1, 0  # 협력 / 배신
SEED = 12345
REPS_STOCH = 4000  # 확률적 전략 쌍의 보수 평균 반복수

PAYOFF = {  # (내 행동, 상대 행동) -> (내 점수, 상대 점수)
    (C, C): (3, 3),
    (C, D): (0, 5),
    (D, C): (5, 0),
    (D, D): (1, 1),
}


def payoff(a, b):
    return PAYOFF[(a, b)]


# ----------------------------------------------------------------------
# 전략 정의.  각 전략은 move(self, me, opp) 를 가진다.
#   me  = 내 과거 행동 리스트 (오래된 -> 최근)
#   opp = 상대 과거 행동 리스트
#   self.total_rounds 는 게임 시작 전에 주입됨 (Feld, Tideman, Graaskamp용)
# stochastic = True 인 전략은 보수 추정 시 여러 번 반복 평균.
# ----------------------------------------------------------------------
class Strategy:
    name = "?"
    stochastic = False
    total_rounds = 12

    def move(self, me, opp):
        raise NotImplementedError


# 1. Tit for Tat (Rapoport) -- 거주 전략
class TFT(Strategy):
    name = "TFT"

    def move(self, me, opp):
        return C if not opp else opp[-1]


# 2. Tideman & Chieruzzi  (근사 구현)
#    TFT 기반 + 보복 누적 + fresh start + 마지막 2턴 자동 배신
class Tideman(Strategy):
    name = "Tideman&Chieruzzi"

    def __init__(self):
        self.last_fresh = -(10**9)
        self.streaks = 0
        self.retal_remaining = 0
        self.fresh_pending = 0

    def move(self, me, opp):
        t = len(me)
        T = self.total_rounds
        if t == 0:
            return C
        if t >= T - 2:  # 마지막 2턴 자동 배신
            return D
        if self.fresh_pending > 0:  # fresh start: 연속 2회 협력 중
            self.fresh_pending -= 1
            return C
        mys = sum(payoff(me[i], opp[i])[0] for i in range(t))
        ops = sum(payoff(me[i], opp[i])[1] for i in range(t))
        # fresh start 조건: 상대가 10점 이상 뒤짐 & 직전 fresh 후 20턴 & 잔여 10턴 이상
        if (ops <= mys - 10) and (t - self.last_fresh >= 20) and (T - t >= 10):
            self.last_fresh = t
            self.streaks = 0
            self.retal_remaining = 0
            self.fresh_pending = 1  # 이번 + 다음 = 2회 협력
            return C
        if self.retal_remaining > 0:  # 누적 보복 진행 중
            self.retal_remaining -= 1
            return D
        if opp[-1] == D and (t == 1 or opp[-2] == C):  # 상대의 새 배신 구간 시작
            self.streaks += 1
            self.retal_remaining = self.streaks - 1
            return D
        return opp[-1]  # 그 외 TFT


# 3. Nydegger
#    1~3턴 TFT(특수규칙 포함), 4턴 이후 A=16a1+4a2+a3 lookup
class Nydegger(Strategy):
    name = "Nydegger"
    AS = {1, 6, 7, 17, 22, 23, 26, 29, 30, 31, 33, 38, 39, 45, 49, 54, 55, 58, 61}

    @staticmethod
    def _code(my, op):
        if my == C and op == C:
            return 0
        if my == C and op == D:
            return 2  # 상대만 D
        if my == D and op == C:
            return 1  # 나만 D
        return 3  # 둘 다 D

    def move(self, me, opp):
        t = len(me)
        if t == 0:
            return C
        if t == 1:
            return opp[-1]  # TFT
        if t == 2:
            # 나만 협력(C,D) 후 나만 배신(D,C) 패턴이면 배신
            if me[0] == C and opp[0] == D and me[1] == D and opp[1] == C:
                return D
            return opp[-1]  # TFT
        a1 = self._code(me[-1], opp[-1])
        a2 = self._code(me[-2], opp[-2])
        a3 = self._code(me[-3], opp[-3])
        A = 16 * a1 + 4 * a2 + a3
        return D if A in self.AS else C


# 4. Grofman (확률적)
class Grofman(Strategy):
    name = "Grofman"
    stochastic = True

    def move(self, me, opp):
        if not me:
            return C
        if me[-1] == opp[-1]:
            return C
        return C if random.random() < 2.0 / 7.0 else D


# 5. Shubik  (누적 보복 길이 증가)
class Shubik(Strategy):
    name = "Shubik"

    def __init__(self):
        self.retal_len = 0
        self.retal_remaining = 0

    def move(self, me, opp):
        if not me:
            return C
        if self.retal_remaining > 0:
            self.retal_remaining -= 1
            return D
        if me[-1] == C and opp[-1] == D:  # 협력 중 상대 배신 -> 보복 길이 +1
            self.retal_len += 1
            self.retal_remaining = self.retal_len - 1
            return D
        return C


# 7. Grudger (Friedman) : 한 번이라도 배신하면 영원히 배신
class Grudger(Strategy):
    name = "Grudger"

    def move(self, me, opp):
        return D if D in opp else C


# 8. Davis : 1~10턴 협력, 이후 grudger
class Davis(Strategy):
    name = "Davis"

    def move(self, me, opp):
        if len(me) < 10:
            return C
        return D if D in opp else C


# 9. Graaskamp (근사 구현)
#    1~50 TFT, 51 배신, 52~56 TFT, 이후 랜덤판별/주기적 시험배신
class Graaskamp(Strategy):
    name = "Graaskamp"
    stochastic = True

    def __init__(self):
        self.defect_forever = False
        self.next_probe = None

    def move(self, me, opp):
        t = len(me)
        if t == 0:
            return C
        if t < 50:
            return opp[-1]  # TFT
        if t == 50:
            return D  # 51번째턴 시험 배신
        if t <= 55:
            return opp[-1]  # 52~56 TFT
        if self.defect_forever:
            return D
        # 56턴 직후 1회 판별: 상대가 52~56에서 협력 복귀 못했으면 랜덤/착취형으로 간주
        if t == 56:
            window = opp[51:56]
            if window.count(C) <= 2:  # 협력 복귀 미흡 -> 영원히 배신
                self.defect_forever = True
                return D
            self.next_probe = t + random.randint(5, 15)
        if self.next_probe is not None and t >= self.next_probe:
            self.next_probe = t + random.randint(5, 15)
            return D  # 5~15턴 간격 시험 배신
        return opp[-1]  # 기본 TFT


# 10. Downing (Revised Downing 근사)
#     조건부 협력확률 추정 후 기대점수 큰 쪽 선택
class Downing(Strategy):
    name = "Downing"

    def move(self, me, opp):
        t = len(me)
        if t == 0:
            return C  # 초기 협력으로 추정 시작
        cc = tc = cd = td = 0
        for i in range(t - 1):
            if me[i] == C:
                tc += 1
                if opp[i + 1] == C:
                    cc += 1
            else:
                td += 1
                if opp[i + 1] == C:
                    cd += 1
        pC = cc / tc if tc else 0.9  # 내가 협력했을 때 상대 협력확률
        pD = cd / td if td else 0.9  # 내가 배신했을 때 상대 협력확률
        EC = pC * 3 + (1 - pC) * 0  # 협력 시 기대점수 (R/S)
        ED = pD * 5 + (1 - pD) * 1  # 배신 시 기대점수 (T/P)
        return C if EC >= ED else D


# 11. Feld (확률적) : 협력 확률이 게임 후반으로 갈수록 1.0->0.5 선형 감소
class Feld(Strategy):
    name = "Feld"
    stochastic = True

    def move(self, me, opp):
        if not opp:
            return C
        if opp[-1] == D:
            return D
        t = len(me)
        frac = t / max(1, self.total_rounds - 1)
        p_coop = 1.0 - 0.5 * frac
        return C if random.random() < p_coop else D


# 12. Joss (확률적) : 상대 배신->배신, 상대 협력->90% 협력
class Joss(Strategy):
    name = "Joss"
    stochastic = True

    def move(self, me, opp):
        if not opp:
            return C
        if opp[-1] == D:
            return D
        return C if random.random() < 0.9 else D


# 13. Tullock (확률적) : 1~11 협력, 이후 최근10턴 상대협력률-10%로 협력
class Tullock(Strategy):
    name = "Tullock"
    stochastic = True

    def move(self, me, opp):
        t = len(me)
        if t < 11:
            return C
        last10 = opp[-10:]
        rate = last10.count(C) / len(last10)
        p = max(0.0, rate - 0.10)
        return C if random.random() < p else D


# 15. Random
class RandomStrat(Strategy):
    name = "Random"
    stochastic = True

    def move(self, me, opp):
        return C if random.random() < 0.5 else D


# 16. ALLC
class ALLC(Strategy):
    name = "ALLC"

    def move(self, me, opp):
        return C


# 17. ALLD
class ALLD(Strategy):
    name = "ALLD"

    def move(self, me, opp):
        return D


# 18. ZD Extortioner (ZD-Extort-2 근사) : memory-one 확률벡터
#     순서 = P(C | 직전(내,상대)) for (CC,CD,DC,DD)
class ZDExtort(Strategy):
    name = "ZD-Extortioner"
    stochastic = True
    P = {(C, C): 8.0 / 9.0, (C, D): 0.5, (D, C): 1.0 / 3.0, (D, D): 0.0}

    def move(self, me, opp):
        if not me:
            return C
        p = self.P[(me[-1], opp[-1])]
        return C if random.random() < p else D


# 19. Tester : 첫턴 배신, 상대 보복하면 사과 후 TFT, 아니면 착취(교대)
class Tester(Strategy):
    name = "Tester"

    def __init__(self):
        self.is_tft = False

    def move(self, me, opp):
        if not me:
            return D
        if not self.is_tft and opp[-1] == D:  # 상대가 강하게 보복 -> 사과
            self.is_tft = True
            return C
        if self.is_tft:
            return opp[-1]  # TFT
        t = len(me)
        if t in (1, 2):
            return C
        return D if me[-1] == C else C  # 착취: C/D 교대


# 20. Pavlov (Win-Stay Lose-Shift) : 직전 두 행동이 같으면 협력
class Pavlov(Strategy):
    name = "Pavlov(WSLS)"

    def move(self, me, opp):
        if not me:
            return C
        return C if me[-1] == opp[-1] else D


# 21. Tit for Two Tats : 상대가 직전 2턴 연속 배신해야 배신
class TF2T(Strategy):
    name = "TitForTwoTats"

    def move(self, me, opp):
        if len(opp) < 2:
            return C
        return D if (opp[-1] == D and opp[-2] == D) else C


# 22. Prober : DCC, 상대가 _CC면 영원히 배신, 3턴내 배신하면 TFT
class Prober(Strategy):
    name = "Prober"

    def move(self, me, opp):
        t = len(me)
        if t == 0:
            return D
        if t in (1, 2):
            return C
        if opp[1] == C and opp[2] == C:  # 2,3턴 모두 협력 -> 착취
            return D
        return opp[-1]  # 그 외 TFT


# ----------------------------------------------------------------------
# 침입 전략 목록 (TFT 자신 포함 = 20개, TFT는 자기 대조군)
# ----------------------------------------------------------------------
INVADERS = [
    TFT,
    Tideman,
    Nydegger,
    Grofman,
    Shubik,
    Grudger,
    Davis,
    Graaskamp,
    Downing,
    Feld,
    Joss,
    Tullock,
    RandomStrat,
    ALLC,
    ALLD,
    ZDExtort,
    Tester,
    Pavlov,
    TF2T,
    Prober,
]


# ----------------------------------------------------------------------
# 게임 엔진
# ----------------------------------------------------------------------
def play_game(ClsA, ClsB, rounds):
    a, b = ClsA(), ClsB()
    a.total_rounds = b.total_rounds = rounds
    ha, hb = [], []
    sa = sb = 0
    for _ in range(rounds):
        ma = a.move(ha, hb)
        mb = b.move(hb, ha)
        pa, pb = payoff(ma, mb)
        sa += pa
        sb += pb
        ha.append(ma)
        hb.append(mb)
    return sa, sb


def mean_payoff(ClsA, ClsB, rounds):
    """A 가 B 에게서 얻는 평균 총점."""
    reps = REPS_STOCH if (ClsA.stochastic or ClsB.stochastic) else 1
    tot = 0.0
    for _ in range(reps):
        sa, _ = play_game(ClsA, ClsB, rounds)
        tot += sa
    return tot / reps


# ----------------------------------------------------------------------
# replicator dynamics & ESS 판정
# ----------------------------------------------------------------------
def replicate(p0, a_tt, a_tx, a_xt, a_xx, gens=200000, eps=1e-13):
    p = p0
    for _ in range(gens):
        pi_t = p * a_tt + (1 - p) * a_tx
        pi_x = p * a_xt + (1 - p) * a_xx
        denom = p * pi_t + (1 - p) * pi_x
        if denom <= 0:
            break
        pn = p * pi_t / denom
        if pn >= 1 - 1e-15:
            return 1.0
        if pn <= 1e-15:
            return 0.0
        if abs(pn - p) < eps:
            return pn
        p = pn
    return p


def outcome_label(final_p, p0, neutral):
    if neutral:
        return "중립(드리프트)"
    if final_p > 0.999:
        return "몰아냄(TFT 고정)"
    if final_p < 0.001:
        return "TFT 멸종"
    return f"공존(TFT={final_p:.3f})"


def ess_verdict(a_tt, a_tx, a_xt, a_xx, tol=0.20):
    if a_tt > a_xt + tol:
        return "ESS(강한 1차조건)", False
    if a_tt < a_xt - tol:
        return "비ESS(침입당함)", False
    # 1차 동률 -> 2차
    if a_tx > a_xx + tol:
        return "ESS(2차조건)", False
    if a_tx < a_xx - tol:
        return "비ESS(2차조건)", False
    return "중립(strict ESS 아님)", True


# ----------------------------------------------------------------------
# 보수행렬 검증 (개요서 11/16/36/12 및 1/21 재현 확인)
# ----------------------------------------------------------------------
def validate(rounds):
    tt, _ = play_game(TFT, TFT, rounds)
    t_vs_d, d_vs_t = play_game(TFT, ALLD, rounds)
    dd, _ = play_game(ALLD, ALLD, rounds)
    print(f"[검증 R={rounds}] TFT-TFT={tt}, ALLD-ALLD={dd}, " f"TFT vs ALLD = {t_vs_d}/{d_vs_t}")
    # TFT vs ALLD 의 replicator 불안정 고정점 (개요서 1/21)
    a_tt, a_tx, a_xt, a_xx = tt, t_vs_d, d_vs_t, dd
    lo, hi = 1e-6, 1 - 1e-6
    for _ in range(200):  # F(p)=p 의 내부근 이분탐색
        mid = (lo + hi) / 2
        pi_t = mid * a_tt + (1 - mid) * a_tx
        pi_x = mid * a_xt + (1 - mid) * a_xx
        f = mid * pi_t / (mid * pi_t + (1 - mid) * pi_x) - mid
        # F(p)-p : p* 부근에서 부호 변화
        pi_t2 = lo * a_tt + (1 - lo) * a_tx
        pi_x2 = lo * a_xt + (1 - lo) * a_xx
        flo = lo * pi_t2 / (lo * pi_t2 + (1 - lo) * pi_x2) - lo
        if (flo > 0) == (f > 0):
            lo = mid
        else:
            hi = mid
    print(f"           내부 고정점 p* ≈ {mid:.5f}  (이론 1/21 = {1/21:.5f})\n")


# ----------------------------------------------------------------------
# 메인 실행
# ----------------------------------------------------------------------
def run(rounds, p_list=(0.99, 0.95)):
    print("=" * 100)
    print(f"  TFT 진화적 안정성(ESS) 검증   |   게임 길이 R = {rounds} 라운드")
    print("=" * 100)
    validate(rounds)

    a_tt = mean_payoff(TFT, TFT, rounds)  # = 36 (R=12)
    rows = []

    header = (
        f"{'침입 전략':<20}{'A(X|TFT)':>9}{'A(TFT|X)':>9}{'A(X|X)':>8}"
        f"{'  99%→':>9}{'  95%→':>9}   {'ESS 판정'}"
    )
    print(header)
    print("-" * 100)

    for Cls in INVADERS:
        a_tx = mean_payoff(TFT, Cls, rounds)  # TFT 가 X 에게
        a_xt = mean_payoff(Cls, TFT, rounds)  # X 가 TFT 에게
        a_xx = mean_payoff(Cls, Cls, rounds)  # X 가 X 에게

        verdict, neutral = ess_verdict(a_tt, a_tx, a_xt, a_xx)
        res = {}
        for p0 in p_list:
            fp = replicate(p0, a_tt, a_tx, a_xt, a_xx)
            res[p0] = (fp, outcome_label(fp, p0, neutral))

        print(
            f"{Cls.name:<20}{a_xt:>9.2f}{a_tx:>9.2f}{a_xx:>8.2f}"
            f"{res[0.99][0]:>9.3f}{res[0.95][0]:>9.3f}   {verdict}"
        )

        rows.append(
            {
                "rounds": rounds,
                "invader": Cls.name,
                "A_X_vs_TFT": round(a_xt, 4),
                "A_TFT_vs_X": round(a_tx, 4),
                "A_X_vs_X": round(a_xx, 4),
                "A_TFT_vs_TFT": round(a_tt, 4),
                "final_p_99": round(res[0.99][0], 5),
                "result_99": res[0.99][1],
                "final_p_95": round(res[0.95][0], 5),
                "result_95": res[0.95][1],
                "ESS_verdict": verdict,
            }
        )
    print()
    return rows


if __name__ == "__main__":
    random.seed(SEED)
    all_rows = []
    for R in (12, 200):
        all_rows.extend(run(R))

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "tft_ess_results.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"결과 저장: {out}")
