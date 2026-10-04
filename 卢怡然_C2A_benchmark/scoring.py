#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scoring.py — ΔE-Bench 评分内核（Module A / B 指标）

指标族（刻意把三件事分开测，因为它们是三件事）：
  1. 校准 Calibration   —— 声称的覆盖率 vs 实际命中率（覆盖率缺口）
  2. 锐度 Sharpness     —— 区间宽度（只给超宽区间不算本事）
  3. 辨别力 Discrimination —— 自报的不确定性是否真的预测了实际误差大小

主指标用 **Winkler 区间得分**（Gneiting & Raftery 2007 的 proper scoring rule）：
它是 校准 + 锐度 的联合度量，且**不可通过套牢式加宽区间来刷分**——
这一点我们在 simulate.py 里用"永远给超宽区间的对冲者"实证验证过。

只依赖标准库。
"""

import math

# ---------------------------------------------------------------- 单题得分


def winkler_score(low, high, truth, alpha=0.10):
    """Winkler / interval score。越小越好。

    S = (u-l) + (2/α)(l-y)·1{y<l} + (2/α)(y-u)·1{y>u}
    真值落在区间内 → 只罚宽度（越窄越好）
    真值落在区间外 → 额外按偏离距离罚（罚得越重，声称的覆盖率越高）
    """
    if high < low:
        low, high = high, low
    width = high - low
    if truth < low:
        return width + (2.0 / alpha) * (low - truth)
    if truth > high:
        return width + (2.0 / alpha) * (truth - high)
    return width


def relative_winkler(low, high, truth, alpha=0.10):
    """按真值量级归一化的 Winkler 得分，便于跨题族比较。"""
    scale = max(abs(truth), 1e-9)
    return winkler_score(low, high, truth, alpha) / scale


# ---------------------------------------------------------------- 统计工具


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _ranks(values):
    """平均秩（并列取平均），用于 Spearman。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def spearman(xs, ys):
    return pearson(_ranks(xs), _ranks(ys))


def auroc(scores, labels):
    """AUROC（秩和法，无依赖）。scores 越大表示"越有把握"，labels 1=答对。"""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    total = 0.0
    for p in pos:
        for n in neg:
            total += 1.0 if p > n else (0.5 if p == n else 0.0)
    return total / (len(pos) * len(neg))


def ece(confidences, corrects, bins=10):
    """期望校准误差 ECE：按置信度分箱，加权 |平均置信度 − 实际正确率|。"""
    pairs = [(c, y) for c, y in zip(confidences, corrects) if 0.0 <= c <= 1.0]
    if not pairs:
        return float("nan")
    total = 0.0
    n = len(pairs)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [p for p in pairs if lo <= p[0] < hi or (b == bins - 1 and p[0] >= hi)]
        if not bucket:
            continue
        avg_conf = mean([c for c, _ in bucket])
        acc = mean([1.0 if y else 0.0 for _, y in bucket])
        total += (len(bucket) / n) * abs(avg_conf - acc)
    return total


# ---------------------------------------------------------------- 主体评分


def score_interval_responses(records, claimed_coverage=0.90):
    """Module A 评分。

    records: [{low, high, truth, point(可选), family, contamination}]
    返回指标字典。
    """
    alpha = 1.0 - claimed_coverage
    hits, rel_w, half_widths, rel_errs = [], [], [], []
    for r in records:
        lo, hi, y = float(r["low"]), float(r["high"]), float(r["truth"])
        hits.append(1 if lo <= y <= hi else 0)
        rel_w.append(relative_winkler(lo, hi, y, alpha))
        scale = max(abs(y), 1e-9)
        half_widths.append((hi - lo) / 2.0 / scale)
        point = r.get("point")
        if point is not None:
            rel_errs.append(abs(float(point) - y) / scale)

    hit_rate = mean(hits) if hits else float("nan")
    sens = spearman(half_widths, rel_errs) if len(rel_errs) >= 3 else float("nan")

    return {
        "n": len(records),
        "claimed_coverage": claimed_coverage,
        "hit_rate": hit_rate,
        "coverage_gap": claimed_coverage - hit_rate,      # >0 = 过度自信
        "mean_rel_winkler": mean(rel_w),                  # 主指标，越小越好
        "mean_rel_half_width": mean(half_widths),         # 锐度（越小越敢下判断）
        "metacog_sensitivity": sens,                      # 辨别力：宽度 vs 实际误差
        "mean_rel_abs_error": mean(rel_errs) if rel_errs else float("nan"),
    }


def contamination_split(records, claimed_coverage=0.90):
    """F1（已知题）vs F2（新题）分组评分 → 元认知的抗污染落差。"""
    fresh = [r for r in records if r.get("contamination") == "fresh"]
    known = [r for r in records if r.get("contamination") == "known"]
    out = {"fresh": None, "known": None, "gap": None}
    if len(fresh) >= 3:
        out["fresh"] = score_interval_responses(fresh, claimed_coverage)
    if len(known) >= 3:
        out["known"] = score_interval_responses(known, claimed_coverage)
    if out["fresh"] and out["known"]:
        f = out["fresh"]["coverage_gap"]
        k = out["known"]["coverage_gap"]
        out["gap"] = f - k          # >0：在新题上更过度自信（元认知依赖记忆）
    return out


def score_decisions(decisions, correct_reward=1.0, wrong_penalty=-2.0, abstain_reward=0.0):
    """Module B —— 元认知**控制**评分。

    decisions: [{p_success, correct(若作答时的对错), action: answer|abstain, ...}]
    返回：实际得分 / 总是作答基线 / 全知最优（oracle），以及净收益。
    """
    actual = 0.0
    always = 0.0
    oracle = 0.0
    for d in decisions:
        correct = bool(d["correct"])
        always += correct_reward if correct else wrong_penalty
        oracle += correct_reward if correct else abstain_reward
        if d["action"] == "abstain":
            actual += abstain_reward
        else:
            actual += correct_reward if correct else wrong_penalty
    n = len(decisions) or 1
    return {
        "n": len(decisions),
        "actual_total": actual,
        "always_answer_total": always,
        "oracle_total": oracle,
        "net_gain_vs_always": (actual - always) / n,          # 元认知带来的净收益/题
        "efficiency_vs_oracle": (actual - always) / (oracle - always) if oracle != always else float("nan"),
    }


def deltaE_convergence(trials, window=5, threshold=0.10):
    """Module C —— ΔE 随反馈收敛的速度。

    trials: 按时间顺序的 [{delta_e}]，ΔE = 自报把握 − 实际正确(1/0)
    返回：首窗 ΔE 均值、末窗 ΔE 均值、收敛斜率、收敛所需 trial 数（None=未收敛）。
    """
    if len(trials) < window * 2:
        return {"error": "样本不足，至少需要 %d 个 trial" % (window * 2)}
    des = [float(t["delta_e"]) for t in trials]
    first = mean(des[:window])
    last = mean(des[-window:])
    slope = pearson(list(range(len(des))), des)
    converged_at = None
    for i in range(len(des) - window + 1):
        if abs(mean(des[i : i + window])) <= threshold:
            converged_at = i + window
            break
    return {
        "n": len(des),
        "delta_e_first_window": first,
        "delta_e_last_window": last,
        "reduction": first - last,
        "slope": slope,
        "converged_at_trial": converged_at,
        "threshold": threshold,
    }
