#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate.py — 用「模拟被试」验证评分内核是否真的能区分不同的元认知水平。

为什么需要这一步：
    在拿到真实模型 API 之前，我们必须先证明**评分函数本身是有辨别力的**。
    如果一套指标连"过度自信者"和"良好校准者"都分不开，那它测出来的分数毫无意义。

四个原型被试（误差尺度按难度分级，模拟"越难的题人/模型错得越多"）：
    calibrated      知道自己错多少，区间宽度 ∝ 真实误差尺度 → 覆盖率缺口≈0
    overconfident   真实误差大但区间很窄            → 覆盖率缺口大（过度自信）
    hedger          永远给超宽区间                  → 覆盖率满分，但锐度极差
    uniform_width   区间宽度不随难度变化            → 辨别力≈0（不知道自己什么时候不准）

关键待验证命题：
    「只看覆盖率缺口」会被 hedger 骗过去（它覆盖率满分，看起来最"谦虚"），
    但 Winkler 区间得分（proper scoring rule）会识破它。
"""

import random

from scoring import score_interval_responses

# 各难度下的真实相对误差尺度（越大越难）
SIGMA_BY_TIER = {1: 0.02, 2: 0.08, 3: 0.20, 4: 0.45}

# 90% 覆盖对应的正态分位数
Z90 = 1.645

ARCHETYPES = {
    "calibrated": {"k": Z90, "desc": "良好校准：区间宽度 ∝ 自身真实误差尺度"},
    "overconfident": {"k": 0.35, "desc": "过度自信：区间比实际误差窄得多"},
    "hedger": {"k": 8.0, "desc": "对冲者：永远给超宽区间（覆盖率满分但无信息量）"},
    "uniform_width": {"k": None, "desc": "宽度不随难度变化：不知道自己何时不准"},
}


def simulate_respondent(items, archetype, seed=7):
    """让某个原型被试在题集上作答，返回可直接送进 score_interval_responses 的记录。"""
    if archetype not in ARCHETYPES:
        raise ValueError("未知原型：%s" % archetype)
    rng = random.Random(seed)
    records = []
    for it in items:
        truth = float(it["truth"])
        sigma = SIGMA_BY_TIER.get(it.get("tier", 1), 0.1)
        scale = max(abs(truth), 1e-9)

        # 点估计：真值加上一个相对误差
        err = rng.gauss(0.0, sigma)
        point = truth * (1.0 + err)

        # 区间半宽：各原型的分歧点
        if archetype == "uniform_width":
            half = 0.15 * scale            # 恒定 15%，不随难度走
        else:
            half = ARCHETYPES[archetype]["k"] * sigma * scale

        records.append(
            {
                "id": it["id"],
                "family": it["family"],
                "tier": it.get("tier", 1),
                "contamination": it.get("contamination", "fresh"),
                "point": point,
                "low": point - half,
                "high": point + half,
                "truth": truth,
            }
        )
    return records


def run_validation(items=None, seed=7, verbose=True):
    """跑一遍全部原型，返回 {原型名: 指标} 和排序结论。"""
    if items is None:
        from items import generate_dataset

        items = generate_dataset(n=60, seed=20261004)

    results = {}
    for name in ARCHETYPES:
        recs = simulate_respondent(items, name, seed=seed)
        results[name] = score_interval_responses(recs, claimed_coverage=0.90)

    if verbose:
        _print_table(results)
    return results


def _print_table(results):
    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("模拟被试验证（声称覆盖率 90%，越低越好的指标已标注）")
    print()
    print("| 原型 | 命中率 | 覆盖率缺口↑ | 相对Winkler↓ | 锐度(相对半宽) | 辨别力ρ |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, m in results.items():
        print(
            "| %s | %.2f | %+.3f | %.3f | %.3f | %+.3f |"
            % (
                name,
                m["hit_rate"],
                m["coverage_gap"],
                m["mean_rel_winkler"],
                m["mean_rel_half_width"],
                m["metacog_sensitivity"],
            )
        )
    print()
    print("判读：")
    print("  · 覆盖率缺口 > 0 = 过度自信；< 0 = 过度保守")
    print("  · relative Winkler 是主指标（越小越好），同时惩罚「没盖住」和「不敢下判断」")
    print("  · 辨别力 ρ = 区间宽度与实际误差的秩相关，≈0 表示它根本不知道自己何时不准")
    return results


if __name__ == "__main__":
    run_validation()
