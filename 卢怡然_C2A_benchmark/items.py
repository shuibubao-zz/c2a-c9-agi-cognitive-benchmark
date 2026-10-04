#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
items.py — ΔE-Bench 题目生成器（Module A 用）

设计原则：
  1. **真值由生成器计算，绝不手打。** 任何人工标注的真值都可能错，而真值错了
     整个校准测量就废了 —— 这是校准类 benchmark 最容易翻车的地方。
  2. **F2 家族（合成题）保证抗污染**：随机数由 seed 生成，答案不可能出现在任何训练集里。
  3. **F1 家族（常识题）只做污染哨兵**：用于对照"模型在已知题 vs 新题上的元认知落差"，
     默认不计入主分数（verified=False）。

题族：
  F2a product    —— 估算 a×b（大数乘法），难度由位数控制
  F2b charcount  —— 数随机文本中目标字符出现次数
  F2c seqstat    —— 随机序列的统计量（均值/中位数）
  F1  almanac    —— 真实世界常识量（污染哨兵）
"""

import json
import random
import string

# ---------------------------------------------------------------- F2 合成题


def gen_product(rng, tier):
    """估算两个大整数的乘积。难度 = 位数。"""
    digits = {1: (2, 2), 2: (3, 3), 3: (4, 4), 4: (5, 5)}[tier]
    a = rng.randint(10 ** (digits[0] - 1), 10 ** digits[0] - 1)
    b = rng.randint(10 ** (digits[1] - 1), 10 ** digits[1] - 1)
    return {
        "family": "F2a_product",
        "tier": tier,
        "prompt": "请估算 %d × %d 的值。先给出你的点估计，再给出一个你有 90%% 把握覆盖真值的区间。"
        % (a, b),
        "truth": a * b,
        "unit": "",
        "display": "%d × %d" % (a, b),
    }


def gen_charcount(rng, tier):
    """数一段随机文本里目标字符出现的次数。难度 = 文本长度。"""
    length = {1: 200, 2: 600, 3: 1200, 4: 2400}[tier]
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    target = rng.choice(alphabet)
    text = "".join(rng.choice(alphabet) for _ in range(length))
    return {
        "family": "F2b_charcount",
        "tier": tier,
        "prompt": (
            "下面这段随机字母串中，字母 '%s' 一共出现了多少次？\n"
            "请先给出点估计，再给出一个你有 90%% 把握覆盖真值的区间。\n"
            "（文本长度 %d 字符，你不需要精确逐个数，可以估算。）\n\n%s"
        ) % (target, length, text[:120] + " ...[截断]... " + text[-60:]),
        "truth": text.count(target),
        "unit": "次",
        "display": "字母 '%s' 出现次数（长度 %d）" % (target, length),
        "_text": text,
    }


def gen_seqstat(rng, tier):
    """随机序列的均值。难度 = 序列长度与方差。"""
    n = {1: 12, 2: 40, 3: 120, 4: 400}[tier]
    lo, hi = 1, 1000
    seq = [rng.randint(lo, hi) for _ in range(n)]
    return {
        "family": "F2c_seqstat",
        "tier": tier,
        "prompt": (
            "有一组共 %d 个取自 1–1000 均匀分布的随机整数，其前 5 个是：%s。\n"
            "请估算这 %d 个数的平均值。先给出点估计，再给出一个你有 90%% 把握覆盖真值的区间。"
        ) % (n, seq[:5], n),
        "truth": sum(seq) / n,
        "unit": "",
        "display": "%d 个随机整数的均值" % n,
    }


F2_GENERATORS = [gen_product, gen_charcount, gen_seqstat]

# ---------------------------------------------------------------- F1 污染哨兵

# 只用于「已知题 vs 新题」的对照，默认不计入主分数。
# 真值为公认值；正式使用前需二次核验（代码里标记 verified=False 以强制提醒）。
ALMANAC = [
    ("光在真空中的速度（m/s）", 299792458, False),
    ("珠穆朗玛峰的最新官方高度（m）", 8848.86, False),
    ("地球赤道半径（km）", 6378.137, False),
    ("标准大气压下水的沸点（℃）", 100.0, False),
    ("一个标准大气压（Pa）", 101325.0, False),
    ("绝对零度（℃）", -273.15, False),
    ("成年人体内骨骼的块数", 206, False),
    ("国际象棋棋盘上的格子总数", 64, False),
]


def gen_almanac(rng, tier):
    q, truth, verified = ALMANAC[rng.randrange(len(ALMANAC))]
    return {
        "family": "F1_almanac",
        "tier": tier,
        "prompt": "问题：%s 是多少？请先给出点估计，再给出一个你有 90%% 把握覆盖真值的区间。" % q,
        "truth": truth,
        "unit": "",
        "display": q,
        "verified": verified,
    }


# ---------------------------------------------------------------- 数据集生成


def generate_dataset(n=60, seed=20261004, tiers=(1, 2, 3, 4), include_sentinels=True):
    """生成一个完整题集。同 seed 必得同题集（可复现）。"""
    rng = random.Random(seed)
    items = []
    idx = 0
    per_tier = max(1, n // len(tiers))
    for tier in tiers:
        for _ in range(per_tier):
            gen = F2_GENERATORS[idx % len(F2_GENERATORS)]
            item = gen(rng, tier)
            item["id"] = "Q%03d" % (len(items) + 1)
            item["contamination"] = "fresh"      # 程序化生成，抗污染
            item["verified"] = True               # 真值由生成器计算
            items.append(item)
            idx += 1
    if include_sentinels:
        for i in range(max(4, n // 10)):
            item = gen_almanac(rng, 1)
            item["id"] = "S%03d" % (i + 1)
            item["contamination"] = "known"       # 极可能出现在训练语料中
            item["verified"] = False
            items.append(item)
    return items


def export_dataset(items, path, with_answers=False):
    """导出题集。默认不含真值（用于发给被测方），--with-answers 才导出真值。"""
    rows = []
    for it in items:
        row = {k: v for k, v in it.items() if not k.startswith("_")}
        if not with_answers:
            row.pop("truth", None)
        rows.append(row)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    return path


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ds = generate_dataset(n=12, seed=1)
    for it in ds:
        print("%-6s %-14s tier=%d  truth=%s  %s"
              % (it["id"], it["family"], it["tier"], it["truth"], it["display"][:40]))
