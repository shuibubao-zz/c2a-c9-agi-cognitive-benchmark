#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harness.py — ΔE-Bench 命令行入口

子命令：
    gen     生成题集（导出含真值版 + 发给被测方的无真值版）
    demo    跑模拟被试验证，证明评分内核有辨别力
    score   对一份作答文件（JSONL/JSON）评分
    selftest 断言核心指标行为正确（任何人拿到都能验证没坏）

用法：
    python harness.py gen --n 60 --out dataset/
    python harness.py demo
    python harness.py score --items dataset/items_with_truth.json --responses pilot/xxx.json
    python harness.py selftest
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from items import generate_dataset, export_dataset          # noqa: E402
from scoring import (                                        # noqa: E402
    score_interval_responses,
    contamination_split,
    score_decisions,
    deltaE_convergence,
    winkler_score,
)
from simulate import run_validation, ARCHETYPES             # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------- gen


def cmd_gen(args):
    items = generate_dataset(n=args.n, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    with_truth = os.path.join(args.out, "items_with_truth.json")
    public = os.path.join(args.out, "items_public.json")
    export_dataset(items, with_truth, with_answers=True)
    export_dataset(items, public, with_answers=False)
    print("题集已生成：%d 题（seed=%d）" % (len(items), args.seed))
    print("  含真值（评分用，不要发给被测方）：%s" % with_truth)
    print("  无真值（发给被测方）          ：%s" % public)
    fams = {}
    for it in items:
        fams[it["family"]] = fams.get(it["family"], 0) + 1
    for f, c in sorted(fams.items()):
        print("    %-16s %d 题" % (f, c))
    return 0


# ---------------------------------------------------------------- demo


def cmd_demo(args):
    items = generate_dataset(n=args.n, seed=args.seed)
    print("=" * 72)
    print("ΔE-Bench 评分内核验证 · 模拟被试")
    print("=" * 72)
    print("题集：%d 题（seed=%d），声称覆盖率 90%%\n" % (len(items), args.seed))
    res = run_validation(items, seed=7)

    print("\n验证命题：**只盯覆盖率会被对冲者骗过，Winkler 不会**")
    cal = res["calibrated"]
    hed = res["hedger"]
    print("  · hedger 的覆盖率缺口 = %+.3f（比 calibrated 的 %+.3f 更「保守」，看似更好）"
          % (hed["coverage_gap"], cal["coverage_gap"]))
    print("  · 但 hedger 的相对 Winkler = %.3f，远差于 calibrated 的 %.3f"
          % (hed["mean_rel_winkler"], cal["mean_rel_winkler"]))
    ok = hed["mean_rel_winkler"] > cal["mean_rel_winkler"]
    print("  → 结论：%s" % ("✅ 成立：Winkler 识破了套牢式对冲" if ok else "❌ 不成立，指标需要重设计"))
    return 0 if ok else 1


# ---------------------------------------------------------------- score


def load_any(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read().strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def cmd_score(args):
    items = {it["id"]: it for it in load_any(args.items)}
    resp = load_any(args.responses)
    records = []
    unknown = []
    for r in resp:
        iid = r.get("id")
        if iid not in items:
            unknown.append(iid)
            continue
        it = items[iid]
        if "low" not in r or "high" not in r:
            continue
        records.append(
            {
                "id": iid,
                "family": it.get("family"),
                "tier": it.get("tier"),
                "contamination": it.get("contamination", "fresh"),
                "point": r.get("point"),
                "low": r["low"],
                "high": r["high"],
                "truth": it["truth"],
            }
        )

    if unknown:
        print("⚠️ 题集中找不到这些 id，已跳过：%s" % ", ".join(map(str, unknown[:10])))
    if not records:
        print("没有可评分的记录。")
        return 2

    print("=" * 72)
    print("ΔE-Bench 评分报告")
    print("=" * 72)
    print("作答方：%s ｜ 有效记录 %d 条 ｜ 声称覆盖率 %.0f%%\n"
          % (args.name, len(records), args.coverage * 100))

    m = score_interval_responses(records, claimed_coverage=args.coverage)
    print("【Module A · 预测性元认知】")
    print("  命中率               : %.3f" % m["hit_rate"])
    print("  覆盖率缺口 (>0过度自信): %+.3f" % m["coverage_gap"])
    print("  相对 Winkler 得分↓   : %.3f" % m["mean_rel_winkler"])
    print("  锐度（相对半宽）     : %.3f" % m["mean_rel_half_width"])
    print("  辨别力 ρ（宽度vs误差）: %+.3f" % m["metacog_sensitivity"])
    if m["mean_rel_abs_error"] == m["mean_rel_abs_error"]:
        print("  平均相对绝对误差     : %.3f" % m["mean_rel_abs_error"])

    split = contamination_split(records, claimed_coverage=args.coverage)
    if split["fresh"] and split["known"]:
        print("\n【污染对照 · F1已知题 vs F2新题】")
        print("  新题(fresh) 覆盖率缺口 : %+.3f  (n=%d)"
              % (split["fresh"]["coverage_gap"], split["fresh"]["n"]))
        print("  已知题(known) 覆盖率缺口: %+.3f  (n=%d)"
              % (split["known"]["coverage_gap"], split["known"]["n"]))
        print("  落差 gap (>0=新题上更过度自信): %+.3f" % split["gap"])

    if args.decisions and os.path.exists(args.decisions):
        d = score_decisions(load_any(args.decisions))
        print("\n【Module B · 元认知控制】")
        print("  实际得分        : %.1f" % d["actual_total"])
        print("  总是作答基线    : %.1f" % d["always_answer_total"])
        print("  全知最优(oracle): %.1f" % d["oracle_total"])
        print("  净收益/题       : %+.3f" % d["net_gain_vs_always"])
        print("  相对最优效率    : %.3f" % d["efficiency_vs_oracle"])

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"module_a": m, "contamination": split}, fh, ensure_ascii=False, indent=2)
        print("\n已写入 %s" % args.out)
    return 0


# ---------------------------------------------------------------- selftest


def cmd_selftest(args):
    passed = failed = 0

    def case(name, fn):
        nonlocal passed, failed
        try:
            fn()
            print("  ✅ PASS  %s" % name)
            passed += 1
        except AssertionError as e:
            print("  ❌ FAIL  %s → %s" % (name, e))
            failed += 1

    print("ΔE-Bench 自检\n")

    def t_winkler_inside():
        # 盖住真值：只罚宽度
        assert abs(winkler_score(90, 110, 100, 0.1) - 20.0) < 1e-9

    def t_winkler_below():
        # 真值在下界外 10，罚 2/0.1*10 = 200，加宽度 20
        assert abs(winkler_score(90, 110, 80, 0.1) - 220.0) < 1e-9

    def t_winkler_above():
        assert abs(winkler_score(90, 110, 120, 0.1) - 220.0) < 1e-9

    def t_hedger_penalised():
        # 对冲者区间极宽：一定盖住，但 Winkler 必然比校准者差
        narrow = winkler_score(95, 105, 100, 0.1)
        wide = winkler_score(-1000, 1000, 100, 0.1)
        assert wide > narrow, "超宽区间必须被罚得更重"

    def t_archetypes_separate():
        res = run_validation(seed=7, verbose=False)
        cal, hed, over = res["calibrated"], res["hedger"], res["overconfident"]
        assert hed["mean_rel_winkler"] > cal["mean_rel_winkler"], "对冲者 Winkler 必须差于校准者"
        assert over["coverage_gap"] > cal["coverage_gap"], "过度自信者缺口必须更大"
        assert abs(cal["coverage_gap"]) < 0.12, "校准者缺口应接近 0，实际 %+.3f" % cal["coverage_gap"]

    def t_uniform_no_sensitivity():
        res = run_validation(seed=7, verbose=False)
        u = res["uniform_width"]["metacog_sensitivity"]
        assert abs(u) < 0.35, "宽度不随难度变化的原型，辨别力应接近 0，实际 %+.3f" % u

    def t_dataset_deterministic():
        a = generate_dataset(n=24, seed=42)
        b = generate_dataset(n=24, seed=42)
        assert [i["truth"] for i in a] == [i["truth"] for i in b], "同 seed 必须复现同题集"

    def t_ground_truth_computed():
        # 真值必须由生成器计算：抽样复核乘法题
        for it in generate_dataset(n=24, seed=3):
            if it["family"] == "F2a_product":
                a, b = it["display"].split(" × ")
                assert it["truth"] == int(a) * int(b), "乘法真值必须由计算得出"

    def t_public_has_no_truth():
        import tempfile

        d = tempfile.mkdtemp()
        p = os.path.join(d, "pub.json")
        export_dataset(generate_dataset(n=12, seed=1), p, with_answers=False)
        rows = load_any(p)
        assert all("truth" not in r for r in rows), "发给被测方的题集不能含真值"

    case("Winkler：真值落在区间内 → 只罚宽度", t_winkler_inside)
    case("Winkler：真值低于下界 → 加罚偏离量", t_winkler_below)
    case("Winkler：真值高于上界 → 加罚偏离量", t_winkler_above)
    case("对冲者被罚：超宽区间 Winkler 更差", t_hedger_penalised)
    case("原型可区分：校准/过度自信/对冲 三者排序正确", t_archetypes_separate)
    case("辨别力：宽度不随难度变化者 ρ≈0", t_uniform_no_sensitivity)
    case("可复现：同 seed 生成同题集", t_dataset_deterministic)
    case("真值由生成器计算（不手打）", t_ground_truth_computed)
    case("公开题集不含真值", t_public_has_no_truth)

    print("\n自检结果：%d 通过 / %d 失败" % (passed, failed))
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description="ΔE-Bench harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="生成题集")
    g.add_argument("--n", type=int, default=60)
    g.add_argument("--seed", type=int, default=20261004)
    g.add_argument("--out", default=os.path.join(HERE, "dataset"))
    g.set_defaults(func=cmd_gen)

    d = sub.add_parser("demo", help="模拟被试验证")
    d.add_argument("--n", type=int, default=60)
    d.add_argument("--seed", type=int, default=20261004)
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("score", help="对作答文件评分")
    s.add_argument("--items", required=True)
    s.add_argument("--responses", required=True)
    s.add_argument("--decisions", default=None)
    s.add_argument("--name", default="未命名被测方")
    s.add_argument("--coverage", type=float, default=0.90)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_score)

    t = sub.add_parser("selftest", help="自检评分内核")
    t.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
