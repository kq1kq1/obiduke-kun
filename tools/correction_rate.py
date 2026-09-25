"""本番での「手直しが必要だったページの割合」を出す。

なぜこれを見るか:
  凍結検証セット29枚では band も logo も既に再現率1.000（天井）で、
  **これ以上良くなったことを測れない**。悪化の検出には使えるが、改善は測れない。
  一方この指標は「人が実際に手を入れた割合」なので、モデルが良くなれば必ず下がる。
  ラベル付けの手間もいらず、本番の実データそのもので測れる。

計算の注意:
  記録されるのは「修正した全ページ」と「直さなかったページのうち一部（抽選）」だけ。
  そのまま割ると修正が濃縮されて実態より高く出る（抽選率20%なら5倍に見える）。
  そこで抽選分を割り戻してから割合を出す。

      実際に処理したページ ≒ 修正したページ + 抽選で記録されたページ ÷ 抽選率
      修正率 = 修正したページ ÷ 実際に処理したページ

使い方:
    python tools/correction_rate.py                       # 落とし済みのデータを見る
    python tools/correction_rate.py --since 2026-09-25    # モデル差し替え後だけ見る
    python tools/correction_rate.py --weekly              # 週ごとの推移
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DEFAULT_DATA = Path("datasets/hub/data")


def load_all(records_dir):
    """記録を全部読む。同じページの再記録は「最後の1件」だけ採用する。"""
    latest = {}
    bad = 0
    for f in sorted(records_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            img = r.get("image")
            if not img:
                bad += 1
                continue
            prev = latest.get(img)
            if prev is None or r.get("ts", "") >= prev.get("ts", ""):
                latest[img] = r
    return latest, bad


def rate_of(rows, sample_rate):
    """(修正数, 抽選記録数, 推定処理ページ数, 修正率) を返す。"""
    edited = sum(1 for r in rows if r.get("edited"))
    sampled = len(rows) - edited
    # 抽選分を割り戻す。抽選率0なら割り戻せないので記録数をそのまま使う
    est_total = edited + (sampled / sample_rate if sample_rate > 0 else sampled)
    return edited, sampled, est_total, (edited / est_total if est_total else 0.0)


def main():
    ap = argparse.ArgumentParser(description="本番での手直し率を出す")
    ap.add_argument("--data", default=str(DEFAULT_DATA),
                    help="records/ を含むフォルダ（既定: datasets/hub/data）")
    ap.add_argument("--sample-rate", type=float, default=0.2,
                    help="アプリ側の抽選率。app.py の TRAINING_SAMPLE_RATE と揃える（既定 0.2）")
    ap.add_argument("--since", default=None, help="この日以降だけ見る（例 2026-09-25）")
    ap.add_argument("--until", default=None, help="この日より前だけ見る")
    ap.add_argument("--weekly", action="store_true", help="週ごとの推移を出す")
    args = ap.parse_args()

    data = Path(args.data)
    if not (data / "records").is_dir():
        print(f"エラー: {data}/records がありません。\n"
              f"       先に python tools/fetch_training_data.py <リポジトリ> を実行してください。",
              file=sys.stderr)
        return 1

    latest, bad = load_all(data / "records")
    if not latest:
        print("記録が0件でした。", file=sys.stderr)
        return 1

    rows = list(latest.values())
    if args.since:
        rows = [r for r in rows if r.get("ts", "") >= args.since]
    if args.until:
        rows = [r for r in rows if r.get("ts", "") < args.until]
    if not rows:
        print("その期間の記録はありません。", file=sys.stderr)
        return 1

    ts = sorted(r.get("ts", "") for r in rows if r.get("ts"))
    edited, sampled, est, rate = rate_of(rows, args.sample_rate)

    print(f"期間: {ts[0][:10]} 〜 {ts[-1][:10]}" if ts else "期間: 不明")
    print(f"抽選率: {args.sample_rate * 100:.0f}%（app.py の TRAINING_SAMPLE_RATE）")
    print()
    print(f"  修正したページ      : {edited:>6}  （全部記録される）")
    print(f"  抽選で記録したページ: {sampled:>6}  → 実際は約 {sampled / args.sample_rate:.0f} ページ")
    print(f"  実際に処理したページ: {est:>6.0f}  （推定）")
    print()
    print(f"  ★ 手直し率: {rate * 100:.2f}%   （{edited} / 約{est:.0f}）")
    print(f"     手を入れずに通ったページ: {(1 - rate) * 100:.2f}%")
    if bad:
        print(f"  （読めなかった記録 {bad} 件は除外）")

    if args.weekly:
        print()
        print("週ごとの推移:")
        buckets = defaultdict(list)
        for r in rows:
            t = r.get("ts", "")
            if not t:
                continue
            try:
                d = datetime.fromisoformat(t.replace("Z", "+00:00"))
            except ValueError:
                continue
            y, w, _ = d.isocalendar()
            buckets[(y, w)].append(r)
        print(f"  {'週':<12}{'修正':>6}{'推定処理':>10}{'手直し率':>11}")
        for key in sorted(buckets):
            e, s, t, rt = rate_of(buckets[key], args.sample_rate)
            bar = "█" * min(30, int(rt * 300))
            print(f"  {key[0]}-W{key[1]:<7}{e:>6}{t:>10.0f}{rt * 100:>10.2f}%  {bar}")

    print()
    print("使い方: モデルを差し替えたら、その日を --since に指定して測り直す。")
    print("        率が下がっていれば本当に良くなっている（凍結検証セットでは測れない部分）。")
    print()
    print("注意: この数字は人の丁寧さにも影響される。忙しくて直さなかった日は下がる。")
    print("      1週間ぶんだけで判断せず、数百ページ貯まってから比べること。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
