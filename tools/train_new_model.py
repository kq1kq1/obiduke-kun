"""貯まったデータで再学習し、良くなっていなければ採用しない。

Colab（無料GPU）で回す想定。HF_TOKEN さえあれば学習データは全部Hubからそろう。

流れ:
  1. Hubから土台データ(base/)と蓄積データ(data/)を落とす
  2. 学習用フォルダを組む   学習 = 土台の学習分 + 蓄積データ / 検証 = 凍結セットだけ
  3. 検証画像が学習に混ざっていないか確認する（混ざると基準スコアが甘く出て比較にならない）
  4. 今の best.pt から続きを学習する（ゼロからより速く、覚えたことも残る）
  5. 凍結検証セットで採点し、基準スコアと比べる
  6. 良くなっていれば採用、悪くなっていれば採用しない（ここが一番大事）

「毎回、蓄積した全データで学習し直す」のが前提。増分だけで学習すると、
以前できていたことが劣化する（忘却）。

使い方（Colab）:
    !git clone https://github.com/kq1kq1/obiduke-kun && cd obiduke-kun
    !pip install -q ultralytics huggingface_hub
    import os; os.environ['HF_TOKEN'] = 'hf_xxx'
    !python tools/train_new_model.py kq1kq1/obiduke-training-data --epochs 100

使い方（手元にGPUがあるなら）:
    $env:HF_TOKEN = "hf_xxx"
    python tools/train_new_model.py kq1kq1/obiduke-training-data
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_model import APP_CONF, IMGSZ, NAMES, evaluate, summarize  # noqa: E402
from fetch_training_data import load_records  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE = REPO_ROOT / "eval" / "baseline_best_pt.json"
# 帯の見逃しは赤い「未検出」になって手作業が増えるので、少しの悪化も許さない。
# 誤検出は人が消せばいいので、こちらは多少の増減を許す。
RECALL_TOLERANCE = 0.0


def download(repo_id, token, workdir):
    from huggingface_hub import snapshot_download
    print(f"Hubから取得中: {repo_id}")
    local = Path(snapshot_download(repo_id=repo_id, repo_type="dataset", token=token))
    print(f"  {local}")
    return local


def build_dataset(hub_dir, out_dir):
    """学習用のフォルダを組む。返り値: (学習枚数, 検証枚数, 蓄積から入った枚数)"""
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = out_dir / split / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    # --- 土台の学習分 ---
    n_base = 0
    base_train = hub_dir / "base" / "train"
    if (base_train / "images").is_dir():
        for img in sorted((base_train / "images").iterdir()):
            lbl = base_train / "labels" / (img.stem + ".txt")
            if not lbl.exists():
                continue
            shutil.copyfile(img, out_dir / "train" / "images" / img.name)
            shutil.copyfile(lbl, out_dir / "train" / "labels" / lbl.name)
            n_base += 1
    else:
        print("[warn] base/train がありません。先に tools/upload_base_dataset.py を実行してください")

    # --- 凍結検証セット ---
    n_val = 0
    base_val = hub_dir / "base" / "frozen_val"
    if (base_val / "images").is_dir():
        for img in sorted((base_val / "images").iterdir()):
            lbl = base_val / "labels" / (img.stem + ".txt")
            if not lbl.exists():
                continue
            shutil.copyfile(img, out_dir / "val" / "images" / img.name)
            shutil.copyfile(lbl, out_dir / "val" / "labels" / lbl.name)
            n_val += 1

    # --- 蓄積データ（人が確認・修正したページ）---
    n_col = 0
    data_dir = hub_dir / "data"
    if (data_dir / "records").is_dir():
        latest, ok_lines, bad = load_records(data_dir)
        print(f"  蓄積データ: {ok_lines}行 → {len(latest)}ページ" +
              (f"（壊れた行 {bad} を除外）" if bad else ""))
        for rel, rec in sorted(latest.items()):
            src = data_dir / rel
            if not src.is_file():
                continue
            stem = Path(rel).stem
            shutil.copyfile(src, out_dir / "train" / "images" / f"{stem}.jpg")
            label = rec.get("label", "") or ""
            (out_dir / "train" / "labels" / f"{stem}.txt").write_text(
                (label + "\n") if label else "", encoding="utf-8")
            n_col += 1

    # --- 検証画像が学習に混ざっていないか ---
    train_names = {p.stem for p in (out_dir / "train" / "images").iterdir()}
    val_names = {p.stem for p in (out_dir / "val" / "images").iterdir()}
    leak = train_names & val_names
    if leak:
        print(f"エラー: 検証用の画像が学習にも入っています（{len(leak)}件）: "
              f"{sorted(leak)[:3]}", file=sys.stderr)
        print("       このまま学習するとスコアが甘く出て、比較になりません。", file=sys.stderr)
        raise SystemExit(1)

    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: train/images\nval: val/images\n\n"
        f"nc: {len(NAMES)}\nnames: {NAMES}\n", encoding="utf-8")
    return n_base + n_col, n_val, n_col


def judge(base, new):
    """基準スコアと比べて採用可否を返す。返り値: (判定, 理由のリスト)"""
    bb = {r["cls"]: r for r in base["per_class"]}
    nn = {r["cls"]: r for r in new["per_class"]}
    reasons, blocking = [], False

    d_rec = nn["band"]["recall"] - bb["band"]["recall"]
    if d_rec < -RECALL_TOLERANCE:
        blocking = True
        reasons.append(f"帯の再現率が下がった（{bb['band']['recall']:.3f} → "
                       f"{nn['band']['recall']:.3f}）。見逃しが増えるので採用しない")
    else:
        reasons.append(f"帯の再現率: {bb['band']['recall']:.3f} → {nn['band']['recall']:.3f}"
                       f"（{d_rec:+.3f}）")

    d_missing = new["pages_missing_band"] - base["pages_missing_band"]
    if d_missing > 0:
        blocking = True
        reasons.append(f"帯を出せないページが増えた（{base['pages_missing_band']} → "
                       f"{new['pages_missing_band']}）。赤い未検出が増えるので採用しない")
    else:
        reasons.append(f"帯を出せないページ: {base['pages_missing_band']} → "
                       f"{new['pages_missing_band']}（{d_missing:+d}）")

    d_pre = nn["band"]["precision"] - bb["band"]["precision"]
    reasons.append(f"帯の適合率: {bb['band']['precision']:.3f} → "
                   f"{nn['band']['precision']:.3f}（{d_pre:+.3f}）")
    for c in ("logo", "map"):
        reasons.append(f"{c}のF1: {bb[c]['f1']:.3f} → {nn[c]['f1']:.3f}"
                       f"（{nn[c]['f1'] - bb[c]['f1']:+.3f}）")

    if blocking:
        return "採用しない", reasons
    if d_rec > 0 or d_pre > 0 or d_missing < 0:
        return "採用してよい", reasons
    return "判断が必要", reasons


def main():
    ap = argparse.ArgumentParser(description="貯まったデータで再学習し、良くなったかを判定する")
    ap.add_argument("repo_id", help="例: kq1kq1/obiduke-training-data")
    ap.add_argument("--base-model", default="best.pt", help="続きから学習する重み")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--work", default="training_run", help="作業フォルダ")
    ap.add_argument("--device", default=None, help="'0'でGPU、'cpu'でCPU（既定は自動）")
    ap.add_argument("--skip-train", action="store_true",
                    help="学習せず、既にある結果の採点だけやり直す")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print('エラー: 環境変数 HF_TOKEN を設定してください。', file=sys.stderr)
        return 1
    if not BASELINE.exists():
        print(f"エラー: 基準スコア {BASELINE} がありません。\n"
              f"       先に python tools/eval_model.py best.pt --save {BASELINE}", file=sys.stderr)
        return 1
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    ds = work / "dataset"
    weights = work / "train" / "weights" / "best.pt"

    if not args.skip_train:
        hub = download(args.repo_id, token, work)
        n_train, n_val, n_col = build_dataset(hub, ds)
        print(f"\n学習 {n_train}枚（うち蓄積データ {n_col}枚） / 検証 {n_val}枚（凍結セット）")
        if n_val == 0:
            print("エラー: 検証セットが空です。判定できないので中止します。", file=sys.stderr)
            return 1

        from ultralytics import YOLO
        print(f"\n学習開始: {args.base_model} から {args.epochs}エポック")
        model = YOLO(args.base_model)
        model.train(data=str((ds / "data.yaml").resolve()), epochs=args.epochs,
                    imgsz=args.imgsz, project=str(work), name="train",
                    exist_ok=True, device=args.device)

    if not weights.exists():
        print(f"エラー: 学習結果が見つかりません: {weights}", file=sys.stderr)
        return 1

    # ---- 凍結検証セットで採点（アプリと同条件） ----
    print("\n" + "=" * 64)
    print("凍結検証セットで採点（アプリと同条件 conf=%.2f imgsz=%d）" % (APP_CONF, IMGSZ))
    print("=" * 64)
    from ultralytics import YOLO
    imgs = sorted((ds / "val" / "images").iterdir())
    stats, missing = evaluate(YOLO(str(weights)), imgs, ds / "val" / "labels", APP_CONF)
    new = summarize(stats, len(imgs), missing)

    print(f"\n{'クラス':<8}{'再現率':>10}{'適合率':>10}{'F1':>10}")
    print("-" * 40)
    bb = {r["cls"]: r for r in baseline["per_class"]}
    for r in new["per_class"]:
        b = bb[r["cls"]]
        print(f"{r['cls']:<8}{b['recall']:>5.3f}→{r['recall']:<4.3f}"
              f"{b['precision']:>5.3f}→{r['precision']:<4.3f}"
              f"{b['f1']:>5.3f}→{r['f1']:<4.3f}")

    decision, reasons = judge(baseline, new)
    print("\n判定の内訳:")
    for x in reasons:
        print(f"  - {x}")

    out = work / "candidate_score.json"
    new["model"] = str(weights)
    out.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"  判定: {decision}")
    print("=" * 64)
    if decision == "採用しない":
        print("\nこのモデルは使わないこと。データを増やしてからやり直す。")
        print(f"（スコアは {out} に残してある）")
        return 2

    print(f"\n新しい重み: {weights}")
    print("\n採用する手順:")
    print(f"  1. {weights} をプロジェクト直下の best.pt に上書き")
    print('  2. python -c "from ultralytics import YOLO; '
          "YOLO('best.pt').export(format='openvino', imgsz=640)\"")
    print("  3. python tools/eval_model.py best.pt   ← 上書き後にもう一度確認")
    print(f"  4. python tools/eval_model.py best.pt --save {BASELINE}   ← 基準を更新")
    print("  5. git commit して .\\redeploy_hf.ps1")
    if decision == "判断が必要":
        print("\n※ 明確に良くなってはいない。検証セットは29枚と小さいので、"
              "差が小さいときは見送るのが安全。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
