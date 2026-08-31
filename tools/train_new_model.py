"""貯まったデータで再学習し、良くなっていなければ採用しない。

Colab（無料GPU）で回す想定。HF_TOKEN さえあれば学習データは全部Hubからそろう。

流れ:
  1. Hubから土台データ(base/)と蓄積データ(data/)を落とす
  2. 学習用フォルダを3つに組む
       train  … 重みを学習する分（土台 ＋ 蓄積データ）
       val    … ultralyticsがエポックごとに見る分。best.pt の選び方に使う
       frozen … 凍結検証セット。学習中は一切見せず、最終判定だけに使う
  3. 凍結検証の画像が学習側に混ざっていないか確認する（混ざるとスコアが甘く出る）
  4. 今の best.pt から続きを学習する（ゼロからより速く、覚えたことも残る）
  5. 凍結検証セットで採点し、基準スコアと比べる
  6. 良くなっていれば採用、悪くなっていれば採用しない（ここが一番大事）

なぜ内部検証(val)と凍結検証(frozen)を分けるか:
  ultralyticsは「検証成績がいちばん良かったエポック」を best.pt として選ぶ。
  そこに凍結検証セットを使うと「選ぶのに使ったもので採点する」ことになり、
  成績が実際より良く出てしまう。判定を信じられるようにするため分けている。

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


def build_dataset(hub_dir, out_dir, val_every=10):
    """学習用のフォルダを組む。3つに分ける。

      train/  … 重みを学習する分（土台 ＋ 蓄積データ）
      val/    … ultralyticsがエポックごとに見る分。**best.pt の選び方に使う**
      frozen/ … 凍結検証セット。学習中は一切見せず、**最終判定だけに使う**

    ultralyticsは「検証成績がいちばん良かったエポック」を best.pt として選ぶ。
    そこに凍結検証セットを使うと「選ぶのに使ったもので採点する」ことになり、
    成績が実際より良く出てしまう。だから内部検証は学習データから取り分ける。

    返り値: (学習枚数, 内部検証枚数, 凍結検証枚数, 蓄積から入った枚数)
    """
    for split in ("train", "val", "frozen"):
        for sub in ("images", "labels"):
            d = out_dir / split / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    # --- 学習に回す候補を集める（土台 ＋ 蓄積）---
    pool = []          # [(画像パス, ラベルの中身, 出力名)]
    n_col = 0

    base_train = hub_dir / "base" / "train"
    if (base_train / "images").is_dir():
        for img in sorted((base_train / "images").iterdir()):
            lbl = base_train / "labels" / (img.stem + ".txt")
            if lbl.exists():
                pool.append((img, lbl.read_text(encoding="utf-8"), img.name))
    else:
        print("[warn] base/train がありません。先に tools/upload_base_dataset.py を実行してください")

    data_dir = hub_dir / "data"
    if (data_dir / "records").is_dir():
        latest, ok_lines, bad = load_records(data_dir)
        print(f"  蓄積データ: {ok_lines}行 → {len(latest)}ページ"
              + (f"（壊れた行 {bad} を除外）" if bad else ""))
        for rel, rec in sorted(latest.items()):
            src = data_dir / rel
            if not src.is_file():
                continue
            label = rec.get("label", "") or ""
            pool.append((src, (label + "\n") if label else "", Path(rel).stem + ".jpg"))
            n_col += 1

    # --- 内部検証を等間隔に取り分ける（毎回同じ分け方になるよう決め打ち）---
    n_train = n_val = 0
    for i, (img, label_text, name) in enumerate(pool):
        split = "val" if (val_every and i % val_every == 0) else "train"
        shutil.copyfile(img, out_dir / split / "images" / name)
        (out_dir / split / "labels" / (Path(name).stem + ".txt")).write_text(
            label_text, encoding="utf-8")
        if split == "val":
            n_val += 1
        else:
            n_train += 1

    # --- 凍結検証セット（学習には一切使わない）---
    n_frozen = 0
    base_val = hub_dir / "base" / "frozen_val"
    if (base_val / "images").is_dir():
        for img in sorted((base_val / "images").iterdir()):
            lbl = base_val / "labels" / (img.stem + ".txt")
            if not lbl.exists():
                continue
            shutil.copyfile(img, out_dir / "frozen" / "images" / img.name)
            shutil.copyfile(lbl, out_dir / "frozen" / "labels" / lbl.name)
            n_frozen += 1

    # --- 凍結検証の画像が学習側に混ざっていないか ---
    seen = {p.stem for p in (out_dir / "train" / "images").iterdir()}
    seen |= {p.stem for p in (out_dir / "val" / "images").iterdir()}
    frozen_names = {p.stem for p in (out_dir / "frozen" / "images").iterdir()}
    leak = seen & frozen_names
    if leak:
        print(f"エラー: 凍結検証の画像が学習にも入っています（{len(leak)}件）: "
              f"{sorted(leak)[:3]}", file=sys.stderr)
        print("       このまま学習するとスコアが甘く出て、比較になりません。", file=sys.stderr)
        raise SystemExit(1)

    (out_dir / "data.yaml").write_text(
        f"# frozen/ は書かない。ultralyticsに見せると best.pt の選び方に混ざるため。\n"
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: train/images\nval: val/images\n\n"
        f"nc: {len(NAMES)}\nnames: {NAMES}\n", encoding="utf-8")
    return n_train, n_val, n_frozen, n_col


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
    ap.add_argument("--weights", default=None,
                    help="採点する重みを直接指定する"
                         "（学習は終わっているのに見つからないと言われたとき用）")
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

    # ultralyticsは project が相対パスだと runs/detect/ の下に置いてしまうので、
    # 必ず絶対パスにしてから渡す（相対のままだと保存先が想定とズレる）
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    ds = work / "dataset"
    weights = work / "train" / "weights" / "best.pt"

    if not args.skip_train:
        hub = download(args.repo_id, token, work)
        n_train, n_val, n_frozen, n_col = build_dataset(hub, ds)
        print(f"\n学習     {n_train}枚（うち蓄積データ {n_col}枚）")
        print(f"内部検証 {n_val}枚（best.pt の選び方に使う）")
        print(f"凍結検証 {n_frozen}枚（学習には見せない・最終判定だけに使う）")
        if n_frozen == 0:
            print("エラー: 凍結検証セットが空です。判定できないので中止します。", file=sys.stderr)
            return 1
        if n_train == 0:
            print("エラー: 学習データが空です。中止します。", file=sys.stderr)
            return 1

        from ultralytics import YOLO
        print(f"\n学習開始: {args.base_model} から {args.epochs}エポック")
        model = YOLO(args.base_model)
        model.train(data=str((ds / "data.yaml").resolve()), epochs=args.epochs,
                    imgsz=args.imgsz, project=str(work), name="train",
                    exist_ok=True, device=args.device)
        # 実際の保存先をultralyticsから受け取る（推測しない）
        try:
            trained = Path(model.trainer.save_dir) / "weights" / "best.pt"
            if trained.exists():
                weights = trained
        except Exception:
            pass

    if args.weights:
        weights = Path(args.weights).resolve()
    if not weights.exists():
        # 保存先がズレていることがあるので探しに行く（学習は成功しているのに
        # 「見つかりません」で止まるのを避ける）
        found = sorted(Path.cwd().rglob("weights/best.pt"), key=lambda q: -q.stat().st_mtime)
        if found:
            weights = found[0]
            print(f"[info] 想定の場所に無かったので、いちばん新しいものを使います: {weights}")
        else:
            print(f"エラー: 学習結果が見つかりません: {weights}", file=sys.stderr)
            print("       --weights で直接指定してください。", file=sys.stderr)
            return 1

    # ---- 凍結検証セットで採点（アプリと同条件） ----
    print("\n" + "=" * 64)
    print("凍結検証セットで採点（学習に一切使っていない分・アプリと同条件 conf=%.2f imgsz=%d）"
          % (APP_CONF, IMGSZ))
    print("=" * 64)
    from ultralytics import YOLO
    imgs = sorted((ds / "frozen" / "images").iterdir())
    stats, missing = evaluate(YOLO(str(weights)), imgs, ds / "frozen" / "labels", APP_CONF)
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
