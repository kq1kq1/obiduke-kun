"""検証セットを凍結する（再学習で精度が下がるのを防ぐための土台）。

なぜ必要か:
  Roboflowは新しいバージョンを作るたびに train/valid/test の分け方を切り直す。
  そのままだと「v3のモデルはv3のvalidで、v4のモデルはv4のvalidで」測ることになり、
  物差しが毎回変わって新旧を比較できない。「良くなった気がする」でモデルを差し替えるのが
  精度を下げる一番のルートなので、ここで検証用の画像を固定して二度と動かさない。

何を作るか:
  datasets/frozen_val/
    images/                 検証画像（Roboflowのvalid+test＝現モデルが学習に使っていない分）
    labels/                 元の規約のラベル（band＝帯にぴったり）
    labels_fullwidth/       案Bの規約のラベル（band＝全幅）
    data.yaml / data_fullwidth.yaml
  eval/frozen_val_manifest.json   どの画像を使うかの定義（Git管理下・これが正）

ラベルを2種類持つ理由:
  今の best.pt は「帯にぴったり」で学習されていて、これから作るモデルは「全幅」で学習する。
  同じ物差しで比べるには両方の規約で測れる必要がある。

使い方:
    python tools/freeze_val_set.py datasets/roboflow/v3
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

BAND_CLASS_ID = 0
NAMES = ["band", "logo", "map"]
OUT_DIR = Path("datasets/frozen_val")
MANIFEST = Path("eval/frozen_val_manifest.json")
# 現モデルが学習に使っていない分だけを凍結する。trainを混ぜると基準スコアが甘く出る。
SPLITS = ["valid", "test"]


def to_fullwidth(line):
    """band行を全幅(x_center=0.5, width=1.0)に書き換える。y方向とlogo/mapは触らない。"""
    p = line.split()
    if len(p) != 5:
        return line
    try:
        cid = int(float(p[0])); xc, yc, w, h = (float(v) for v in p[1:])
    except ValueError:
        return line
    if cid != BAND_CLASS_ID:
        return line
    return "%d %.6f %.6f %.6f %.6f" % (cid, 0.5, yc, 1.0, h)


def main():
    ap = argparse.ArgumentParser(description="検証セットを凍結する")
    ap.add_argument("root", help="RoboflowのYOLOv8エクスポートを展開したフォルダ")
    ap.add_argument("--force", action="store_true",
                    help="既に凍結済みでも作り直す（原則やらない。物差しが変わる）")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"エラー: {root} が見つかりません", file=sys.stderr); return 1

    if MANIFEST.exists() and not args.force:
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        print(f"すでに凍結済みです（{m['count']}枚 / 作成 {m['source']}）。")
        print("作り直すと過去のスコアと比較できなくなります。どうしても必要なら --force。")
        return 0

    for sub in ("images", "labels", "labels_fullwidth"):
        d = OUT_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    entries = []
    for split in SPLITS:
        img_dir, lbl_dir = root / split / "images", root / split / "labels"
        if not img_dir.is_dir():
            print(f"[warn] {img_dir} がありません。飛ばします")
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                print(f"[warn] ラベルが無い: {img.name}。飛ばします")
                continue
            data = img.read_bytes()
            shutil.copyfile(img, OUT_DIR / "images" / img.name)
            orig = [l.strip() for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]
            (OUT_DIR / "labels" / lbl.name).write_text(
                ("\n".join(orig) + "\n") if orig else "", encoding="utf-8")
            (OUT_DIR / "labels_fullwidth" / lbl.name).write_text(
                ("\n".join(to_fullwidth(l) for l in orig) + "\n") if orig else "", encoding="utf-8")
            entries.append({
                "image": img.name,
                "split_origin": split,
                "sha256": hashlib.sha256(data).hexdigest(),
                "boxes": len(orig),
            })

    if not entries:
        print("エラー: 1枚も集まりませんでした", file=sys.stderr); return 1

    # ultralytics が読む data.yaml。labels/ を見に行くので2種類作る
    for name, lbl_sub in (("data.yaml", "labels"), ("data_fullwidth.yaml", "labels_fullwidth")):
        (OUT_DIR / name).write_text(
            f"# 凍結検証セット（{lbl_sub}の規約）。中身は絶対に変えないこと。\n"
            f"path: {OUT_DIR.resolve().as_posix()}\n"
            f"train: images\nval: images\n\n"
            f"nc: {len(NAMES)}\nnames: {NAMES}\n", encoding="utf-8")
    # ultralyticsは images/ の隣の labels/ を自動で探すため、全幅版は別フォルダに複製して渡す
    fw = OUT_DIR / "fullwidth"
    if fw.exists():
        shutil.rmtree(fw)
    (fw / "images").mkdir(parents=True)
    shutil.copytree(OUT_DIR / "labels_fullwidth", fw / "labels")
    for p in (OUT_DIR / "images").iterdir():
        shutil.copyfile(p, fw / "images" / p.name)
    (OUT_DIR / "data_fullwidth.yaml").write_text(
        "# 凍結検証セット（案B＝band全幅の規約）。中身は絶対に変えないこと。\n"
        f"path: {fw.resolve().as_posix()}\n"
        "train: images\nval: images\n\n"
        f"nc: {len(NAMES)}\nnames: {NAMES}\n", encoding="utf-8")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "source": str(root).replace("\\", "/"),
        "splits_used": SPLITS,
        "count": len(entries),
        "note": "このリストが検証セットの定義。学習には絶対に使わないこと。",
        "images": entries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    total_boxes = sum(e["boxes"] for e in entries)
    print(f"凍結しました: {len(entries)}枚 / 枠 {total_boxes}個")
    print(f"  画像とラベル: {OUT_DIR}/")
    print(f"  定義ファイル: {MANIFEST}  ← これをGitで管理する")
    return 0


if __name__ == "__main__":
    sys.exit(main())
