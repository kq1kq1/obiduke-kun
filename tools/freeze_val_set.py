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


def name_key(filename):
    """Roboflowが付けるハッシュを除いた識別子。

    Roboflowは版を作り直すと画像を処理し直すのでファイル名のハッシュが変わるが、
    この部分は変わらない。前処理を変えて再エクスポートしても「同じ画像」を追える。
      例: 5-30-2-pdf_page_1_png.rf.4943....jpg  →  5-30-2-pdf_page_1_png
    """
    return Path(filename).stem.split(".rf.")[0]


def find_by_key(root, keys):
    """新しいエクスポートの全split から、指定した識別子の画像を探す。

    split(train/valid/test)がどこに変わっていても拾えるようにする。
    版を作り直したときにRoboflowが割り当てを変えても、凍結セットの中身を保てる。
    """
    found = {}
    for split in ("train", "valid", "test"):
        img_dir = root / split / "images"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            k = name_key(img.name)
            if k in keys and k not in found:
                lbl = root / split / "labels" / (img.stem + ".txt")
                if lbl.exists():
                    found[k] = (img, lbl, split)
    return found


def write_label_pair(name, lines):
    """ラベルを2種類の規約で書き出す（元のまま／bandを全幅に揃えたもの）。"""
    (OUT_DIR / "labels" / name).write_text(
        ("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    (OUT_DIR / "labels_fullwidth" / name).write_text(
        ("\n".join(to_fullwidth(l) for l in lines) + "\n") if lines else "", encoding="utf-8")


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
    ap.add_argument("--refresh", action="store_true",
                    help="今と同じ画像のまま、新しいエクスポートから作り直す"
                         "（前処理を変えて再エクスポートしたとき用。物差しの中身は変わらない）")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"エラー: {root} が見つかりません", file=sys.stderr); return 1

    # --refresh: 今の29枚と同じ画像を、新しいエクスポートから取り直す。
    # 版を作り直すとRoboflowはsplitを割り当て直すことがあるので、splitではなく
    # ファイル名の識別子で探す。1枚でも見つからなければ何もせず中止する。
    if args.refresh:
        if not MANIFEST.exists():
            print("エラー: まだ凍結していません。--refresh ではなく通常実行してください。",
                  file=sys.stderr)
            return 1
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        keys = {name_key(e["image"]) for e in old["images"]}
        found = find_by_key(root, keys)
        missing = sorted(keys - set(found))
        if missing:
            print(f"エラー: {len(missing)}枚が新しいエクスポートに見つかりません。中止します。",
                  file=sys.stderr)
            for k in missing[:5]:
                print(f"  - {k}", file=sys.stderr)
            print("       Roboflowから画像を消していないか確認してください。", file=sys.stderr)
            return 1

        for sub in ("images", "labels", "labels_fullwidth"):
            d = OUT_DIR / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

        entries, split_moved = [], 0
        for k in sorted(keys):
            img, lbl, split = found[k]
            if split not in SPLITS:
                split_moved += 1
            shutil.copyfile(img, OUT_DIR / "images" / img.name)
            orig = [l.strip() for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]
            write_label_pair(lbl.name, orig)
            entries.append({"image": img.name, "split_origin": split,
                            "sha256": hashlib.sha256(img.read_bytes()).hexdigest(),
                            "boxes": len(orig)})
        _write_outputs(entries, root, note_refresh=old.get("source"))
        print(f"作り直しました: {len(entries)}枚（中身は同じ画像のまま）")
        if split_moved:
            print(f"  ※ {split_moved}枚は新しい版でtrainに割り当てられていました。"
                  f"検証用なので学習からは自動で除外されます。")
        print(f"  取得元: {root}")
        print("  基準スコアを測り直してください:")
        print("    python tools/eval_model.py best.pt --save eval/baseline_best_pt.json")
        return 0

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

    _write_outputs(entries, root)
    total_boxes = sum(e["boxes"] for e in entries)
    print(f"凍結しました: {len(entries)}枚 / 枠 {total_boxes}個")
    print(f"  画像とラベル: {OUT_DIR}/")
    print(f"  定義ファイル: {MANIFEST}  ← これをGitで管理する")
    return 0


def _write_outputs(entries, root, note_refresh=None):
    """data.yaml と定義ファイルを書き出す（通常実行と --refresh の共通処理）。"""
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
    payload = {
        "source": str(root).replace("\\", "/"),
        "splits_used": SPLITS,
        "count": len(entries),
        "note": "このリストが検証セットの定義。学習には絶対に使わないこと。",
        "images": entries,
    }
    if note_refresh:
        payload["refreshed_from"] = note_refresh
        payload["note"] += " 前処理を変えたエクスポートから取り直したが、画像の顔ぶれは同じ。"
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
