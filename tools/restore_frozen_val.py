"""凍結検証セットをHubから復元する（別のPCで作業を始めるとき用）。

`datasets/` はGit管理外なので、リポジトリをcloneしただけでは
`datasets/frozen_val/` が無く、`tools/eval_model.py` が動かない。

ただし中身は `tools/upload_base_dataset.py` で Hub の `base/frozen_val/` に
上げてあるので、そこから戻せる。Roboflowのエクスポート（118MB）を
持ち回る必要は無い。

`eval/frozen_val_manifest.json`（Git管理下）と突き合わせて、
**同じ29枚が揃っているか**を必ず確認する。1枚でも違えば過去のスコアと
比較できなくなるため、その場合は中止する。

使い方:
    hf auth login                # 初回のみ
    python tools/restore_frozen_val.py kq1kq1/obiduke-training-data
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

OUT_DIR = Path("datasets/frozen_val")
MANIFEST = Path("eval/frozen_val_manifest.json")
NAMES = ["band", "logo", "map"]

MSG_NO_TOKEN = """エラー: HFのアクセストークンが見つかりません。どちらかをしてください。

  A) 一度ログインしておく（以後ずっと不要になる・おすすめ）
       hf auth login

  B) この端末セッションだけ設定する
       $env:HF_TOKEN = Read-Host "HFトークンを貼ってEnter"

  トークンは https://huggingface.co/settings/tokens で発行。
  読み取りだけなので Fine-grained の Read で足りる。"""


def get_hf_token():
    """環境変数 → hf auth login で保存したもの、の順で探す。"""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if tok:
        return tok.strip()
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="凍結検証セットをHubから復元する")
    ap.add_argument("repo_id", help="例: kq1kq1/obiduke-training-data")
    ap.add_argument("--work", default="datasets/hub", help="ダウンロード先")
    ap.add_argument("--force", action="store_true", help="既にあっても上書きする")
    args = ap.parse_args()

    if OUT_DIR.is_dir() and not args.force:
        n = len(list((OUT_DIR / "images").glob("*"))) if (OUT_DIR / "images").is_dir() else 0
        print(f"すでにあります（{n}枚）。作り直すなら --force。")
        return 0
    if not MANIFEST.exists():
        print(f"エラー: {MANIFEST} がありません。リポジトリのルートで実行してください。",
              file=sys.stderr)
        return 1

    token = get_hf_token()
    if not token:
        print(MSG_NO_TOKEN, file=sys.stderr)
        return 1

    from huggingface_hub import snapshot_download
    dest = Path(args.work)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Hubから取得中: {args.repo_id}（base/frozen_val だけ）")
    kw = dict(repo_id=args.repo_id, repo_type="dataset", token=token,
              local_dir=str(dest), allow_patterns=["base/frozen_val/*"])
    try:
        local = Path(snapshot_download(**kw))
    except Exception as e:
        print(f"[warn] ダウンロードに失敗しました: {e}")
        print("       Xet転送を切ってやり直します...")
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        local = Path(snapshot_download(**kw))

    src = local / "base" / "frozen_val"
    if not (src / "images").is_dir():
        print(f"エラー: {src}/images が見つかりません。\n"
              f"       先に tools/upload_base_dataset.py を実行しておく必要があります。",
              file=sys.stderr)
        return 1

    # 定義ファイルと突き合わせる。顔ぶれが違ったら過去のスコアと比較できない
    want = {e["image"] for e in json.loads(MANIFEST.read_text(encoding="utf-8"))["images"]}
    got = {p.name for p in (src / "images").iterdir()
           if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    missing, extra = sorted(want - got), sorted(got - want)
    if missing or extra:
        print(f"エラー: 定義ファイルと中身が一致しません。中止します。", file=sys.stderr)
        if missing:
            print(f"  足りない {len(missing)}枚: {missing[:3]}", file=sys.stderr)
        if extra:
            print(f"  余分な {len(extra)}枚: {extra[:3]}", file=sys.stderr)
        return 1

    for sub in ("images", "labels", "labels_fullwidth"):
        d = OUT_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    total = 0
    for img in sorted((src / "images").iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = src / "labels" / (img.stem + ".txt")
        if not lbl.exists():
            continue
        shutil.copyfile(img, OUT_DIR / "images" / img.name)
        # Hubに上げてあるのは案B（band全幅）の規約のもの。両方に同じものを置く
        # （orig規約は現行モデルの学習に使っていないので、もう使い道が無い）
        text = lbl.read_text(encoding="utf-8")
        (OUT_DIR / "labels" / lbl.name).write_text(text, encoding="utf-8")
        (OUT_DIR / "labels_fullwidth" / lbl.name).write_text(text, encoding="utf-8")
        total += 1

    for name in ("data.yaml", "data_fullwidth.yaml"):
        (OUT_DIR / name).write_text(
            "# 凍結検証セット。中身は絶対に変えないこと。\n"
            f"path: {OUT_DIR.resolve().as_posix()}\n"
            "train: images\nval: images\n\n"
            f"nc: {len(NAMES)}\nnames: {NAMES}\n", encoding="utf-8")

    sha_ok = sum(1 for e in json.loads(MANIFEST.read_text(encoding="utf-8"))["images"]
                 if (OUT_DIR / "images" / e["image"]).exists()
                 and hashlib.sha256((OUT_DIR / "images" / e["image"]).read_bytes()).hexdigest()
                 == e["sha256"])

    print(f"復元しました: {total}枚 → {OUT_DIR}/")
    print(f"  定義ファイルとSHA256が一致: {sha_ok} / {total}")
    if sha_ok != total:
        print("  ※ 一致しない画像があります。過去のスコアと比較する際は注意してください。")
    print()
    print("確認:")
    print("  python tools/eval_model.py best.pt --labels fullwidth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
