"""土台データ（Roboflowの学習分＋凍結検証セット）をHFのDatasetリポジトリに置く。

一度だけ実行するもの。以後の再学習は Colab から HF_TOKEN だけで全部そろう。

なぜHubに置くか:
  再学習はGPUが要るのでColabで回すことになる。そのとき「蓄積データはHubにあるが、
  土台データは自分のPCにある」だと毎回アップロードすることになって面倒だし、
  凍結検証セットが自分のPCにしか無いのは（消えたら比較の物差しを失うので）危ない。

置かれる形:
  <リポジトリ>/base/train/images, labels        Roboflowの学習分（bandは全幅に統一済み）
  <リポジトリ>/base/frozen_val/images, labels   凍結検証セット（絶対に学習に使わない）
  <リポジトリ>/base/README.md                   中身の説明
  ※ アプリが送る蓄積データは data/ 配下。ここには触らない。

使い方:
    $env:HF_TOKEN = "hf_xxx"
    python tools/upload_base_dataset.py kq1kq1/obiduke-training-data --roboflow datasets/roboflow/v3
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

BAND_CLASS_ID = 0
NAMES = ["band", "logo", "map"]
FROZEN = Path("datasets/frozen_val")


def to_fullwidth(line):
    """band行を全幅に揃える（tools/relabel_band_fullwidth.py と同じ規約）。"""
    p = line.split()
    if len(p) != 5:
        return line
    try:
        cid = int(float(p[0]))
        xc, yc, w, h = (float(v) for v in p[1:])
    except ValueError:
        return line
    if cid != BAND_CLASS_ID:
        return line
    return "%d %.6f %.6f %.6f %.6f" % (cid, 0.5, yc, 1.0, h)


def copy_split(src_img_dir, src_lbl_dir, dst_root, relabel):
    """画像とラベルを1組コピーする。relabel=Trueならbandを全幅に直す。"""
    (dst_root / "images").mkdir(parents=True, exist_ok=True)
    (dst_root / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for img in sorted(src_img_dir.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = src_lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue
        shutil.copyfile(img, dst_root / "images" / img.name)
        lines = [l.strip() for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]
        if relabel:
            lines = [to_fullwidth(l) for l in lines]
        (dst_root / "labels" / lbl.name).write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="土台データをHF Datasetリポジトリに置く")
    ap.add_argument("repo_id", help="例: kq1kq1/obiduke-training-data")
    ap.add_argument("--roboflow", default="datasets/roboflow/v3",
                    help="RoboflowのYOLOv8エクスポートを展開したフォルダ")
    ap.add_argument("--dry-run", action="store_true", help="送らずに中身だけ確認する")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token and not args.dry_run:
        print('エラー: 環境変数 HF_TOKEN を設定してください。\n'
              '       PowerShell:  $env:HF_TOKEN = "hf_xxx"', file=sys.stderr)
        return 1

    rf = Path(args.roboflow)
    if not (rf / "train" / "images").is_dir():
        print(f"エラー: {rf}/train/images が見つかりません", file=sys.stderr)
        return 1
    if not (FROZEN / "images").is_dir():
        print("エラー: 凍結検証セットがありません。先に tools/freeze_val_set.py を実行してください。",
              file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "base"

        # 学習分: bandを全幅に揃えてから置く（アプリが貯めるラベルと規約を合わせるため）
        n_train = copy_split(rf / "train" / "images", rf / "train" / "labels",
                             base / "train", relabel=True)
        # 検証分: 凍結セットの全幅版ラベルをそのまま使う
        n_val = copy_split(FROZEN / "images", FROZEN / "labels_fullwidth",
                           base / "frozen_val", relabel=False)

        # 検証画像が学習分に混ざっていないか（混ざると基準スコアが甘く出て比較にならない）
        train_names = {p.name for p in (base / "train" / "images").iterdir()}
        val_names = {p.name for p in (base / "frozen_val" / "images").iterdir()}
        overlap = train_names & val_names
        if overlap:
            print(f"エラー: 検証用の画像が学習分にも入っています（{len(overlap)}件）。"
                  f"例: {sorted(overlap)[:3]}", file=sys.stderr)
            return 1

        (base / "README.md").write_text(
            "# 帯付けくん 再学習の土台データ\n\n"
            "| 場所 | 中身 |\n|---|---|\n"
            f"| `base/train/` | Roboflowの学習分 {n_train}枚。bandは全幅に統一済み |\n"
            f"| `base/frozen_val/` | 凍結検証セット {n_val}枚。**学習に使わないこと** |\n"
            "| `data/` | アプリが自動で送る蓄積データ（人が確認・修正したページ） |\n\n"
            f"クラス: {NAMES}（0=band, 1=logo, 2=map）\n\n"
            "bandのラベルは常に全幅（x_center=0.5, width=1.0）。アプリが白塗り時に必ず\n"
            "全幅へ広げるため、規約を揃えないと学習が濁る。\n",
            encoding="utf-8")

        print(f"学習分   : {n_train}枚（bandを全幅に統一）")
        print(f"検証分   : {n_val}枚（凍結セット・学習には使わない）")
        print(f"重複なし : OK")

        if args.dry_run:
            print("\n--dry-run のため送信しませんでした。")
            return 0

        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                        private=True, exist_ok=True)
        print(f"\n送信中: {args.repo_id} の base/ ...")
        api.upload_folder(folder_path=str(base), path_in_repo="base",
                          repo_id=args.repo_id, repo_type="dataset",
                          commit_message="再学習の土台データ（Roboflow学習分＋凍結検証セット）")

    print("完了。Colabからは HF_TOKEN だけで学習データが全部そろう。")
    print(f"  https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
