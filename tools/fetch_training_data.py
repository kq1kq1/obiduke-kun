"""貯まった学習データを取り出して、YOLO形式のフォルダに組み立てるスクリプト。

レビュー画面で確認・修正したページは、privateなHF Datasetリポジトリに
    data/images/<sha先頭2文字>/<sha>.jpg   ← 帯を貼る前の元画像
    data/records/records_*.jsonl           ← 1行1ページの記録（追記のみ）
という形で貯まる。JSONLは追記のみなので、同じページを何度も直すと行が増える。
そのため「画像ごとに最後の行を採用」して最新の修正だけを使う。

このスクリプトは
    1. リポジトリを丸ごとダウンロード
    2. 画像ごとに最後の記録を採用
    3. images/ と labels/ が並ぶYOLO形式のフォルダを作る
    4. 中身の内訳を表示（データがまともに貯まっているかの確認用）
までをやる。出来たフォルダを既存のRoboflowデータと混ぜて再学習に回す。

使い方:
    $env:HF_TOKEN = "hf_xxx"
    python tools/fetch_training_data.py kq1kq1/obiduke-training-data --out collected

    # 取り出したあと、リポジトリの履歴を1コミットに畳む（コミット数が増え続けるのを防ぐ）
    python tools/fetch_training_data.py kq1kq1/obiduke-training-data --out collected --squash
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

# detect.py の names と同じ順（0:band 1:logo 2:map）
CLASS_NAMES = ["band", "logo", "map"]


def load_records(data_dir):
    """records/*.jsonl を全部読み、画像ごとに最後の記録を返す。

    返り値: (画像パス -> 記録, 読めた行数, 壊れていて飛ばした行数)
    """
    latest = {}
    ok_lines = 0
    bad_lines = 0
    records_dir = data_dir / "records"
    for jsonl in sorted(records_dir.glob("*.jsonl")):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            image = rec.get("image")
            if not image:
                bad_lines += 1
                continue
            ok_lines += 1
            # 同じ画像の記録は後から来たものが勝つ（＝最新の修正を採用）
            prev = latest.get(image)
            if prev is None or rec.get("ts", "") >= prev.get("ts", ""):
                latest[image] = rec
    return latest, ok_lines, bad_lines


def main():
    ap = argparse.ArgumentParser(description="学習データを取り出してYOLO形式に組み立てる")
    ap.add_argument("repo_id", help="例: kq1kq1/obiduke-training-data")
    ap.add_argument("--out", default="collected", help="出力フォルダ（既定: collected）")
    ap.add_argument("--cache", default=None, help="ダウンロード先（既定: HFのキャッシュ）")
    ap.add_argument("--squash", action="store_true",
                    help="取り出し後にリポジトリの履歴を1コミットに畳む（データは消えない・履歴のみ消える）")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("エラー: 環境変数 HF_TOKEN を設定してください（privateリポジトリの読み取りに必要）。\n"
              '       PowerShell:  $env:HF_TOKEN = "hf_xxx"', file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        print("エラー: huggingface_hub が入っていません。 pip install huggingface_hub", file=sys.stderr)
        return 1

    print(f"ダウンロード中: {args.repo_id}")
    try:
        local = Path(snapshot_download(
            repo_id=args.repo_id, repo_type="dataset", token=token, local_dir=args.cache,
        ))
    except Exception as e:
        print(f"エラー: ダウンロードに失敗しました: {e}", file=sys.stderr)
        return 1

    data_dir = local / "data"
    if not (data_dir / "records").is_dir():
        print(f"エラー: {data_dir}/records が見つかりません。まだデータが貯まっていない可能性があります。",
              file=sys.stderr)
        return 1

    latest, ok_lines, bad_lines = load_records(data_dir)
    if not latest:
        print("記録が0件でした。レビュー画面で✓を押すか、枠を直して貼り直すと貯まります。")
        return 0

    out = Path(args.out)
    img_out, lbl_out = out / "images", out / "labels"
    for d in (img_out, lbl_out):
        d.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_img = 0
    cls_count = Counter()
    empty_pages = 0
    edited_pages = 0
    for image_rel, rec in sorted(latest.items()):
        src = data_dir / image_rel
        if not src.is_file():
            missing_img += 1
            continue
        stem = Path(image_rel).stem
        shutil.copyfile(src, img_out / f"{stem}.jpg")
        label = rec.get("label", "") or ""
        (lbl_out / f"{stem}.txt").write_text(
            (label + "\n") if label else "", encoding="utf-8"
        )
        written += 1
        if rec.get("edited"):
            edited_pages += 1
        lines = [l for l in label.splitlines() if l.strip()]
        if not lines:
            empty_pages += 1
        for l in lines:
            try:
                cid = int(float(l.split()[0]))
            except (ValueError, IndexError):
                continue
            cls_count[CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)] += 1

    print()
    print(f"出力先: {out.resolve()}")
    print(f"  記録の行数         : {ok_lines}（同じページの再修正を除いたページ数: {len(latest)}）")
    print(f"  書き出したページ数 : {written}")
    if bad_lines:
        print(f"  壊れていて飛ばした行: {bad_lines}")
    if missing_img:
        print(f"  画像が無くて飛ばした: {missing_img}")
    print(f"  内訳: 人が修正 {edited_pages} / 確認のみ {written - edited_pages}")
    print(f"  枠が0個のページ    : {empty_pages}  ← 「ここには何も無い」の負例。誤検出を減らすのに効く")
    print("  クラス別の枠の数   : " + (", ".join(f"{k}={v}" for k, v in sorted(cls_count.items())) or "なし"))
    print()
    print("このフォルダを既存のRoboflowデータと混ぜて学習してください。")
    print("bandラベルの規約を揃えていない既存データがあれば、先に")
    print("  python tools/relabel_band_fullwidth.py <既存データのルート>")
    print("を実行して全幅に揃えること（混ざると精度が落ちます）。")

    if args.squash:
        # データを取り出した直後に履歴を畳むと、コミット数が増え続けるのを防げる。
        # 追記専用のデータ置き場なので履歴に価値はない。データ本体は消えない。
        print()
        print("履歴を1コミットに畳みます（データは消えません／履歴のみ消えます・取り消し不可）...")
        if written == 0:
            print("  中止: 1件も書き出せていないので畳みません（取り出しに失敗している可能性）。")
            return 1
        try:
            HfApi(token=token).super_squash_history(
                repo_id=args.repo_id, repo_type="dataset",
                commit_message="学習データの取り出し後に履歴を整理",
            )
            print("  完了: リポジトリの履歴を整理しました。")
        except Exception as e:
            print(f"  失敗: {e}", file=sys.stderr)
            print("  （データはHub上に残っています。畳めなかっただけなので次回また試せます）")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
