"""既存のRoboflowラベル(band)を全幅に揃えるスクリプト。

帯付けくんは「他社帯は全幅の横帯」としてラベルの規約を統一している。理由:
- 白塗りは必ず全幅に広げるので、レビュー画面で人が直す枠も全幅になる
- 検出枠と手修正枠でx方向の規約が混ざると、YOLOの枠回帰が濁って精度が落ちる

そのため、これまでRoboflowで「帯にぴったり」付けていたbandラベルも全幅に揃える。
RoboflowからYOLOv8形式でExportしたフォルダに対して実行すると、band(class 0)の行だけ
x_center=0.5 / width=1.0 に書き換える。y方向とlogo/mapは触らない。
Roboflow上で枠を描き直す必要はない。

使い方:
    python tools/relabel_band_fullwidth.py <データセットのルート> --dry-run   # 確認だけ
    python tools/relabel_band_fullwidth.py <データセットのルート>             # 書き換え

<ルート>以下の labels/ フォルダにある .txt すべてが対象（train/valid/test を自動でたどる）。
"""
import argparse
import sys
from pathlib import Path

# detect.py の names と同じ順（0:band 1:logo 2:map）
BAND_CLASS_ID = 0


def convert_line(line):
    """1行を変換する。返り値: (新しい行, 変えたか)。不正な行はそのまま返す。"""
    parts = line.split()
    if len(parts) != 5:
        return line, False
    try:
        cid = int(float(parts[0]))
        xc, yc, w, h = (float(v) for v in parts[1:])
    except ValueError:
        return line, False
    if cid != BAND_CLASS_ID:
        return line, False
    if abs(xc - 0.5) < 1e-6 and abs(w - 1.0) < 1e-6:
        return line, False  # すでに全幅
    return "%d %.6f %.6f %.6f %.6f" % (cid, 0.5, yc, 1.0, h), True


def main():
    ap = argparse.ArgumentParser(description="bandラベルを全幅に揃える")
    ap.add_argument("root", help="YOLO形式データセットのルート（labels/ を含むフォルダ）")
    ap.add_argument("--dry-run", action="store_true", help="書き換えずに件数だけ表示")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"エラー: フォルダが見つかりません: {root}", file=sys.stderr)
        return 1

    txts = sorted(p for p in root.rglob("*.txt") if p.parent.name == "labels")
    if not txts:
        print(f"エラー: {root} 以下に labels/*.txt が見つかりません。\n"
              f"       RoboflowのYOLOv8形式Exportを展開したフォルダを指定してください。",
              file=sys.stderr)
        return 1

    files_changed = 0
    lines_changed = 0
    for path in txts:
        try:
            original = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[warn] 読めませんでした {path}: {e}")
            continue

        out_lines = []
        touched = 0
        for line in original.splitlines():
            if not line.strip():
                continue
            new_line, changed = convert_line(line.strip())
            out_lines.append(new_line)
            if changed:
                touched += 1

        if touched:
            files_changed += 1
            lines_changed += touched
            if not args.dry_run:
                try:
                    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
                except Exception as e:
                    print(f"[warn] 書けませんでした {path}: {e}")

    verb = "変換対象" if args.dry_run else "変換しました"
    print(f"ラベルファイル {len(txts)} 件を確認 → {verb}: {files_changed} ファイル / band {lines_changed} 個")
    if args.dry_run and files_changed:
        print("--dry-run を外すと実際に書き換えます。元データのバックアップを取ってから実行してください。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
