"""学習用画像の書き出しスクリプト。

samples/ 内のPDFを1ページ=1枚のPNGに変換し、training_images/ に出力する。
出来た画像をRoboflowにアップロードして、他社帯を四角で囲む（ラベリング）。

使い方:
    python export_images.py
"""
import sys
from pathlib import Path

import fitz  # PyMuPDF

# 出力解像度。長辺をこのピクセル数にそろえる（アプリの処理解像度と同じ＝学習と本番の見え方を一致させる）
LONG_EDGE_PX = 1500

SRC_DIR = Path("samples")
OUT_DIR = Path("training_images")


def export_pdf(pdf_path: Path) -> int:
    """1つのPDFを全ページPNG化して保存。保存枚数を返す。"""
    count = 0
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            # 長辺がLONG_EDGE_PXになる倍率を求める（72dpi基準のページサイズから算出）
            rect = page.rect
            long_edge = max(rect.width, rect.height)
            scale = LONG_EDGE_PX / long_edge
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            # ファイル名: 元PDF名_pページ番号.png（重複しないよう元名を含める）
            out_name = f"{pdf_path.stem}_p{i + 1:02d}.png"
            pix.save(OUT_DIR / out_name)
            count += 1
    return count


def main():
    if not SRC_DIR.exists():
        print(f"エラー: {SRC_DIR}/ が見つかりません。PDFを置いてから実行してください。")
        sys.exit(1)

    OUT_DIR.mkdir(exist_ok=True)
    pdfs = sorted(SRC_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"エラー: {SRC_DIR}/ にPDFがありません。")
        sys.exit(1)

    total = 0
    for pdf in pdfs:
        n = export_pdf(pdf)
        total += n
        print(f"  {pdf.name}: {n}枚")

    print(f"\n完了: 計{total}枚を {OUT_DIR}/ に出力しました。")
    print("次の手順: training_images/ の画像をRoboflowにアップロードして、他社帯を四角で囲んでください。")


if __name__ == "__main__":
    main()
