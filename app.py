print("=== Flask server started! ===")
import os
import uuid
from dataclasses import dataclass
from typing import Optional, List

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    send_from_directory,
    redirect,
    url_for,
    flash,
)

import numpy as np
from PIL import Image, ImageDraw
import fitz  # PyMuPDF
from skimage.metrics import structural_similarity as ssim

# ============================================
# 設定
# ============================================


@dataclass
class DetectionConfig:
    landscape_bottom_ratio: float = 0.15
    portrait_center_ratio: float = 0.5
    portrait_half_height_ratio: float = 0.1
    ssim_threshold: float = 0.80
    mse_threshold: float = 0.02
    step_ratio: float = 0.03
    duplicate_ssim_threshold: float = 0.98
    band_height_ratio: float = 0.16


BASE = os.path.dirname(__file__)
DIR_TEMPLATES = os.path.join(BASE, "band_templates")
DIR_UPLOAD = os.path.join(BASE, "uploads")
DIR_OUTPUT = os.path.join(BASE, "outputs")
OWN_BAND = os.path.join(BASE, "band_default.png")

os.makedirs(DIR_TEMPLATES, exist_ok=True)
os.makedirs(DIR_UPLOAD, exist_ok=True)
os.makedirs(DIR_OUTPUT, exist_ok=True)

config = DetectionConfig()

# ============================================
# ユーティリティ関数
# ============================================


def load_gray(path):
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def resize_gray(arr, h, w):
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.resize((w, h), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def mse(a, b):
    return float(np.mean((a - b)**2))


# ============================================
# テンプレDB
# ============================================


class BandTemplate:

    def __init__(self, name, arr):
        self.name = name
        self.gray = arr
        self.h, self.w = arr.shape


class TemplateDB:

    def __init__(self):
        self.items: List[BandTemplate] = []

    def load(self):
        files = [
            f for f in os.listdir(DIR_TEMPLATES)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        for f in files:
            path = os.path.join(DIR_TEMPLATES, f)
            arr = load_gray(path)
            self.items.append(BandTemplate(f, arr))


db = TemplateDB()

# ============================================
# 帯検出
# ============================================


def detect_band(gray):
    h, w = gray.shape
    orient = "landscape" if w >= h else "portrait"

    if orient == "landscape":
        bh = int(h * config.landscape_bottom_ratio)
        y0, y1 = h - bh, h
    else:
        center = int(h * config.portrait_center_ratio)
        half = int(h * config.portrait_half_height_ratio)
        y0, y1 = center - half, center + half

    region = gray[y0:y1, :]

    # 色反転も試す
    region_inv = 1 - region

    best = None

    for tmpl in db.items:
        t_h, t_w = tmpl.h, tmpl.w

        target_h = region.shape[0]
        target_w = int(t_w * (target_h / t_h))
        if target_w <= 0 or target_w > region.shape[1]:
            continue

        resized = resize_gray(tmpl.gray, target_h, target_w)

        w_region = region.shape[1]
        step = max(1, int(w_region * config.step_ratio))

        for x0 in range(0, w_region - target_w + 1, step):
            x1 = x0 + target_w

            patch = region[:, x0:x1]
            patch_inv = region_inv[:, x0:x1]

            s1 = ssim(patch, resized, data_range=1.0)
            m1 = mse(patch, resized)

            s2 = ssim(patch_inv, resized, data_range=1.0)
            m2 = mse(patch_inv, resized)

            if s2 > s1:
                s1, m1 = s2, m2

            if (s1 >= config.ssim_threshold and m1 <= config.mse_threshold):
                return (orient, x0, y0, x1, y1)

    return None


# ============================================
# 白塗り＋自社帯追加
# ============================================


def apply_band(page_img, det):
    orient, x0, y0, x1, y1 = det

    page = page_img.convert("RGB")
    draw = ImageDraw.Draw(page)
    W, H = page.size

    # 帯領域白塗り
    draw.rectangle([(x0, y0), (x1, y1)], fill=(255, 255, 255))

    # 自社帯
    bh = int(H * config.band_height_ratio)
    draw.rectangle([(0, H - bh), (W, H)], fill=(255, 255, 255))
    band = Image.open(OWN_BAND).convert("RGBA")
    band = band.resize((W, bh), Image.LANCZOS)
    page.paste(band, (0, H - bh), band)

    return page


# ============================================
# PDF処理
# ============================================


def process_pdf(input_path, output_path):
    db.load()

    doc = fitz.open(input_path)
    pages_out = []

    for i in range(len(doc)):
        p = doc.load_page(i)
        pix = p.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        g = img.convert("L")
        g_arr = np.asarray(g, dtype=np.float32) / 255.0

        det = detect_band(g_arr)

        if det:
            img = apply_band(img, det)

        pages_out.append(img)

    first, *rest = pages_out
    first.save(output_path, save_all=True, append_images=rest)


# ============================================
# Flask
# ============================================

app = Flask(__name__)
app.secret_key = "secretkey"


@app.route("/")
def index():
    files = os.listdir(DIR_TEMPLATES)
    return render_template("index.html", templates=files)


@app.route("/process", methods=["POST"])
def handle_pdf():
    if "pdf_file" not in request.files:
        flash("PDFファイルがありません")
        return redirect("/")

    pdf = request.files["pdf_file"]
    if not pdf.filename.lower().endswith(".pdf"):
        flash("PDFをアップしてください")
        return redirect("/")

    pdf_id = str(uuid.uuid4())
    in_path = os.path.join(DIR_UPLOAD, pdf_id + ".pdf")
    out_path = os.path.join(DIR_OUTPUT, pdf_id + "_out.pdf")
    pdf.save(in_path)

    try:
        process_pdf(in_path, out_path)
    except Exception as e:
        flash("処理中にエラー: " + str(e))
        return redirect("/")

    return send_file(out_path, as_attachment=True, download_name="output.pdf")


@app.route("/bands")
def bands():
    files = os.listdir(DIR_TEMPLATES)
    return render_template("bands.html", templates=files)


@app.route("/band_templates/<path:filename>")
def band_template_file(filename):
    """band_templates フォルダ内の画像をブラウザに配信する"""
    return send_from_directory(DIR_TEMPLATES, filename)


@app.route("/bands/delete", methods=["POST"])
def delete_band():
    """テンプレ画像を削除"""
    filename = request.form.get("filename")
    if not filename:
        flash("ファイル名が指定されていません")
        return redirect(url_for("bands"))

    safe_name = os.path.basename(filename)
    path = os.path.join(DIR_TEMPLATES, safe_name)

    if os.path.isfile(path):
        os.remove(path)
        flash(f"「{safe_name}」を削除しました")
    else:
        flash("ファイルが見つかりませんでした")

    return redirect(url_for("bands"))


@app.route("/bands/rename", methods=["POST"])
def rename_band():
    """テンプレ画像の名前変更"""
    old_name = request.form.get("old_name")
    new_name = request.form.get("new_name")

    if not old_name or not new_name:
        flash("名前が正しく指定されていません")
        return redirect(url_for("bands"))

    old_name = os.path.basename(old_name)
    new_name = os.path.basename(new_name)

    # 拡張子は元のまま保持する（.png など）
    old_root, old_ext = os.path.splitext(old_name)
    new_root, new_ext = os.path.splitext(new_name)

    if new_ext == "":
        new_name_final = new_root + old_ext
    else:
        new_name_final = new_root + new_ext

    old_path = os.path.join(DIR_TEMPLATES, old_name)
    new_path = os.path.join(DIR_TEMPLATES, new_name_final)

    if not os.path.isfile(old_path):
        flash("元ファイルが見つかりませんでした")
        return redirect(url_for("bands"))

    if os.path.exists(new_path) and new_path != old_path:
        flash("同じ名前のファイルが既に存在します")
        return redirect(url_for("bands"))

    os.rename(old_path, new_path)
    flash(f"「{old_name}」を「{new_name_final}」に変更しました")

    return redirect(url_for("bands"))


@app.route("/bands/upload", methods=["POST"])
def upload_band():
    if "band_file" not in request.files:
        flash("帯画像がありません")
        return redirect("/bands")

    img = request.files["band_file"]
    if img.filename == "":
        flash("ファイル未選択")
        return redirect("/bands")

    save_path = os.path.join(DIR_TEMPLATES, img.filename)
    img.save(save_path)

    flash("登録しました")
    return redirect("/bands")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
