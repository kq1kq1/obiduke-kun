print("=== Flask server started! ===")
import os
import uuid
from dataclasses import dataclass
from typing import Optional, List
from threading import Thread, Lock

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    send_from_directory,
    redirect,
    url_for,
    flash,
    jsonify,
)

import numpy as np
from PIL import Image, ImageDraw
import fitz  # PyMuPDF
from skimage.metrics import structural_similarity as ssim

# ============================================
# 設定
# ============================================

DEBUG_DETECT = True


@dataclass
class DetectionConfig:
    landscape_bottom_ratio: float = 0.15
    portrait_center_ratio: float = 0.5
    portrait_half_height_ratio: float = 0.1

    ssim_threshold: float = 0.65
    mse_threshold: float = 0.06

    step_ratio: float = 0.03
    duplicate_ssim_threshold: float = 0.98
    band_height_ratio: float = 0.16


BASE = os.path.dirname(__file__)
DIR_TEMPLATES = os.path.join(BASE, "band_templates")
DIR_UPLOAD = os.path.join(BASE, "uploads")
DIR_OUTPUT = os.path.join(BASE, "outputs")
OWN_BAND = os.path.join(BASE, "band_default.png")


def clear_old_files():
    """前回のアップロードPDFと出力PDFを削除する"""
    for d in (DIR_UPLOAD, DIR_OUTPUT):
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            path = os.path.join(d, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"[warn] {path} を削除できませんでした: {e}")


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
        self.items.clear()

        files = [
            f for f in os.listdir(DIR_TEMPLATES)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        for f in files:
            path = os.path.join(DIR_TEMPLATES, f)
            arr = load_gray(path)
            self.items.append(BandTemplate(f, arr))

        if DEBUG_DETECT:
            print(f"[debug] loaded templates: {len(self.items)}")


db = TemplateDB()

# ============================================
# 帯検出
# ============================================


def detect_band(gray, page_index=None):
    h, w = gray.shape
    orient = "landscape" if w >= h else "portrait"

    if DEBUG_DETECT:
        print(f"[page-info] page={page_index} orient={orient} size=({w}x{h})")

    height_min = 0.05
    height_max = 0.18
    step_height = 0.01

    height_ratios: list[float] = []
    r = height_min
    while r <= height_max + 1e-6:
        height_ratios.append(r)
        r += step_height

    resized_cache: dict[tuple[int, int, int], np.ndarray] = {}
    mse_prefilter = float(getattr(config, "mse_threshold", 0.15)) * 1.2

    region_w_global = w
    best_any = None

    def build_candidate_windows(
            mode: str,
            offset_step_landscape: float = 0.02,
            bottom_step_portrait: float = 0.01,
            center_step_portrait: float = 0.01) -> list[tuple[int, int]]:
        candidate_windows: list[tuple[int, int]] = []

        if orient == "landscape":
            for hr in height_ratios:
                win_h = int(h * hr)
                if win_h < 8:
                    continue
                offset = 0.0
                while offset <= 0.20 + 1e-6:
                    y1 = h - int(offset * h)
                    y0 = y1 - win_h
                    if y0 < 0 or y1 > h:
                        break
                    candidate_windows.append((y0, y1))
                    offset += offset_step_landscape

        else:
            for hr in height_ratios:
                win_h = int(h * hr)
                if win_h < 8:
                    continue

                if mode in ("all", "bottom"):
                    offset = 0.0
                    while offset <= 0.10 + 1e-6:
                        y1 = h - int(offset * h)
                        y0 = y1 - win_h
                        if y0 < 0 or y1 > h:
                            break
                        candidate_windows.append((y0, y1))
                        offset += bottom_step_portrait

                if mode in ("all", "center"):
                    center_ratio = 0.40
                    while center_ratio <= 0.55 + 1e-6:
                        center = int(h * center_ratio)
                        y0 = center - win_h // 2
                        y1 = y0 + win_h
                        if 0 <= y0 and y1 <= h:
                            candidate_windows.append((y0, y1))
                        center_ratio += center_step_portrait

        return sorted(set(candidate_windows))

    def scan(candidate_windows: list[tuple[int, int]]):
        nonlocal best_any

        for (y0, y1) in candidate_windows:
            region = gray[y0:y1, :]
            if region.size == 0:
                continue

            region_h, region_w = region.shape

            for tmpl in db.items:
                t_h, t_w = tmpl.h, tmpl.w
                if t_h <= 0 or t_w <= 0:
                    continue

                target_w = region_w
                scale = target_w / float(t_w)
                target_h = int(t_h * scale)

                if target_h <= 0 or target_h > region_h:
                    continue

                key = (id(tmpl), target_h, target_w)
                resized = resized_cache.get(key)
                if resized is None:
                    resized = resize_gray(tmpl.gray, target_h, target_w)
                    resized_cache[key] = resized

                v_offset = (region_h - target_h) // 2
                y0_patch = v_offset
                y1_patch = v_offset + target_h
                patch = region[y0_patch:y1_patch, :]

                m1 = mse(patch, resized)
                if m1 > mse_prefilter:
                    continue

                s1 = ssim(patch, resized, data_range=1.0)

                gx0 = 0
                gx1 = region_w_global
                gy0 = y0 + y0_patch
                gy1 = y0 + y1_patch

                if (best_any is None) or (s1 > best_any[0]):
                    best_any = (s1, m1, orient, gx0, gy0, gx1, gy1)

                if (s1 >= config.ssim_threshold
                        and m1 <= config.mse_threshold):
                    return (s1, m1, orient, gx0, gy0, gx1, gy1)

        return None

    results: list[tuple[str, int, int, int, int]] = []

    if orient == "portrait":
        bottom_windows = build_candidate_windows("bottom",
                                                 bottom_step_portrait=0.01,
                                                 center_step_portrait=0.01)
        hit_bottom = scan(bottom_windows)
        if hit_bottom is not None:
            _, m_hit, o, x0, y0, x1, y1 = hit_bottom
            results.append((o, x0, y0, x1, y1))

            center_windows = build_candidate_windows("center",
                                                     bottom_step_portrait=0.01,
                                                     center_step_portrait=0.01)
            hit_center = scan(center_windows)
            if hit_center is not None:
                _, m_hit2, o2, x02, y02, x12, y12 = hit_center
                if not (abs(y02 - y0) < 5 and abs(y12 - y1) < 5):
                    results.append((o2, x02, y02, x12, y12))

        else:
            center_windows = build_candidate_windows("center",
                                                     bottom_step_portrait=0.01,
                                                     center_step_portrait=0.01)
            hit_center = scan(center_windows)
            if hit_center is not None:
                _, m_hit2, o2, x02, y02, x12, y12 = hit_center
                results.append((o2, x02, y02, x12, y12))

    else:
        windows = build_candidate_windows("all", offset_step_landscape=0.02)
        hit = scan(windows)
        if hit is not None:
            _, _, o, x0, y0, x1, y1 = hit
            results.append((o, x0, y0, x1, y1))

    if DEBUG_DETECT and best_any is not None:
        s_best, m_best, _, bx0, by0, bx1, by1 = best_any
        print(f"[detect-best] page={page_index} "
              f"best_ssim={s_best:.3f} best_mse={m_best:.4f} "
              f"rect=({bx0},{by0})-({bx1},{by1})")

    if DEBUG_DETECT:
        if not results:
            print(f"[detect] page={page_index} NO band detected")
        else:
            for (o, x0, y0, x1, y1) in results:
                print(
                    f"[detect] page={page_index} HIT orient={o} rect=({x0},{y0})-({x1},{y1})"
                )

    return results


# ============================================
# 白塗り＋自社帯追加
# ============================================


def apply_band(page_img, det):
    orient, x0, y0, x1, y1 = det

    page = page_img.convert("RGBA")
    W, H = page.size

    x0 = max(0, min(int(x0), W))
    x1 = max(0, min(int(x1), W))
    y0 = max(0, min(int(y0), H))
    y1 = max(0, min(int(y1), H))

    if x1 <= x0 or y1 <= y0:
        return page_img

    band_w = x1 - x0
    band_h = y1 - y0

    draw = ImageDraw.Draw(page)
    draw.rectangle([(x0, y0), (x1, y1)], fill=(255, 255, 255, 255))

    band = Image.open(OWN_BAND).convert("RGBA")
    band = band.resize((band_w, band_h), Image.LANCZOS)

    page.paste(band, (x0, y0), band)

    return page.convert("RGB")


# ============================================
# PDF処理
# ============================================


def process_pdf(input_path, output_path):
    db.load()
    doc = fitz.open(input_path)
    pages_out = []
    with progress_lock:
        progress["status"] = "running"
        progress["current"] = 0
        progress["total"] = len(doc)
        progress["error"] = None
        progress["out_path"] = output_path

    for i, page in enumerate(doc):
        max_px = 2200
        rect = page.rect
        scale = min(max_px / rect.width, max_px / rect.height)

        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        g = img.convert("L")
        g_arr = np.asarray(g, dtype=np.float32) / 255.0

        dets = detect_band(g_arr, page_index=i + 1)

        for det in dets:
            img = apply_band(img, det)

        with progress_lock:
            progress["current"] = i + 1

        pages_out.append(img)

    first, *rest = pages_out
    first.save(output_path, save_all=True, append_images=rest)


# ============================================
# Flask
# ============================================

app = Flask(__name__)
app.secret_key = "secretkey"

progress_lock = Lock()
progress = {
    "status": "idle",  # idle / running / done / error
    "current": 0,
    "total": 0,
    "job_id": None,
    "error": None,
}


def render_page_to_pil(doc,
                       page_index: int,
                       max_px: int = 2200) -> Image.Image:
    page = doc[page_index]
    rect = page.rect
    scale = min(max_px / rect.width, max_px / rect.height)

    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def auto_extract_band_from_page(page_img: Image.Image):
    g = page_img.convert("L")
    g_arr = np.asarray(g, dtype=np.float32) / 255.0
    h, w = g_arr.shape

    orient = "landscape" if w >= h else "portrait"

    height_ratios = [0.10, 0.14, 0.18, 0.22]
    candidate = []

    if orient == "landscape":
        for hr in height_ratios:
            win_h = int(h * hr)
            if win_h < 10:
                continue
            for offset_ratio in [0.0, 0.05, 0.10, 0.15, 0.20]:
                y1 = h - int(offset_ratio * h)
                y0 = y1 - win_h
                if y0 < 0 or y1 > h:
                    continue
                candidate.append((y0, y1))
    else:
        for hr in height_ratios:
            win_h = int(h * hr)
            if win_h < 10:
                continue
            for center_ratio in [0.40, 0.45, 0.50, 0.60]:
                center = int(h * center_ratio)
                y0 = center - win_h // 2
                y1 = y0 + win_h
                if y0 < 0 or y1 > h:
                    continue
                candidate.append((y0, y1))

    if not candidate:
        return None

    y0, y1 = sorted(candidate, key=lambda r: r[1])[-1]
    return (y0 / h, y1 / h)


@app.route("/bands/extract_from_pdf", methods=["POST"])
def extract_band_from_pdf():
    if "pdf_file" not in request.files:
        flash("PDFファイルがありません")
        return redirect(url_for("bands"))

    pdf_file = request.files["pdf_file"]
    if pdf_file.filename == "":
        flash("ファイルが選択されていません")
        return redirect(url_for("bands"))

    if not pdf_file.filename.lower().endswith(".pdf"):
        flash("PDFファイルを指定してください")
        return redirect(url_for("bands"))

    pdf_id = str(uuid.uuid4())
    pdf_filename = f"preview_{pdf_id}.pdf"
    pdf_path = os.path.join(DIR_UPLOAD, pdf_filename)
    pdf_file.save(pdf_path)

    try:
        doc = fitz.open(pdf_path)
        page_index = 0
        img = render_page_to_pil(doc, page_index, max_px=2200)

        preview_name = f"preview_{pdf_id}.png"
        preview_path = os.path.join(DIR_UPLOAD, preview_name)
        img.save(preview_path)

        init = auto_extract_band_from_page(img)
        if init is None:
            y0_ratio, y1_ratio = 0.7, 0.9
        else:
            y0_ratio, y1_ratio = init

    except Exception as e:
        print("[extract_from_pdf] error:", e)
        flash("PDFの解析中にエラーが発生しました: " + str(e))
        return redirect(url_for("bands"))

    return render_template(
        "band_preview.html",
        pdf_id=pdf_id,
        pdf_filename=pdf_filename,
        preview_image=preview_name,
        page_index=page_index,
        y0_ratio=y0_ratio,
        y1_ratio=y1_ratio,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(DIR_UPLOAD, filename)


@app.route("/bands/save_cropped", methods=["POST"])
def save_cropped_band():
    pdf_id = request.form.get("pdf_id")
    pdf_filename = request.form.get("pdf_filename")
    preview_name = request.form.get("preview_image")
    page_index = int(request.form.get("page_index", "0"))
    y0_ratio = float(request.form.get("y0_ratio"))
    y1_ratio = float(request.form.get("y1_ratio"))
    template_name = request.form.get("template_name") or "band_template"

    if not pdf_filename or not pdf_id:
        flash("情報が不足しています")
        return redirect(url_for("bands"))

    pdf_path = os.path.join(DIR_UPLOAD, pdf_filename)

    try:
        doc = fitz.open(pdf_path)
        img = render_page_to_pil(doc, page_index, max_px=2200)
        w, h = img.size

        y0 = int(h * min(max(y0_ratio, 0.0), 1.0))
        y1 = int(h * min(max(y1_ratio, 0.0), 1.0))
        if y1 <= y0:
            flash("帯の範囲が不正です（下側が上側以下になっています）")
            return redirect(url_for("bands"))

        band_img = img.crop((0, y0, w, y1))

        target_width = 1200
        if w != target_width:
            new_h = int((y1 - y0) * (target_width / w))
            band_img = band_img.resize((target_width, new_h), Image.LANCZOS)

        base = os.path.splitext(os.path.basename(template_name))[0]
        out_name = f"{base}_{pdf_id}.png"
        out_path = os.path.join(DIR_TEMPLATES, out_name)
        band_img.save(out_path)

        flash(f"テンプレート「{out_name}」を作成しました")

    except Exception as e:
        print("[save_cropped_band] error:", e)
        flash("テンプレ保存中にエラーが発生しました: " + str(e))

    try:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        preview_path = os.path.join(DIR_UPLOAD, preview_name)
        if os.path.exists(preview_path):
            os.remove(preview_path)
    except Exception:
        pass

    return redirect(url_for("bands"))


@app.route("/")
def index():
    files = os.listdir(DIR_TEMPLATES)
    return render_template("index.html", templates=files)


@app.route("/process", methods=["POST"])
def handle_pdf():
    if "pdf_file" not in request.files:
        flash("PDFファイルがありません")
        return redirect(url_for("index"))

    pdf = request.files["pdf_file"]
    if pdf.filename == "" or (not pdf.filename.lower().endswith(".pdf")):
        flash("PDFをアップしてください")
        return redirect(url_for("index"))

    clear_old_files()

    job_id = str(uuid.uuid4())
    in_path = os.path.join(DIR_UPLOAD, job_id + ".pdf")
    out_path = os.path.join(DIR_OUTPUT, job_id + "_out.pdf")
    pdf.save(in_path)

    with progress_lock:
        progress["status"] = "running"
        progress["current"] = 0
        progress["total"] = 0
        progress["job_id"] = job_id
        progress["error"] = None
        progress["download"] = None

    def worker():
        try:
            process_pdf(in_path, out_path)

            with progress_lock:
                progress["status"] = "done"
                progress["download"] = os.path.basename(out_path)

        except Exception as e:
            with progress_lock:
                progress["status"] = "error"
                progress["error"] = str(e)

    Thread(target=worker, daemon=True).start()

    return render_template("processing.html", job_id=job_id)


@app.route("/bands")
def bands():
    files = [
        f for f in os.listdir(DIR_TEMPLATES)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    return render_template("bands.html", templates=files)


@app.route("/band_templates/<path:filename>")
def band_template_file(filename):
    return send_from_directory(DIR_TEMPLATES, filename)


@app.route("/bands/upload", methods=["POST"])
def upload_band():
    if "band_file" not in request.files:
        flash("帯画像がありません")
        return redirect(url_for("bands"))

    img = request.files["band_file"]
    if img.filename == "":
        flash("ファイル未選択")
        return redirect(url_for("bands"))

    save_path = os.path.join(DIR_TEMPLATES, os.path.basename(img.filename))
    img.save(save_path)

    flash("登録しました")
    return redirect(url_for("bands"))


@app.route("/bands/delete", methods=["POST"])
def delete_band():
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
    old_name = request.form.get("old_name")
    new_name = request.form.get("new_name")

    if not old_name or not new_name:
        flash("名前が正しく指定されていません")
        return redirect(url_for("bands"))

    old_name = os.path.basename(old_name)
    new_name = os.path.basename(new_name)

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


@app.route("/progress")
def get_progress():
    with progress_lock:
        return {
            "status": progress["status"],
            "current": progress["current"],
            "total": progress["total"],
            "job_id": progress["job_id"],
            "error": progress["error"],
        }


@app.route("/download/<job_id>")
def download(job_id):
    out_path = os.path.join(DIR_OUTPUT, job_id + "_out.pdf")
    if not os.path.exists(out_path):
        return "not ready", 404
    return send_file(out_path, as_attachment=True, download_name="output.pdf")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
