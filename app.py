print("=== Flask server started! ===")
import os
import uuid
import json
import re
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
from PIL import Image, ImageDraw, ImageFilter
import fitz  # PyMuPDF
from skimage.metrics import structural_similarity as ssim

# ============================================
# 設定
# ============================================

DEBUG_DETECT = True

BLUR_RADIUS = 3
MAX_PX = 1600  # 2200→1600: 処理速度約2倍・出力PDFサイズ約半分


@dataclass
class DetectionConfig:
    landscape_bottom_ratio: float = 0.15
    portrait_center_ratio: float = 0.5
    portrait_half_height_ratio: float = 0.1

    ssim_threshold: float = 0.65
    mse_threshold: float = 0.06
    mse_prefilter_multiplier: float = 1.7  # 0.06 * 1.7 = 0.102

    step_ratio: float = 0.03
    duplicate_ssim_threshold: float = 0.98
    band_height_ratio: float = 0.16

    # hrフィルタ: expected_hr ± この値の範囲のウィンドウのみ比較
    hr_filter_margin: float = 0.015


BASE = os.path.dirname(__file__)
DIR_TEMPLATES = os.path.join(BASE, "band_templates")
DIR_OWN_BANDS = os.path.join(BASE, "own_bands")
DIR_UPLOAD = os.path.join(BASE, "uploads")
DIR_OUTPUT = os.path.join(BASE, "outputs")
OWN_BAND_DEFAULT = os.path.join(BASE, "band_default.png")
HIT_COUNT_FILE = os.path.join(BASE, "hit_counts.json")


def get_own_band_path(band_name=None):
    if band_name:
        path = os.path.join(DIR_OWN_BANDS, os.path.basename(band_name))
        if os.path.isfile(path):
            return path
    if os.path.isdir(DIR_OWN_BANDS):
        files = [
            f for f in os.listdir(DIR_OWN_BANDS)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if files:
            return os.path.join(DIR_OWN_BANDS, sorted(files)[0])
    return OWN_BAND_DEFAULT


def list_own_bands():
    if not os.path.isdir(DIR_OWN_BANDS):
        return []
    files = [
        f for f in os.listdir(DIR_OWN_BANDS)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    return sorted(files)


def clear_old_files():
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
            elif os.path.isdir(path):
                try:
                    import shutil
                    shutil.rmtree(path)
                except Exception as e:
                    print(f"[warn] {path} を削除できませんでした: {e}")


os.makedirs(DIR_TEMPLATES, exist_ok=True)
os.makedirs(DIR_OWN_BANDS, exist_ok=True)
os.makedirs(DIR_UPLOAD, exist_ok=True)
os.makedirs(DIR_OUTPUT, exist_ok=True)

config = DetectionConfig()

# ============================================
# ヒット頻度管理
# ============================================

hit_count_lock = Lock()


def load_hit_counts() -> dict:
    if os.path.isfile(HIT_COUNT_FILE):
        try:
            with open(HIT_COUNT_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_hit_counts(counts: dict):
    try:
        with open(HIT_COUNT_FILE, "w") as f:
            json.dump(counts, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[warn] hit_counts保存失敗: {e}")


def increment_hit_count(tmpl_name: str):
    with hit_count_lock:
        counts = load_hit_counts()
        counts[tmpl_name] = counts.get(tmpl_name, 0) + 1
        save_hit_counts(counts)


# ============================================
# ユーティリティ関数
# ============================================


def load_gray_raw(path):
    """グレースケールrawで読み込む（ぼかしなし）"""
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def load_color_raw(path):
    """カラーrawで読み込む（ぼかしなし）"""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def to_gray_blurred(pil_img):
    img = pil_img.convert("L")
    img = img.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
    return np.asarray(img, dtype=np.float32) / 255.0


def to_color_blurred(pil_img):
    """PILカラー画像をぼかしに変換"""
    img = pil_img.convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
    return np.asarray(img, dtype=np.float32) / 255.0


def resize_gray(arr, h, w):
    """グレー画像をリサイズしてからぼかす（resize→blur順）"""
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.resize((w, h), Image.LANCZOS)
    img = img.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
    return np.asarray(img, dtype=np.float32) / 255.0


def resize_color(arr, h, w):
    """カラー画像をリサイズしてからぼかす（resize→blur順）"""
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.resize((w, h), Image.LANCZOS)
    img = img.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
    return np.asarray(img, dtype=np.float32) / 255.0


def mse(a, b):
    return float(np.mean((a - b)**2))


# ============================================
# テンプレDB
# ============================================


class BandTemplate:

    def __init__(self, name, gray, color=None):
        self.name = name
        self.gray = gray  # グレースケールraw（ぼかしなし）
        self.color = color  # カラーraw（ぼかしなし、_colorテンプレのみ）
        self.is_color = color is not None
        self.h, self.w = gray.shape


class TemplateDB:

    def __init__(self):
        self.items: List[BandTemplate] = []

    def load(self):
        self.items.clear()

        files = [
            f for f in os.listdir(DIR_TEMPLATES)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # ヒット頻度順にソート（多い順）
        counts = load_hit_counts()
        files.sort(key=lambda f: counts.get(f, 0), reverse=True)

        for f in files:
            path = os.path.join(DIR_TEMPLATES, f)
            gray = load_gray_raw(path)
            if "_color" in os.path.splitext(f)[0]:
                color = load_color_raw(path)
            else:
                color = None
            self.items.append(BandTemplate(f, gray, color))

        if DEBUG_DETECT:
            print(f"[debug] loaded templates: {len(self.items)}")
            color_count = sum(1 for t in self.items if t.is_color)
            if color_count:
                print(f"[debug] color templates: {color_count}")


db = TemplateDB()

# ============================================
# 帯検出
# ============================================


def detect_band(gray_blurred, color_blurred=None, page_index=None):
    """グレースケール＋ぼかし画像から帯を検出する"""
    h, w = gray_blurred.shape
    orient = "landscape" if w >= h else "portrait"

    if DEBUG_DETECT:
        print(f"[page-info] page={page_index} orient={orient} size=({w}x{h})")

    height_min = 0.04  # 0.05→0.04: メルディアなど細い帯(hr≈4%)対応
    height_max = 0.18
    step_height = 0.01

    height_ratios: list[float] = []
    r = height_min
    while r <= height_max + 1e-6:
        height_ratios.append(r)
        r += step_height

    resized_cache: dict[tuple, np.ndarray] = {}
    mse_prefilter = float(getattr(config, "mse_threshold", 0.06)) * float(
        getattr(config, "mse_prefilter_multiplier", 1.7))
    hr_margin = float(getattr(config, "hr_filter_margin", 0.015))

    region_w_global = w
    best_any = None

    def build_candidate_windows(
            mode: str,
            offset_step_landscape: float = 0.01,  # 0.02→0.01: ±11px→±5px精度
            bottom_step_portrait: float = 0.02,
            center_step_portrait: float = 0.005,  # 0.01→0.005: ±5px精度
    ) -> list[tuple[int, int]]:
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
                    center_ratio = 0.35
                    while center_ratio <= 0.75 + 1e-6:
                        center = int(h * center_ratio)
                        y0 = center - win_h // 2
                        y1 = y0 + win_h
                        if 0 <= y0 and y1 <= h:
                            candidate_windows.append((y0, y1))
                        center_ratio += center_step_portrait

        return sorted(set(candidate_windows))

    def get_patch_and_resized(tmpl, y0, y1, use_color=False):
        """ウィンドウからパッチとリサイズ済みテンプレを取得"""
        source = color_blurred if use_color else gray_blurred
        if source is None:
            return None, None

        region = source[y0:y1, :]
        if region.size == 0:
            return None, None

        if use_color:
            region_h, region_w, _ = region.shape
        else:
            region_h, region_w = region.shape

        t_h, t_w = tmpl.h, tmpl.w
        if t_h <= 0 or t_w <= 0:
            return None, None

        target_w = region_w
        scale = target_w / float(t_w)
        target_h = int(t_h * scale)

        if target_h <= 0 or target_h > region_h:
            return None, None

        key = (id(tmpl), target_h, target_w, use_color)
        resized = resized_cache.get(key)
        if resized is None:
            if use_color:
                resized = resize_color(tmpl.color, target_h, target_w)
            else:
                resized = resize_gray(tmpl.gray, target_h, target_w)
            resized_cache[key] = resized

        v_offset = (region_h - target_h) // 2
        patch = region[v_offset:v_offset + target_h, :]

        return patch, resized

    def scan(candidate_windows: list[tuple[int, int]]):
        nonlocal best_any
        mse_pass_count = 0
        skipped_hr = 0

        for (y0, y1) in candidate_windows:
            win_h = y1 - y0
            win_hr = win_h / h

            for tmpl in db.items:
                # ★ hrフィルタ: テンプレの期待高さ比率から外れたウィンドウをスキップ
                scale_to_page = w / float(tmpl.w)
                expected_hr = (tmpl.h * scale_to_page) / h
                if abs(win_hr - expected_hr) > hr_margin:
                    skipped_hr += 1
                    continue

                use_color = tmpl.is_color and color_blurred is not None

                # ★ グレーMSEで先行フィルタ（カラーテンプレも先にグレーで弾く）
                gray_patch, gray_resized = get_patch_and_resized(
                    tmpl, y0, y1, use_color=False)
                if gray_patch is None:
                    continue

                m_gray = mse(gray_patch, gray_resized)
                if m_gray > mse_prefilter:
                    continue

                mse_pass_count += 1

                if use_color:
                    patch, resized = get_patch_and_resized(tmpl,
                                                           y0,
                                                           y1,
                                                           use_color=True)
                    if patch is None:
                        continue
                    m1 = mse(patch, resized)
                    s1 = ssim(patch, resized, data_range=1.0, channel_axis=2)
                    if DEBUG_DETECT:
                        print(
                            f"[color-ssim] win=({y0},{y1}) s1={s1:.3f} mse={m1:.4f}"
                        )
                else:
                    s1 = ssim(gray_patch, gray_resized, data_range=1.0)
                    m1 = m_gray

                region_h = y1 - y0
                scale = w / float(tmpl.w)
                target_h = int(tmpl.h * scale)
                v_offset = (region_h - target_h) // 2
                gy0 = y0 + v_offset
                gy1 = gy0 + target_h

                if (best_any is None) or (s1 > best_any[0]):
                    best_any = (s1, m1, orient, 0, gy0, region_w_global, gy1)

                if s1 >= config.ssim_threshold and m1 <= config.mse_threshold:
                    if DEBUG_DETECT:
                        print(
                            f"[scan] page={page_index} mse_pass={mse_pass_count} "
                            f"HIT ssim={s1:.3f} tmpl={tmpl.name}")
                    return (s1, m1, orient, 0, gy0, region_w_global, gy1, tmpl)

                if s1 >= 0.92:
                    if DEBUG_DETECT:
                        print(
                            f"[scan] page={page_index} mse_pass={mse_pass_count} "
                            f"HIT ssim={s1:.3f} tmpl={tmpl.name}")
                    return (s1, m1, orient, 0, gy0, region_w_global, gy1, tmpl)

        if DEBUG_DETECT:
            print(f"[scan] page={page_index} mse_pass={mse_pass_count} "
                  f"skipped_hr={skipped_hr} no_hit")
        return None

    def scan_single_tmpl(candidate_windows: list[tuple[int, int]], tmpl):
        use_color = tmpl.is_color and color_blurred is not None

        for (y0, y1) in candidate_windows:
            win_h = y1 - y0
            win_hr = win_h / h

            # hrフィルタ
            scale_to_page = w / float(tmpl.w)
            expected_hr = (tmpl.h * scale_to_page) / h
            if abs(win_hr - expected_hr) > hr_margin:
                continue

            # グレーMSEで先行フィルタ
            gray_patch, gray_resized = get_patch_and_resized(tmpl,
                                                             y0,
                                                             y1,
                                                             use_color=False)
            if gray_patch is None:
                continue
            m_gray = mse(gray_patch, gray_resized)
            if m_gray > mse_prefilter:
                continue

            if use_color:
                patch, resized = get_patch_and_resized(tmpl,
                                                       y0,
                                                       y1,
                                                       use_color=True)
                if patch is None:
                    continue
                m1 = mse(patch, resized)
                s1 = ssim(patch, resized, data_range=1.0, channel_axis=2)
            else:
                s1 = ssim(gray_patch, gray_resized, data_range=1.0)
                m1 = m_gray

            region_h = y1 - y0
            scale = w / float(tmpl.w)
            target_h = int(tmpl.h * scale)
            v_offset = (region_h - target_h) // 2
            gy0 = y0 + v_offset
            gy1 = gy0 + target_h

            if s1 >= config.ssim_threshold and m1 <= config.mse_threshold:
                return (s1, m1, orient, 0, gy0, region_w_global, gy1, tmpl)

        return None

    results: list[tuple[str, int, int, int, int]] = []

    if orient == "portrait":
        bottom_windows = build_candidate_windows("bottom")
        hit_bottom = scan(bottom_windows)
        if hit_bottom is not None:
            _, m_hit, o, x0, y0, x1, y1, hit_tmpl = hit_bottom
            results.append((o, x0, y0, x1, y1))
            increment_hit_count(hit_tmpl.name)

            center_windows = build_candidate_windows("center")
            hit_center = scan_single_tmpl(center_windows, hit_tmpl)
            if hit_center is not None:
                _, m_hit2, o2, x02, y02, x12, y12, _ = hit_center
                if not (abs(y02 - y0) < 5 and abs(y12 - y1) < 5):
                    results.append((o2, x02, y02, x12, y12))

        else:
            center_windows = build_candidate_windows("center")
            if DEBUG_DETECT:
                print(
                    f"[center-scan] page={page_index} windows={len(center_windows)} "
                    f"mse_prefilter={mse_prefilter:.4f}")
            hit_center = scan(center_windows)
            if hit_center is not None:
                _, m_hit2, o2, x02, y02, x12, y12, hit_tmpl2 = hit_center
                results.append((o2, x02, y02, x12, y12))
                increment_hit_count(hit_tmpl2.name)

    else:
        windows = build_candidate_windows("all")
        hit = scan(windows)
        if hit is not None:
            _, _, o, x0, y0, x1, y1, hit_tmpl = hit
            results.append((o, x0, y0, x1, y1))
            increment_hit_count(hit_tmpl.name)

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
                print(f"[detect] page={page_index} HIT orient={o} "
                      f"rect=({x0},{y0})-({x1},{y1})")

    return results


# ============================================
# 白塗り＋自社帯追加
# ============================================


def apply_band(page_img, det, own_band_path):
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

    band = Image.open(own_band_path).convert("RGBA")
    band = band.resize((band_w, band_h), Image.LANCZOS)

    page.paste(band, (x0, y0), band)

    return page.convert("RGB")


# ============================================
# PDF処理（1ファイル分）
# ============================================


def process_pdf_single(input_path, own_band_path, page_offset=0):
    doc = fitz.open(input_path)
    pages_out = []
    missed = []

    for i, page in enumerate(doc):
        rect = page.rect
        scale = min(MAX_PX / rect.width, MAX_PX / rect.height)

        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        g_blurred = to_gray_blurred(img)
        c_blurred = to_color_blurred(img)
        dets = detect_band(g_blurred, c_blurred, page_index=i + 1)

        if dets:
            for det in dets:
                img = apply_band(img, det, own_band_path)
        else:
            missed.append({
                "page_index": page_offset + i,
                "page_label": f"{page_offset + i + 1}ページ目",
            })

        with progress_lock:
            progress["current"] += 1

        pages_out.append(img)

    return pages_out, missed


# ============================================
# Flask
# ============================================

app = Flask(__name__)
app.secret_key = "secretkey"

progress_lock = Lock()
progress = {
    "status": "idle",
    "current": 0,
    "total": 0,
    "job_id": None,
    "error": None,
    "download": None,
    "missed_pages": [],
}


def render_page_to_pil(doc,
                       page_index: int,
                       max_px: int = None) -> Image.Image:
    if max_px is None:
        max_px = MAX_PX
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
        img = render_page_to_pil(doc, page_index)

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
    use_color = request.form.get("use_color") == "1"

    if not pdf_filename or not pdf_id:
        flash("情報が不足しています")
        return redirect(url_for("bands"))

    pdf_path = os.path.join(DIR_UPLOAD, pdf_filename)

    try:
        doc = fitz.open(pdf_path)
        img = render_page_to_pil(doc, page_index)
        w, h = img.size

        y0 = int(h * min(max(y0_ratio, 0.0), 1.0))
        y1 = int(h * min(max(y1_ratio, 0.0), 1.0))
        if y1 <= y0:
            flash("帯の範囲が不正です（下側が上側以下になっています）")
            return redirect(url_for("bands"))

        band_img = img.crop((0, y0, w, y1))

        # 1200pxにリサイズして保存（rawで保存、ぼかしなし）
        target_width = 1200
        if w != target_width:
            new_h = int((y1 - y0) * (target_width / w))
            band_img = band_img.resize((target_width, new_h), Image.LANCZOS)

        base = os.path.splitext(os.path.basename(template_name))[0]
        if use_color:
            out_name = f"{base}_color_{pdf_id}.png"
        else:
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
    other_band_files = os.listdir(DIR_TEMPLATES)
    own_bands = list_own_bands()
    return render_template("index.html",
                           templates=other_band_files,
                           own_bands=own_bands)


@app.route("/process", methods=["POST"])
def handle_pdf():
    pdf_files = request.files.getlist("pdf_file")
    own_band_name = request.form.get("own_band", "")

    valid_pdfs = [
        f for f in pdf_files
        if f.filename and f.filename.lower().endswith(".pdf")
    ]
    if not valid_pdfs:
        flash("PDFをアップしてください")
        return redirect(url_for("index"))

    own_band_path = get_own_band_path(own_band_name)
    if not os.path.isfile(own_band_path):
        flash("自社帯が見つかりません。band_default.pngを確認してください。")
        return redirect(url_for("index"))

    clear_old_files()

    job_id = str(uuid.uuid4())
    out_path = os.path.join(DIR_OUTPUT, job_id + "_out.pdf")

    in_paths = []
    for i, pdf in enumerate(valid_pdfs):
        in_path = os.path.join(DIR_UPLOAD, f"{job_id}_{i}.pdf")
        pdf.save(in_path)
        in_paths.append(in_path)

    total_pages = 0
    for p in in_paths:
        try:
            doc = fitz.open(p)
            total_pages += len(doc)
            doc.close()
        except Exception:
            pass

    with progress_lock:
        progress["status"] = "running"
        progress["current"] = 0
        progress["total"] = total_pages
        progress["job_id"] = job_id
        progress["error"] = None
        progress["download"] = None
        progress["missed_pages"] = []

    def worker():
        try:
            db.load()
            all_pages = []
            all_missed = []
            page_offset = 0

            for in_path in in_paths:
                pages, missed = process_pdf_single(in_path, own_band_path,
                                                   page_offset)
                all_pages.extend(pages)
                all_missed.extend(missed)
                page_offset += len(pages)

            if not all_pages:
                raise Exception("処理結果が空です")

            thumb_dir = os.path.join(DIR_OUTPUT, job_id + "_thumbs")
            os.makedirs(thumb_dir, exist_ok=True)
            for m in all_missed:
                idx = m["page_index"]
                thumb = all_pages[idx].copy()
                thumb.thumbnail((300, 400))
                thumb.save(os.path.join(thumb_dir, f"{idx}.jpg"))

            first, *rest = all_pages
            # ★ quality=85でJPEG圧縮: ファイルサイズ1/5〜1/8
            first.save(out_path, save_all=True, append_images=rest, quality=85)

            with progress_lock:
                progress["status"] = "done"
                progress["download"] = os.path.basename(out_path)
                progress["missed_pages"] = all_missed

        except Exception as e:
            with progress_lock:
                progress["status"] = "error"
                progress["error"] = str(e)

    Thread(target=worker, daemon=True).start()

    return render_template("processing.html", job_id=job_id)


@app.route("/bands")
def bands():
    other_bands = [
        f for f in os.listdir(DIR_TEMPLATES)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    own_bands = list_own_bands()
    return render_template("bands.html",
                           templates=other_bands,
                           own_bands=own_bands)


@app.route("/band_templates/<path:filename>")
def band_template_file(filename):
    return send_from_directory(DIR_TEMPLATES, filename)


@app.route("/own_bands/<path:filename>")
def own_band_file(filename):
    return send_from_directory(DIR_OWN_BANDS, filename)


@app.route("/own_bands/upload", methods=["POST"])
def upload_own_band():
    if "own_band_file" not in request.files:
        flash("ファイルがありません")
        return redirect(url_for("bands"))
    f = request.files["own_band_file"]
    if f.filename == "":
        flash("ファイル未選択")
        return redirect(url_for("bands"))
    save_path = os.path.join(DIR_OWN_BANDS, os.path.basename(f.filename))
    f.save(save_path)
    flash(f"自社帯「{f.filename}」を登録しました")
    return redirect(url_for("bands"))


@app.route("/own_bands/delete", methods=["POST"])
def delete_own_band():
    filename = request.form.get("filename")
    if not filename:
        flash("ファイル名が指定されていません")
        return redirect(url_for("bands"))
    safe_name = os.path.basename(filename)
    path = os.path.join(DIR_OWN_BANDS, safe_name)
    if os.path.isfile(path):
        os.remove(path)
        flash(f"「{safe_name}」を削除しました")
    else:
        flash("ファイルが見つかりませんでした")
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


@app.route("/bands/delete_bulk", methods=["POST"])
def delete_band_bulk():
    filenames = request.form.getlist("filenames")
    if not filenames:
        flash("削除するファイルが選択されていません")
        return redirect(url_for("bands"))

    deleted = []
    errors = []
    for filename in filenames:
        safe_name = os.path.basename(filename)
        path = os.path.join(DIR_TEMPLATES, safe_name)
        if os.path.isfile(path):
            try:
                os.remove(path)
                deleted.append(safe_name)
            except Exception as e:
                errors.append(f"{safe_name}: {e}")
        else:
            errors.append(f"{safe_name}: 見つかりませんでした")

    if deleted:
        flash(f"{len(deleted)}件を削除しました")
    for e in errors:
        flash(f"エラー: {e}")

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


@app.route("/bands/toggle_color", methods=["POST"])
def toggle_color_band():
    filename = request.form.get("filename")
    if not filename:
        flash("ファイル名が指定されていません")
        return redirect(url_for("bands"))

    safe_name = os.path.basename(filename)
    old_path = os.path.join(DIR_TEMPLATES, safe_name)

    if not os.path.isfile(old_path):
        flash("ファイルが見つかりませんでした")
        return redirect(url_for("bands"))

    root, ext = os.path.splitext(safe_name)

    if "_color" in root:
        new_root = root.replace("_color", "", 1)
        action = "グレースケール"
    else:
        uuid_pattern = r'^(.+?)(_[0-9a-f\-]{36})$'
        m = re.match(uuid_pattern, root)
        if m:
            new_root = m.group(1) + "_color" + m.group(2)
        else:
            new_root = root + "_color"
        action = "カラー"

    new_name = new_root + ext
    new_path = os.path.join(DIR_TEMPLATES, new_name)

    if os.path.exists(new_path) and new_path != old_path:
        flash("同じ名前のファイルが既に存在します")
        return redirect(url_for("bands"))

    os.rename(old_path, new_path)
    flash(f"「{safe_name}」を{action}モードに変更しました → 「{new_name}」")

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
            "missed_pages": progress.get("missed_pages", []),
        }


@app.route("/download/<job_id>")
def download(job_id):
    out_path = os.path.join(DIR_OUTPUT, job_id + "_out.pdf")
    if not os.path.exists(out_path):
        return "not ready", 404
    return send_file(out_path, as_attachment=True, download_name="output.pdf")


@app.route("/thumb/<job_id>/<int:page_index>")
def serve_thumb(job_id, page_index):
    thumb_dir = os.path.join(DIR_OUTPUT, job_id + "_thumbs")
    filename = f"{page_index}.jpg"
    path = os.path.join(thumb_dir, filename)
    if not os.path.exists(path):
        return "not found", 404
    return send_file(path, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
