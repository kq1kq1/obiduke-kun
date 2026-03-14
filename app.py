print("=== Flask server started! ===")
import os
import uuid
import json
import re
from dataclasses import dataclass
from typing import Optional, List
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

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

BLUR_RADIUS = 3
MAX_PX = 1200  # 2200→1600: 処理速度約2倍・出力PDFサイズ約半分


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
    hr_filter_margin: float = 0.020  # 0.015→0.020: ±24pxまで許容（検出漏れ改善）


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
            f
            for f in os.listdir(DIR_OWN_BANDS)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if files:
            return os.path.join(DIR_OWN_BANDS, sorted(files)[0])
    return OWN_BAND_DEFAULT


def list_own_bands():
    if not os.path.isdir(DIR_OWN_BANDS):
        return []
    files = [
        f
        for f in os.listdir(DIR_OWN_BANDS)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    return sorted(files)


def clear_old_files(current_job_id: str = None, max_age_seconds: int = 3600):
    """古いjob_idのファイル・フォルダのみ削除する。
    current_job_idのファイルは削除しない。
    max_age_seconds（デフォルト1時間）より古いものだけ削除。
    """
    import shutil
    import time

    now = time.time()

    # uploadsフォルダ: current_job_id以外の古いファイルを削除
    if os.path.isdir(DIR_UPLOAD):
        for name in os.listdir(DIR_UPLOAD):
            path = os.path.join(DIR_UPLOAD, name)
            # current_job_idを含むファイルは削除しない
            if current_job_id and current_job_id in name:
                continue
            try:
                age = now - os.path.getmtime(path)
                if age < max_age_seconds:
                    continue
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(f"[warn] {path} を削除できませんでした: {e}")

    # outputsフォルダ: current_job_id以外の古いファイル・フォルダを削除
    if os.path.isdir(DIR_OUTPUT):
        for name in os.listdir(DIR_OUTPUT):
            path = os.path.join(DIR_OUTPUT, name)
            # current_job_idを含むファイルは削除しない
            if current_job_id and current_job_id in name:
                continue
            try:
                age = now - os.path.getmtime(path)
                if age < max_age_seconds:
                    continue
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
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


def increment_hit_count(tmpl_name: str, ssim_score: float = 0.0):
    """ヒット数と最高スコアを記録する"""
    with hit_count_lock:
        counts = load_hit_counts()
        entry = counts.get(tmpl_name)
        # ★ FIX: numpy.float32 → Python float に変換してJSON serializable にする
        ssim_score = float(ssim_score)
        if isinstance(entry, dict):
            entry["count"] = entry.get("count", 0) + 1
            entry["best_ssim"] = float(max(entry.get("best_ssim", 0.0), ssim_score))
        else:
            # 旧形式(int)または未登録
            old_count = entry if isinstance(entry, int) else 0
            entry = {"count": old_count + 1, "best_ssim": ssim_score}
        counts[tmpl_name] = entry
        save_hit_counts(counts)


def get_dynamic_threshold(tmpl_name: str, base_threshold: float) -> float:
    """テンプレごとの実績スコアから動的閾値を返す。
    実績best_ssim × 0.80 が base_threshold より高ければそちらを使う。
    実績がない場合はbase_thresholdをそのまま返す。
    """
    counts = load_hit_counts()
    entry = counts.get(tmpl_name)
    if isinstance(entry, dict):
        best = entry.get("best_ssim", 0.0)
        if best >= 0.80:  # 十分な実績がある場合のみ動的閾値を適用
            dynamic = best * 0.80
            if dynamic > base_threshold:
                return dynamic
    return base_threshold


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
    return float(np.mean((a - b) ** 2))


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
        # アスペクト比から縦長/横長ページどちらのテンプレか推定
        # テンプレ自体は帯なので横長（w >> h）。ページ向きとは無関係に同じ幅で使う。
        # hr（高さ比率）をキャッシュしておく（ページサイズが変わるので動的計算は別途）
        self.aspect = self.w / max(self.h, 1)  # 大きいほど横に細い帯


# リサイズキャッシュ: テンプレ×ウィンドウサイズ → リサイズ済み配列
# db.load()時にクリアされる（並列処理対応のためロック付き）
_resized_cache: dict[tuple, np.ndarray] = {}
_resized_cache_lock = Lock()


class TemplateDB:
    def __init__(self):
        self.items: List[BandTemplate] = []

    def load(self):
        self.items.clear()
        _resized_cache.clear()  # テンプレ更新時にキャッシュをリセット

        files = [
            f
            for f in os.listdir(DIR_TEMPLATES)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # ヒット頻度順にソート（多い順）
        # counts値はdict形式{"count":N,"best_ssim":X}またはint(旧形式)の両方に対応
        counts = load_hit_counts()

        def get_count(f):
            v = counts.get(f, 0)
            if isinstance(v, dict):
                return v.get("count", 0)
            return int(v) if isinstance(v, (int, float)) else 0

        files.sort(key=get_count, reverse=True)

        for f in files:
            path = os.path.join(DIR_TEMPLATES, f)
            gray = load_gray_raw(path)
            if "_color" in os.path.splitext(f)[0]:
                color = load_color_raw(path)
            else:
                color = None
            self.items.append(BandTemplate(f, gray, color))

        color_count = sum(1 for t in self.items if t.is_color)
        print(f"[templates] loaded={len(self.items)} color={color_count}")


db = TemplateDB()

# ============================================
# 帯検出
# ============================================


def detect_band(gray_blurred, color_blurred=None, page_index=None):
    """グレースケール＋ぼかし画像から帯を検出する"""
    h, w = gray_blurred.shape
    orient = "landscape" if w >= h else "portrait"

    height_min = 0.04  # 0.05→0.04: メルディアなど細い帯(hr≈4%)対応
    height_max = 0.18
    step_height = 0.01

    height_ratios: list[float] = []
    r = height_min
    while r <= height_max + 1e-6:
        height_ratios.append(r)
        r += step_height

    # resized_cache はグローバルのものを使用（テンプレ変更時はdb.load()でクリア）
    mse_prefilter = float(getattr(config, "mse_threshold", 0.06)) * float(
        getattr(config, "mse_prefilter_multiplier", 1.7)
    )
    hr_margin = float(getattr(config, "hr_filter_margin", 0.015))

    region_w_global = w
    best_any = None

    def build_candidate_windows(
        mode: str,
        offset_step_landscape: float = 0.015,  # landscape位置ステップ
        bottom_step_portrait: float = 0.005,
        center_step_portrait: float = 0.005,  # 0.005: 真ん中帯の検出精度確保
    ) -> list[tuple[int, int]]:
        candidate_windows: list[tuple[int, int]] = []

        if orient == "landscape":
            for hr in height_ratios:
                win_h = int(h * hr)
                if win_h < 8:
                    continue
                # 下部25%のみスキャン（下端からoffset 0〜0.25）
                offset = 0.0
                while offset <= 0.25 + 1e-6:
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
                    center_ratio = 0.38
                    while center_ratio <= 0.52 + 1e-6:
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
        with _resized_cache_lock:
            resized = _resized_cache.get(key)
        if resized is None:
            if use_color:
                resized = resize_color(tmpl.color, target_h, target_w)
            else:
                resized = resize_gray(tmpl.gray, target_h, target_w)
            with _resized_cache_lock:
                _resized_cache[key] = resized

        v_offset = (region_h - target_h) // 2
        patch = region[v_offset : v_offset + target_h, :]

        return patch, resized

    # テンプレごとの expected_hr を事前計算（スキャンのたびに再計算しない）
    tmpl_expected_hr = {}
    for _t in db.items:
        _scale = w / float(_t.w)
        tmpl_expected_hr[id(_t)] = (_t.h * _scale) / h

    def scan(candidate_windows: list[tuple[int, int]]):
        nonlocal best_any
        mse_pass_count = 0
        skipped_hr = 0

        for y0, y1 in candidate_windows:
            win_h = y1 - y0
            win_hr = win_h / h

            for tmpl in db.items:
                # ★ hrフィルタ: テンプレの期待高さ比率から外れたウィンドウをスキップ
                expected_hr = tmpl_expected_hr[id(tmpl)]
                if abs(win_hr - expected_hr) > hr_margin:
                    skipped_hr += 1
                    continue

                use_color = tmpl.is_color and color_blurred is not None

                # ★ グレーMSEで先行フィルタ（カラーテンプレも先にグレーで弾く）
                gray_patch, gray_resized = get_patch_and_resized(
                    tmpl, y0, y1, use_color=False
                )
                if gray_patch is None:
                    continue

                m_gray = mse(gray_patch, gray_resized)
                if m_gray > mse_prefilter:
                    continue

                mse_pass_count += 1

                if use_color:
                    patch, resized = get_patch_and_resized(tmpl, y0, y1, use_color=True)
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

                if (best_any is None) or (s1 > best_any[0]):
                    best_any = (s1, m1, orient, 0, gy0, region_w_global, gy1)

                dyn_thresh = get_dynamic_threshold(tmpl.name, config.ssim_threshold)
                if s1 >= dyn_thresh and m1 <= config.mse_threshold:
                    print(
                        f"[scan] page={page_index} HIT ssim={s1:.3f} thresh={dyn_thresh:.3f} tmpl={tmpl.name}"
                    )
                    return (s1, m1, orient, 0, gy0, region_w_global, gy1, tmpl)

                if s1 >= 0.92:
                    print(
                        f"[scan] page={page_index} HIT ssim={s1:.3f} tmpl={tmpl.name}"
                    )
                    return (s1, m1, orient, 0, gy0, region_w_global, gy1, tmpl)

        return None

    def scan_single_tmpl(candidate_windows: list[tuple[int, int]], tmpl):
        use_color = tmpl.is_color and color_blurred is not None

        for y0, y1 in candidate_windows:
            win_h = y1 - y0
            win_hr = win_h / h

            # hrフィルタ（事前計算済みを使用）
            expected_hr = tmpl_expected_hr[id(tmpl)]
            if abs(win_hr - expected_hr) > hr_margin:
                continue

            # グレーMSEで先行フィルタ
            gray_patch, gray_resized = get_patch_and_resized(
                tmpl, y0, y1, use_color=False
            )
            if gray_patch is None:
                continue
            m_gray = mse(gray_patch, gray_resized)
            if m_gray > mse_prefilter:
                continue

            if use_color:
                patch, resized = get_patch_and_resized(tmpl, y0, y1, use_color=True)
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

            dyn_thresh = get_dynamic_threshold(tmpl.name, config.ssim_threshold)
            if s1 >= dyn_thresh and m1 <= config.mse_threshold:
                return (s1, m1, orient, 0, gy0, region_w_global, gy1, tmpl)

        return None

    results: list[tuple[str, int, int, int, int]] = []

    if orient == "portrait":
        bottom_windows = build_candidate_windows("bottom")
        hit_bottom = scan(bottom_windows)
        if hit_bottom is not None:
            s_hit, m_hit, o, x0, y0, x1, y1, hit_tmpl = hit_bottom
            results.append((o, x0, y0, x1, y1))
            increment_hit_count(hit_tmpl.name, s_hit)

            center_windows = build_candidate_windows("center")
            hit_center = scan_single_tmpl(center_windows, hit_tmpl)
            if hit_center is not None:
                _, m_hit2, o2, x02, y02, x12, y12, _ = hit_center
                if not (abs(y02 - y0) < 5 and abs(y12 - y1) < 5):
                    results.append((o2, x02, y02, x12, y12))

        else:
            center_windows = build_candidate_windows("center")

            hit_center = scan(center_windows)
            if hit_center is not None:
                s_hit2, m_hit2, o2, x02, y02, x12, y12, hit_tmpl2 = hit_center
                results.append((o2, x02, y02, x12, y12))
                increment_hit_count(hit_tmpl2.name, s_hit2)

    else:
        windows = build_candidate_windows("all")
        hit = scan(windows)
        if hit is not None:
            s_hit3, _, o, x0, y0, x1, y1, hit_tmpl = hit
            results.append((o, x0, y0, x1, y1))
            increment_hit_count(hit_tmpl.name, s_hit3)

    if not results:
        print(f"[detect] page={page_index} NO band detected")
    else:
        for o, x0, y0, x1, y1 in results:
            print(
                f"[detect] page={page_index} HIT orient={o} rect=({x0},{y0})-({x1},{y1})"
            )

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


def process_single_page(img, own_band_path, abs_idx, job_id=None):
    """1ページ分の検出・帯付けを行う（並列処理用）"""
    g_blurred = to_gray_blurred(img)
    c_blurred = to_color_blurred(img)
    dets = detect_band(g_blurred, c_blurred, page_index=abs_idx + 1)

    orig_img = img.copy()

    if dets:
        det_info = []
        for det in dets:
            img = apply_band(img, det, own_band_path)
            _o, _x0, _y0, _x1, _y1 = det
            det_info.append(
                {"x0": _x0, "y0": _y0, "x1": _x1, "y1": _y1, "tmpl_name": ""}
            )
        page_result = {
            "page_index": abs_idx,
            "page_label": f"{abs_idx + 1}ページ目",
            "missed": False,
            "detections": det_info,
        }
        missed_entry = None
    else:
        missed_entry = {"page_index": abs_idx, "page_label": f"{abs_idx + 1}ページ目"}
        page_result = {
            "page_index": abs_idx,
            "page_label": f"{abs_idx + 1}ページ目",
            "missed": True,
            "detections": [],
        }

    # 進捗更新（job_idが渡された場合）
    if job_id:
        with progress_lock:
            if job_id in all_progress:
                all_progress[job_id]["current"] += 1

    return abs_idx, img, orig_img, missed_entry, page_result


def process_pdf_single(input_path, own_band_path, page_offset=0, job_id=None):
    doc = fitz.open(input_path)

    # 先にPDFから全ページ画像をメモリに展開（fitz操作はスレッドセーフでないため直列）
    page_imgs = []
    for i, page in enumerate(doc):
        rect = page.rect
        scale = min(MAX_PX / rect.width, MAX_PX / rect.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        page_imgs.append((page_offset + i, img))
    doc.close()

    # 並列で各ページを処理
    MAX_WORKERS = min(4, len(page_imgs))
    results = [None] * len(page_imgs)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_single_page, img, own_band_path, abs_idx, job_id
            ): local_i
            for local_i, (abs_idx, img) in enumerate(page_imgs)
        }
        for future in as_completed(futures):
            local_i = futures[future]
            results[local_i] = future.result()

    pages_out = []
    orig_pages_out = []
    missed = []
    page_results = []

    for abs_idx, img, orig_img, missed_entry, page_result in results:
        pages_out.append(img)
        orig_pages_out.append(orig_img)
        if missed_entry:
            missed.append(missed_entry)
        page_results.append(page_result)

    return pages_out, orig_pages_out, missed, page_results


# ============================================
# Flask
# ============================================

app = Flask(__name__)
app.secret_key = "secretkey"

progress_lock = Lock()
# job_idをキーにした進捗管理（複数人同時処理対応）
all_progress: dict[str, dict] = {}


def make_progress_entry():
    return {
        "status": "idle",
        "current": 0,
        "total": 0,
        "error": None,
        "missed_pages": [],
    }


def render_page_to_pil(doc, page_index: int, max_px: int = None) -> Image.Image:
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
    return render_template(
        "index.html", templates=other_band_files, own_bands=own_bands
    )


@app.route("/process", methods=["POST"])
def handle_pdf():
    pdf_files = request.files.getlist("pdf_file")
    own_band_name = request.form.get("own_band", "")

    valid_pdfs = [
        f for f in pdf_files if f.filename and f.filename.lower().endswith(".pdf")
    ]
    if not valid_pdfs:
        flash("PDFをアップしてください")
        return redirect(url_for("index"))

    own_band_path = get_own_band_path(own_band_name)
    if not os.path.isfile(own_band_path):
        flash("自社帯が見つかりません。band_default.pngを確認してください。")
        return redirect(url_for("index"))

    job_id = str(uuid.uuid4())
    clear_old_files(current_job_id=job_id)
    out_path = os.path.join(DIR_OUTPUT, job_id + "_out.pdf")

    in_paths = []
    for i, pdf in enumerate(valid_pdfs):
        in_path = os.path.join(DIR_UPLOAD, f"{job_id}_{i}.pdf")
        pdf.save(in_path)
        in_paths.append(in_path)

    # ページ数カウントの前にprogress登録→即座にレスポンスを返せるようにする
    with progress_lock:
        all_progress[job_id] = make_progress_entry()
        all_progress[job_id]["status"] = "running"
        all_progress[job_id]["total"] = 0  # worker内で更新

    def worker():
        try:
            db.load()

            # ページ数をworker内でカウントしてtotalを更新
            total_pages = 0
            for p in in_paths:
                try:
                    doc = fitz.open(p)
                    total_pages += len(doc)
                    doc.close()
                except Exception:
                    pass
            with progress_lock:
                all_progress[job_id]["total"] = total_pages

            all_pages = []
            all_orig_pages = []
            all_missed = []
            all_page_results = []
            page_offset = 0

            for in_path in in_paths:
                pages, orig_pages, missed, page_results = process_pdf_single(
                    in_path, own_band_path, page_offset, job_id=job_id
                )
                all_pages.extend(pages)
                all_orig_pages.extend(orig_pages)
                all_missed.extend(missed)
                all_page_results.extend(page_results)
                page_offset += len(pages)

            if not all_pages:
                raise Exception("処理結果が空です")

            # 全ページ画像を保存（フルサイズ・サムネイル・元画像）
            pages_dir = os.path.join(DIR_OUTPUT, job_id + "_pages")
            thumb_dir = os.path.join(DIR_OUTPUT, job_id + "_thumbs")
            orig_dir = os.path.join(DIR_OUTPUT, job_id + "_orig")
            os.makedirs(pages_dir, exist_ok=True)
            os.makedirs(thumb_dir, exist_ok=True)
            os.makedirs(orig_dir, exist_ok=True)

            for idx, img in enumerate(all_pages):
                img.save(os.path.join(pages_dir, f"page_{idx}.jpg"), quality=90)
                thumb = img.copy()
                thumb.thumbnail((300, 424))
                thumb.save(os.path.join(thumb_dir, f"page_{idx}.jpg"), quality=85)

            for idx, orig in enumerate(all_orig_pages):
                orig.save(os.path.join(orig_dir, f"page_{idx}.jpg"), quality=90)

            # メタデータ保存
            meta_path = os.path.join(DIR_OUTPUT, job_id + "_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(all_page_results, f, ensure_ascii=False)

            with progress_lock:
                all_progress[job_id]["status"] = "done"
                all_progress[job_id]["missed_pages"] = all_missed

        except Exception as e:
            with progress_lock:
                if job_id in all_progress:
                    all_progress[job_id]["status"] = "error"
                    all_progress[job_id]["error"] = str(e)

    Thread(target=worker, daemon=True).start()

    # job_idを即座に返す（フロント側でprocessing画面に遷移）
    return jsonify({"job_id": job_id, "redirect": f"/processing/{job_id}"})


@app.route("/bands")
def bands():
    other_bands = [
        f
        for f in os.listdir(DIR_TEMPLATES)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    own_bands = list_own_bands()
    return render_template("bands.html", templates=other_bands, own_bands=own_bands)


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
        return jsonify({"ok": False, "error": "ファイル名が指定されていません"}), 400
    safe_name = os.path.basename(filename)
    path = os.path.join(DIR_OWN_BANDS, safe_name)
    if os.path.isfile(path):
        os.remove(path)
        return jsonify({"ok": True})
    else:
        return jsonify({"ok": False, "error": "ファイルが見つかりませんでした"}), 404


@app.route("/bands/delete", methods=["POST"])
def delete_band():
    filename = request.form.get("filename")
    if not filename:
        return jsonify({"ok": False, "error": "ファイル名が指定されていません"}), 400

    safe_name = os.path.basename(filename)
    path = os.path.join(DIR_TEMPLATES, safe_name)

    if os.path.isfile(path):
        os.remove(path)
        db.load()
        return jsonify({"ok": True})
    else:
        return jsonify({"ok": False, "error": "ファイルが見つかりませんでした"}), 404


@app.route("/bands/delete_bulk", methods=["POST"])
def delete_band_bulk():
    filenames = request.form.getlist("filenames")
    if not filenames:
        return jsonify(
            {"ok": False, "error": "削除するファイルが選択されていません"}
        ), 400

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

    db.load()
    return jsonify({"ok": True, "deleted": deleted, "errors": errors})


@app.route("/bands/rename", methods=["POST"])
def rename_band():
    old_name = request.form.get("old_name")
    new_name = request.form.get("new_name")

    if not old_name or not new_name:
        return jsonify({"ok": False, "error": "名前が正しく指定されていません"}), 400

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
        return jsonify({"ok": False, "error": "元ファイルが見つかりませんでした"}), 404

    if os.path.exists(new_path) and new_path != old_path:
        return jsonify(
            {"ok": False, "error": "同じ名前のファイルが既に存在します"}
        ), 409

    os.rename(old_path, new_path)
    db.load()
    return jsonify({"ok": True, "new_name": new_name_final})


@app.route("/bands/toggle_color", methods=["POST"])
def toggle_color_band():
    filename = request.form.get("filename")
    if not filename:
        return jsonify({"ok": False, "error": "ファイル名が指定されていません"}), 400

    safe_name = os.path.basename(filename)
    old_path = os.path.join(DIR_TEMPLATES, safe_name)

    if not os.path.isfile(old_path):
        return jsonify({"ok": False, "error": "ファイルが見つかりませんでした"}), 404

    root, ext = os.path.splitext(safe_name)

    if "_color" in root:
        new_root = root.replace("_color", "", 1)
        action = "グレースケール"
    else:
        uuid_pattern = r"^(.+?)(_[0-9a-f\-]{36})$"
        m = re.match(uuid_pattern, root)
        if m:
            new_root = m.group(1) + "_color" + m.group(2)
        else:
            new_root = root + "_color"
        action = "カラー"

    new_name = new_root + ext
    new_path = os.path.join(DIR_TEMPLATES, new_name)

    if os.path.exists(new_path) and new_path != old_path:
        return jsonify(
            {"ok": False, "error": "同じ名前のファイルが既に存在します"}
        ), 409

    os.rename(old_path, new_path)
    db.load()
    return jsonify({"ok": True, "new_name": new_name, "is_color": "_color" in new_name})


@app.route("/progress")
def get_progress():
    from flask import request as freq

    job_id = freq.args.get("job_id")
    with progress_lock:
        p = all_progress.get(job_id)
    if not p:
        return {
            "status": "not_found",
            "current": 0,
            "total": 0,
            "job_id": job_id,
            "error": "job not found",
            "missed_pages": [],
            "review_url": None,
        }
    return {
        "status": p["status"],
        "current": p["current"],
        "total": p["total"],
        "job_id": job_id,
        "error": p.get("error"),
        "missed_pages": p.get("missed_pages", []),
        "review_url": f"/review/{job_id}" if p["status"] == "done" else None,
    }


@app.route("/processing/<job_id>")
def processing_page(job_id):
    return render_template("processing.html", job_id=job_id)


@app.route("/review/<job_id>")
def review(job_id):
    meta_path = os.path.join(DIR_OUTPUT, job_id + "_meta.json")
    if not os.path.exists(meta_path):
        return "job not found", 404
    with open(meta_path, encoding="utf-8") as f:
        page_results = json.load(f)
    missed_pages = [p for p in page_results if p["missed"]]
    ok_pages = [p for p in page_results if not p["missed"]]
    return render_template(
        "review.html",
        job_id=job_id,
        missed_pages=missed_pages,
        ok_pages=ok_pages,
        total=len(page_results),
    )


@app.route("/page_img/<job_id>/<int:page_index>")
def page_img(job_id, page_index):
    pages_dir = os.path.join(DIR_OUTPUT, job_id + "_pages")
    path = os.path.join(pages_dir, f"page_{page_index}.jpg")
    if not os.path.exists(path):
        return "not found", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/thumb_all/<job_id>/<int:page_index>")
def thumb_all(job_id, page_index):
    thumb_dir = os.path.join(DIR_OUTPUT, job_id + "_thumbs")
    path = os.path.join(thumb_dir, f"page_{page_index}.jpg")
    if not os.path.exists(path):
        return "not found", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/manual_band/<job_id>/<int:page_index>", methods=["POST"])
def manual_band(job_id, page_index):
    """手動帯付け: {y0_ratio, y1_ratio} を受け取ってページ画像を更新"""
    data = request.get_json()
    y0_ratio = float(data.get("y0_ratio", 0))
    y1_ratio = float(data.get("y1_ratio", 1))

    pages_dir = os.path.join(DIR_OUTPUT, job_id + "_pages")
    thumb_dir = os.path.join(DIR_OUTPUT, job_id + "_thumbs")
    page_path = os.path.join(pages_dir, f"page_{page_index}.jpg")
    if not os.path.exists(page_path):
        return jsonify({"error": "page not found"}), 404

    own_band_path = get_own_band_path()
    if not os.path.isfile(own_band_path):
        return jsonify({"error": "own band not found"}), 404

    img = Image.open(page_path).convert("RGB")
    w, h = img.size
    y0 = int(y0_ratio * h)
    y1 = int(y1_ratio * h)

    det = ("portrait", 0, y0, w, y1)  # apply_band expects (orient, x0, y0, x1, y1)

    img = apply_band(img, det, own_band_path)
    img.save(page_path, quality=90)

    thumb = img.copy()
    thumb.thumbnail((300, 424))
    thumb.save(os.path.join(thumb_dir, f"page_{page_index}.jpg"), quality=85)

    meta_path = os.path.join(DIR_OUTPUT, job_id + "_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    for p in meta:
        if p["page_index"] == page_index:
            p["missed"] = False
            p["detections"].append(
                {"x0": 0, "y0": y0, "x1": w, "y1": y1, "tmpl_name": "manual"}
            )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    import time

    return jsonify({"ok": True, "ts": int(time.time())})


@app.route("/save_band_from_review", methods=["POST"])
def save_band_from_review():
    """手動帯付け後の帯画像をテンプレートとして登録"""
    data = request.get_json()
    job_id = data.get("job_id")
    page_index = int(data.get("page_index", 0))
    y0_ratio = float(data.get("y0_ratio", 0))
    y1_ratio = float(data.get("y1_ratio", 1))
    tmpl_name = (data.get("template_name") or "band_template").strip()
    use_color = bool(data.get("use_color", False))

    # 元画像（帯付け前）から切り出す。なければ処理済み画像にフォールバック
    orig_dir = os.path.join(DIR_OUTPUT, job_id + "_orig")
    pages_dir = os.path.join(DIR_OUTPUT, job_id + "_pages")
    orig_path = os.path.join(orig_dir, f"page_{page_index}.jpg")
    page_path = os.path.join(pages_dir, f"page_{page_index}.jpg")

    src_path = orig_path if os.path.exists(orig_path) else page_path
    if not os.path.exists(src_path):
        return jsonify({"ok": False, "error": "page not found"}), 404

    try:
        img = Image.open(src_path).convert("RGB")
        w, h = img.size
        y0 = max(0, int(y0_ratio * h))
        y1 = min(h, int(y1_ratio * h))
        if y1 <= y0:
            return jsonify({"ok": False, "error": "無効な範囲です"}), 400

        band_img = img.crop((0, y0, w, y1))

        # 1200pxにリサイズ（既存テンプレと同じ仕様）
        target_w = 1200
        if w != target_w:
            new_h = max(1, int((y1 - y0) * (target_w / w)))
            band_img = band_img.resize((target_w, new_h), Image.LANCZOS)

        uid = str(uuid.uuid4())[:8]
        base = re.sub(r"[^\w\-]", "_", tmpl_name)
        if use_color:
            out_name = f"{base}_color_{uid}.png"
        else:
            out_name = f"{base}_{uid}.png"

        out_path = os.path.join(DIR_TEMPLATES, out_name)
        band_img.save(out_path)

        # テンプレキャッシュをリロード
        db.load()

        return jsonify({"ok": True, "name": out_name})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/download/<job_id>")
def download(job_id):
    """全ページ画像からPDFを生成してダウンロード"""
    meta_path = os.path.join(DIR_OUTPUT, job_id + "_meta.json")
    pages_dir = os.path.join(DIR_OUTPUT, job_id + "_pages")
    out_path = os.path.join(DIR_OUTPUT, job_id + "_out.pdf")

    if not os.path.exists(meta_path):
        return "not ready", 404

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    pages = []
    for p in sorted(meta, key=lambda x: x["page_index"]):
        img_path = os.path.join(pages_dir, f"page_{p['page_index']}.jpg")
        if os.path.exists(img_path):
            pages.append(Image.open(img_path).convert("RGB"))

    if not pages:
        return "no pages", 404

    first, *rest = pages
    first.save(out_path, save_all=True, append_images=rest, quality=85)

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
