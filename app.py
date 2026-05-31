"""帯付けくん — マイソクの他社帯を白塗りして自社帯に差し替えるWebツール。

検出は detect.py（文字ベース＋最下部カラーバー、テンプレ・SSIM不要）に委譲。
社内数人での利用を想定し、1プロセス+スレッドで安定動作させる。
"""
import os
import io
import uuid
import json
import time
import shutil
import threading

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
from PIL import Image, ImageDraw
import fitz  # PyMuPDF

import detect

# ============================================
# 設定・パス
# ============================================

OUT_MAXPX = 1500          # 出力／検出用レンダリングの長辺ピクセル
THUMB_SIZE = (300, 424)   # レビュー用サムネイル
MAX_AGE_SECONDS = 3600    # 古い一時ファイルを掃除する閾値（1時間）

BASE = os.path.dirname(os.path.abspath(__file__))
DIR_OWN_BANDS = os.path.join(BASE, "own_bands")
DIR_UPLOAD = os.path.join(BASE, "uploads")
DIR_OUTPUT = os.path.join(BASE, "outputs")
OWN_BAND_DEFAULT = os.path.join(BASE, "band_default.png")

for d in (DIR_OWN_BANDS, DIR_UPLOAD, DIR_OUTPUT):
    os.makedirs(d, exist_ok=True)

ALLOWED_BAND_EXT = (".png", ".jpg", ".jpeg")


# ============================================
# 自社帯ユーティリティ
# ============================================

def list_own_bands():
    if not os.path.isdir(DIR_OWN_BANDS):
        return []
    files = [f for f in os.listdir(DIR_OWN_BANDS) if f.lower().endswith(ALLOWED_BAND_EXT)]
    return sorted(files)


def get_own_band_path(band_name=None):
    """指定された自社帯のパスを返す。無ければ先頭、それも無ければデフォルト。"""
    if band_name:
        path = os.path.join(DIR_OWN_BANDS, os.path.basename(band_name))
        if os.path.isfile(path):
            return path
    files = list_own_bands()
    if files:
        return os.path.join(DIR_OWN_BANDS, files[0])
    return OWN_BAND_DEFAULT


def clear_old_files(current_job_id=None):
    """current_job_id以外で、一定時間より古い一時ファイルだけ削除する。"""
    now = time.time()
    for base_dir in (DIR_UPLOAD, DIR_OUTPUT):
        if not os.path.isdir(base_dir):
            continue
        for name in os.listdir(base_dir):
            if current_job_id and current_job_id in name:
                continue
            path = os.path.join(base_dir, name)
            try:
                if now - os.path.getmtime(path) < MAX_AGE_SECONDS:
                    continue
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)
            except Exception as e:
                print(f"[warn] 削除できませんでした {path}: {e}")


# ============================================
# レンダリング・帯付け
# ============================================

def render_page_to_pil(page, max_px=OUT_MAXPX):
    rect = page.rect
    scale = min(max_px / rect.width, max_px / rect.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img, scale


def _clamp_rect(rect, W, H):
    """rect=(x0,y0,x1,y1)をページ内に収める。無効なら None。"""
    x0, y0, x1, y1 = rect
    x0 = max(0, min(int(x0), W))
    x1 = max(0, min(int(x1), W))
    y0 = max(0, min(int(y0), H))
    y1 = max(0, min(int(y1), H))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def apply_band(page_img, rect, own_band_path):
    """検出した他社帯(rect)を白塗りで消し、その範囲いっぱいに自社帯を貼る。

    rect=(x0,y0,x1,y1) ピクセル。他社帯は縦横比がバラバラなので合わせる意味がなく、
    検出範囲をしっかり覆って隠すことを最優先する。自社帯は範囲に合わせて引き伸ばす。
    """
    page = page_img.convert("RGBA")
    W, H = page.size
    c = _clamp_rect(rect, W, H)
    if c is None:
        return page_img
    x0, y0, x1, y1 = c

    draw = ImageDraw.Draw(page)
    draw.rectangle([(x0, y0), (x1, y1)], fill=(255, 255, 255, 255))

    band = Image.open(own_band_path).convert("RGBA").resize((x1 - x0, y1 - y0), Image.LANCZOS)
    page.paste(band, (x0, y0), band)
    return page.convert("RGB")


def apply_whiteout(page_img, rect):
    """ロゴ・地図などを白塗りで消すだけ（自社帯は貼らない）。rect=(x0,y0,x1,y1)。"""
    page = page_img.convert("RGB")
    W, H = page.size
    c = _clamp_rect(rect, W, H)
    if c is None:
        return page_img
    x0, y0, x1, y1 = c
    draw = ImageDraw.Draw(page)
    draw.rectangle([(x0, y0), (x1, y1)], fill=(255, 255, 255))
    return page


def rect_for_object(o, W, H):
    """検出物から白塗り矩形と処理種別を決める。

    band(会社情報帯)は全幅の横帯なので左右いっぱいに広げ、上下に少し余白を足して
    取りこぼし（枠線・1行分）まで確実に隠す。logo/mapは局所的なので検出枠＋わずかな余白。
    返り値: ((x0,y0,x1,y1), mode)  mode='band' or 'white'
    """
    cls = o.get("cls", "band")
    if cls == "band":
        pad = int(H * 0.006)
        rect = (0, o["y0"] - pad, W, o["y1"] + pad)
        return rect, "band"
    pad = int(H * 0.004)
    rect = (o["x0"] - pad, o["y0"] - pad, o["x1"] + pad, o["y1"] + pad)
    return rect, "white"


# ============================================
# ジョブ・進捗管理（メモリ上、1プロセス前提）
# ============================================

progress_lock = threading.Lock()
all_progress = {}  # job_id -> {status, current, total, error, missed_pages}


def make_progress_entry():
    return {"status": "idle", "current": 0, "total": 0, "error": None, "missed_pages": []}


def _set_progress(job_id, **kw):
    with progress_lock:
        p = all_progress.get(job_id)
        if p:
            p.update(kw)


def _bump_progress(job_id):
    with progress_lock:
        p = all_progress.get(job_id)
        if p:
            p["current"] += 1


def process_job(job_id, in_paths, own_band_path, whiteout_map=False):
    """1ジョブ分のPDF群を処理する（バックグラウンドスレッドで実行）。

    whiteout_map: 案内図(mapクラス)も白塗りするか。Falseなら案内図は残す。
    """
    pages_dir = os.path.join(DIR_OUTPUT, job_id + "_pages")
    thumb_dir = os.path.join(DIR_OUTPUT, job_id + "_thumbs")
    orig_dir = os.path.join(DIR_OUTPUT, job_id + "_orig")
    for d in (pages_dir, thumb_dir, orig_dir):
        os.makedirs(d, exist_ok=True)

    try:
        # 総ページ数を先に数える
        total = 0
        for p in in_paths:
            try:
                doc = fitz.open(p)
                total += len(doc)
                doc.close()
            except Exception as e:
                raise RuntimeError(f"PDFを開けませんでした（{os.path.basename(p)}）: {e}")
        _set_progress(job_id, total=total)

        page_results = []
        missed = []
        abs_idx = 0

        for in_path in in_paths:
            doc = fitz.open(in_path)
            try:
                for page in doc:
                    img, scale = render_page_to_pil(page)
                    img_w, img_h = img.size
                    objs = detect.detect_objects(img)

                    # 帯付け前の元画像を保存（手動からのやり直し用）
                    img.save(os.path.join(orig_dir, f"page_{abs_idx}.jpg"), quality=90)

                    detections = []
                    for o in objs:
                        # 案内図はオプションがオフなら白塗りしない（区画図は元々学習対象外）
                        if o["cls"] == "map" and not whiteout_map:
                            continue
                        rect, mode = rect_for_object(o, img_w, img_h)
                        if mode == "band":
                            img = apply_band(img, rect, own_band_path)
                        else:
                            img = apply_whiteout(img, rect)
                        detections.append({
                            "cls": o["cls"],
                            "x0": rect[0], "y0": rect[1], "x1": rect[2], "y1": rect[3],
                        })

                    img.save(os.path.join(pages_dir, f"page_{abs_idx}.jpg"), quality=90)
                    thumb = img.copy()
                    thumb.thumbnail(THUMB_SIZE)
                    thumb.save(os.path.join(thumb_dir, f"page_{abs_idx}.jpg"), quality=85)

                    # 「帯」が1つも置けなかったページは赤表示（手動補完を促す）
                    is_missed = not any(d["cls"] == "band" for d in detections)
                    entry = {
                        "page_index": abs_idx,
                        "page_label": f"{abs_idx + 1}ページ目",
                        "missed": is_missed,
                        "img_h": img_h,
                        "detections": detections,
                    }
                    page_results.append(entry)
                    if is_missed:
                        missed.append({"page_index": abs_idx, "page_label": entry["page_label"]})

                    abs_idx += 1
                    _bump_progress(job_id)
            finally:
                doc.close()

        if not page_results:
            raise RuntimeError("処理できるページがありませんでした")

        meta_path = os.path.join(DIR_OUTPUT, job_id + "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(page_results, f, ensure_ascii=False)

        _set_progress(job_id, status="done", missed_pages=missed)

    except Exception as e:
        print(f"[error] job {job_id}: {e}")
        _set_progress(job_id, status="error", error=str(e))


# ============================================
# Flask
# ============================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "obiduke-kun-secret")
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300MBまで


@app.route("/")
def index():
    return render_template("index.html", own_bands=list_own_bands())


@app.route("/process", methods=["POST"])
def handle_pdf():
    pdf_files = request.files.getlist("pdf_file")
    own_band_name = request.form.get("own_band", "")
    whiteout_map = request.form.get("whiteout_map", "0") == "1"

    valid_pdfs = [f for f in pdf_files if f.filename and f.filename.lower().endswith(".pdf")]
    if not valid_pdfs:
        return jsonify({"error": "PDFを1件以上選択してください"}), 400

    own_band_path = get_own_band_path(own_band_name)
    if not os.path.isfile(own_band_path):
        return jsonify({"error": "自社帯が登録されていません。先に自社帯を登録してください。"}), 400

    job_id = str(uuid.uuid4())
    clear_old_files(current_job_id=job_id)

    in_paths = []
    try:
        for i, pdf in enumerate(valid_pdfs):
            in_path = os.path.join(DIR_UPLOAD, f"{job_id}_{i}.pdf")
            pdf.save(in_path)
            in_paths.append(in_path)
    except Exception as e:
        return jsonify({"error": f"アップロードの保存に失敗しました: {e}"}), 500

    with progress_lock:
        all_progress[job_id] = make_progress_entry()
        all_progress[job_id]["status"] = "running"

    threading.Thread(
        target=process_job, args=(job_id, in_paths, own_band_path, whiteout_map), daemon=True
    ).start()

    return jsonify({"job_id": job_id, "redirect": f"/processing/{job_id}"})


@app.route("/processing/<job_id>")
def processing_page(job_id):
    return render_template("processing.html", job_id=job_id)


@app.route("/progress")
def get_progress():
    job_id = request.args.get("job_id")
    with progress_lock:
        p = all_progress.get(job_id)
    if not p:
        return jsonify({
            "status": "not_found", "current": 0, "total": 0, "job_id": job_id,
            "error": "ジョブが見つかりません", "missed_pages": [], "review_url": None,
        })
    return jsonify({
        "status": p["status"],
        "current": p["current"],
        "total": p["total"],
        "job_id": job_id,
        "error": p.get("error"),
        "missed_pages": p.get("missed_pages", []),
        "review_url": f"/review/{job_id}" if p["status"] == "done" else None,
    })


def _order_path(job_id):
    return os.path.join(DIR_OUTPUT, job_id + "_order.json")


def _load_order(job_id, default_indices):
    """保存された並び順(page_indexのリスト)を返す。無ければ元の順序。

    保存順のうち実在するページだけを採用し、漏れたページは末尾に足す。
    """
    valid = list(default_indices)
    path = _order_path(job_id)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
            ordered = [i for i in saved if i in valid]
            ordered += [i for i in valid if i not in ordered]
            return ordered
        except Exception:
            pass
    return valid


@app.route("/review/<job_id>")
def review(job_id):
    meta_path = os.path.join(DIR_OUTPUT, job_id + "_meta.json")
    if not os.path.exists(meta_path):
        return "ジョブが見つかりません", 404
    with open(meta_path, encoding="utf-8") as f:
        page_results = json.load(f)

    # 保存された並び順に従って1列に並べる（未検出/検出済は色分けで区別）
    default_order = [p["page_index"] for p in page_results]
    order = _load_order(job_id, default_order)
    by_index = {p["page_index"]: p for p in page_results}
    pages = [by_index[i] for i in order if i in by_index]

    missed_count = sum(1 for p in page_results if p["missed"])
    ok_count = len(page_results) - missed_count

    # 各ページの現在の帯位置を比率で渡す（モーダルで初期表示し、上下調整しやすくする）
    page_data = {}
    for p in page_results:
        h = p.get("img_h") or 1
        page_data[p["page_index"]] = [
            [round(d["y0"] / h, 4), round(d["y1"] / h, 4)]
            for d in p.get("detections", [])
            if d.get("cls", "band") == "band"  # 手動ドラッグは帯のみ初期表示
        ]

    return render_template(
        "review.html",
        job_id=job_id,
        pages=pages,
        page_data=page_data,
        missed_count=missed_count,
        ok_count=ok_count,
        total=len(page_results),
    )


@app.route("/reorder/<job_id>", methods=["POST"])
def reorder(job_id):
    """ページの並び順を保存する。{order: [page_index, ...]} を受け取る。"""
    data = request.get_json(silent=True) or {}
    order = data.get("order")
    if not isinstance(order, list):
        return jsonify({"error": "順序が不正です"}), 400
    try:
        order = [int(i) for i in order]
    except (TypeError, ValueError):
        return jsonify({"error": "順序が不正です"}), 400
    try:
        with open(_order_path(job_id), "w", encoding="utf-8") as f:
            json.dump(order, f)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True})


@app.route("/page_img/<job_id>/<int:page_index>")
def page_img(job_id, page_index):
    path = os.path.join(DIR_OUTPUT, job_id + "_pages", f"page_{page_index}.jpg")
    if not os.path.exists(path):
        return "not found", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/thumb_all/<job_id>/<int:page_index>")
def thumb_all(job_id, page_index):
    path = os.path.join(DIR_OUTPUT, job_id + "_thumbs", f"page_{page_index}.jpg")
    if not os.path.exists(path):
        return "not found", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/manual_band/<job_id>/<int:page_index>", methods=["POST"])
def manual_band(job_id, page_index):
    """手動帯付け／位置調整: {y0_ratio, y1_ratio} を受け取りページ画像を作り直す。

    白塗りが重ならないよう、帯付け前の元画像(orig)から再構成する。
    手動範囲と重ならない既存の帯は残し、重なる帯は手動指定で置き換える。
    """
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "band")  # 'band'=帯を貼る / 'whiteout'=白塗りのみ
    try:
        y0_ratio = float(data.get("y0_ratio", 0))
        y1_ratio = float(data.get("y1_ratio", 1))
        x0_ratio = float(data.get("x0_ratio", 0))
        x1_ratio = float(data.get("x1_ratio", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "範囲が不正です"}), 400
    if y1_ratio <= y0_ratio or x1_ratio <= x0_ratio:
        return jsonify({"error": "範囲が不正です"}), 400

    orig_path = os.path.join(DIR_OUTPUT, job_id + "_orig", f"page_{page_index}.jpg")
    page_path = os.path.join(DIR_OUTPUT, job_id + "_pages", f"page_{page_index}.jpg")
    if not os.path.exists(orig_path):
        return jsonify({"error": "元ページが見つかりません"}), 404

    own_band_path = get_own_band_path()
    if not os.path.isfile(own_band_path):
        return jsonify({"error": "自社帯が登録されていません"}), 400

    try:
        meta_path = os.path.join(DIR_OUTPUT, job_id + "_meta.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        page_meta = next((p for p in meta if p["page_index"] == page_index), None)

        img = Image.open(orig_path).convert("RGB")  # 帯付け前から再構成
        w, h = img.size
        ny0, ny1 = int(y0_ratio * h), int(y1_ratio * h)
        nx0, nx1 = int(x0_ratio * w), int(x1_ratio * w)

        if mode == "whiteout":
            # 白塗りのみ: 2D範囲をそのまま白塗り。既存検出は全て残す
            new_rect = {"cls": "manual", "x0": nx0, "y0": ny0, "x1": nx1, "y1": ny1}
            rects = list(page_meta.get("detections", [])) if page_meta else []
            rects.append(new_rect)
        else:
            # 帯を貼る: 全幅。縦に重なる既存の帯のみ置き換え、他の検出は残す
            new_rect = {"cls": "band", "x0": 0, "y0": ny0, "x1": w, "y1": ny1}
            rects = []
            if page_meta:
                for d in page_meta.get("detections", []):
                    # 縦に重なる既存の「帯」だけ除外（ロゴ・地図・手動白塗りは保持）
                    if d.get("cls", "band") == "band" and not (d["y1"] < ny0 or d["y0"] > ny1):
                        continue
                    rects.append(d)
            rects.append(new_rect)

        for d in rects:
            rect = (d["x0"], d["y0"], d["x1"], d["y1"])
            if d.get("cls", "band") == "band":
                img = apply_band(img, rect, own_band_path)
            else:
                img = apply_whiteout(img, rect)

        img.save(page_path, quality=90)
        thumb = img.copy()
        thumb.thumbnail(THUMB_SIZE)
        thumb.save(os.path.join(DIR_OUTPUT, job_id + "_thumbs", f"page_{page_index}.jpg"), quality=85)

        if page_meta is not None:
            # 帯が1つでもあれば取りこぼし解消とみなす
            page_meta["missed"] = not any(d.get("cls", "band") == "band" for d in rects)
            page_meta["detections"] = rects
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"ok": True, "ts": int(time.time())})


@app.route("/download/<job_id>")
def download(job_id):
    """全ページ画像からPDFを生成して返す。"""
    meta_path = os.path.join(DIR_OUTPUT, job_id + "_meta.json")
    pages_dir = os.path.join(DIR_OUTPUT, job_id + "_pages")
    if not os.path.exists(meta_path):
        return "まだ準備できていません", 404

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # 保存された並び順に従って結合する
    default_order = [p["page_index"] for p in meta]
    order = _load_order(job_id, default_order)
    pages = []
    for idx in order:
        img_path = os.path.join(pages_dir, f"page_{idx}.jpg")
        if os.path.exists(img_path):
            pages.append(Image.open(img_path).convert("RGB"))
    if not pages:
        return "ページがありません", 404

    buf = io.BytesIO()
    first, *rest = pages
    first.save(buf, format="PDF", save_all=True, append_images=rest, quality=85)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="obiduke_output.pdf",
                     mimetype="application/pdf")


# ---------- 自社帯の管理 ----------

@app.route("/bands")
def bands():
    return render_template("bands.html", own_bands=list_own_bands())


@app.route("/own_bands/<path:filename>")
def own_band_file(filename):
    return send_from_directory(DIR_OWN_BANDS, filename)


@app.route("/own_bands/upload", methods=["POST"])
def upload_own_band():
    if "own_band_file" not in request.files:
        flash("ファイルがありません")
        return redirect(url_for("bands"))
    f = request.files["own_band_file"]
    if not f.filename:
        flash("ファイルが選択されていません")
        return redirect(url_for("bands"))
    if not f.filename.lower().endswith(ALLOWED_BAND_EXT):
        flash("PNG / JPG の画像を指定してください")
        return redirect(url_for("bands"))
    f.save(os.path.join(DIR_OWN_BANDS, os.path.basename(f.filename)))
    flash(f"自社帯「{f.filename}」を登録しました")
    return redirect(url_for("bands"))


@app.route("/own_bands/delete", methods=["POST"])
def delete_own_band():
    filename = request.form.get("filename")
    if not filename:
        return jsonify({"ok": False, "error": "ファイル名が指定されていません"}), 400
    path = os.path.join(DIR_OWN_BANDS, os.path.basename(filename))
    if os.path.isfile(path):
        os.remove(path)
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "ファイルが見つかりませんでした"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
