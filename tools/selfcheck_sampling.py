"""学習データの自動抜き取りの自己チェック（デプロイ前に流す）。

確認すること:
  - 抜き取る割合が設定どおりか
  - 何度ダウンロードしても割合が上がらないか（抽選をやり直さない）
  - 出力から削除したページが混ざらないか
  - 直したページ（失敗例）と抜き取ったページ（成功例）の両方が入るか
  - 送信先が未設定なら何も記録しないか

使い方（リポジトリのルートで）:
    python tools/selfcheck_sampling.py
"""
import json
import os
import random
import re
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402
import training_data  # noqa: E402

training_data.is_configured = lambda: True          # 送信先が設定されている状態にする
captured = []
training_data.save_page = lambda **kw: captured.append(kw) or True
training_data.flush = lambda: True

fails = []


def eq(name, got, want):
    if got != want:
        fails.append(f"{name}: got={got!r} want={want!r}")
        print(f"  NG {name}: got={got!r} want={want!r}")
    else:
        print(f"  ok {name}: {got!r}")


def mkjob(n):
    job = "s-" + uuid.uuid4().hex[:8]
    out = app.DIR_OUTPUT
    for sub in ("_orig", "_pages", "_thumbs"):
        os.makedirs(os.path.join(out, job + sub), exist_ok=True)
    from PIL import Image
    img = Image.new("RGB", (400, 600), "black")
    meta = []
    for i in range(n):
        for sb in ("_orig", "_pages", "_thumbs"):
            img.save(os.path.join(out, job + sb, f"page_{i}.jpg"), quality=90)
        meta.append({"page_index": i, "page_label": f"{i + 1}ページ目", "missed": False,
                     "img_w": 400, "img_h": 600, "rotation": 0,
                     "detections": [{"cls": "band", "x0": 0, "y0": 480, "x1": 400, "y1": 520}],
                     "label_only": []})
    json.dump(meta, open(os.path.join(out, job + "_meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    open(os.path.join(out, job + "_band.txt"), "w", encoding="utf-8").write(
        os.path.join(app.DIR_OWN_BANDS, app.get_default_band()))
    return job


def cleanup(job):
    out = app.DIR_OUTPUT
    for sub in ("_orig", "_pages", "_thumbs", "_work"):
        shutil.rmtree(os.path.join(out, job + sub), ignore_errors=True)
    for suf in ("_meta.json", "_band.txt", "_confirmed.json", "_opts.json",
                "_order.json", "_deleted.json"):
        q = os.path.join(out, job + suf)
        if os.path.exists(q):
            os.remove(q)


def recorded_pages():
    return sorted(int(re.search(r"page_(\d+)", k["orig_path"]).group(1)) for k in captured)


c = app.app.test_client()
print(f"抜き取り率の設定: {app.TRAINING_SAMPLE_RATE * 100:.0f}%\n")

print("== 全体の割合が設定どおりになるか（10ページ×30回=300ページ） ==")
random.seed(3)
total = 0
for _ in range(30):
    job = mkjob(10)
    captured.clear()
    c.get(f"/download/{job}")
    total += len(captured)
    cleanup(job)
print(f"  記録数: {total} / 300ページ = {total / 300 * 100:.0f}%")
eq("15〜27%の範囲に収まる", bool(45 <= total <= 81), True)

print("\n== 何度ダウンロードしても割合が上がらないか ==")
random.seed(7)
job = mkjob(20)
captured.clear()
counts = []
for _ in range(5):
    before = len(captured)
    c.get(f"/download/{job}")
    counts.append(len(captured) - before)
print(f"  1〜5回目の記録数: {counts}")
eq("2回目以降は0件（抽選し直さない）", sum(counts[1:]), 0)
print(f"  累計 {sum(counts)}/20ページ")
cleanup(job)

print("\n== 削除したページが混ざらないか（率100%にして厳しく確認） ==")
app.TRAINING_SAMPLE_RATE = 1.0
job = mkjob(10)
captured.clear()
c.post(f"/set_deleted/{job}", json={"deleted": [0, 1, 2, 3, 4]})
c.get(f"/download/{job}")
eq("残した5ページだけ記録", recorded_pages(), [5, 6, 7, 8, 9])
cleanup(job)

print("\n== 直したページは自動で、残りは抽選（両方入るか） ==")
job = mkjob(6)
captured.clear()
# 1ページ目だけ枠を直す（＝失敗例として自動で記録される）
r = c.post(f"/edit_page/{job}/0", json={"rects": [
    {"cls": "band", "x0_ratio": 0, "y0_ratio": 0.7, "x1_ratio": 1, "y1_ratio": 0.8}]})
eq("貼り直し成功", r.get_json().get("ok"), True)
eq("直した時点で1件記録", len(captured), 1)
eq("それは修正扱い(edited=True)", captured[0]["edited"], True)
c.get(f"/download/{job}")
eq("ダウンロードで残り5ページも記録（率100%のため）", len(captured), 6)
kinds = [k["edited"] for k in captured]
eq("修正1件 + 確認5件", (kinds.count(True), kinds.count(False)), (1, 5))
eq("直したページを二重に送っていない", recorded_pages(), [0, 1, 2, 3, 4, 5])
cleanup(job)
app.TRAINING_SAMPLE_RATE = 0.2

print("\n== 送信先が未設定なら何もしないか ==")
training_data.is_configured = lambda: False
job = mkjob(10)
captured.clear()
c.get(f"/download/{job}")
eq("記録なし", len(captured), 0)
cleanup(job)
training_data.is_configured = lambda: True

print()
if fails:
    print(f"NG が {len(fails)} 件:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("すべてOK")
