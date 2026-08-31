"""回転まわりの自己チェック（デプロイ前に流す）。

いちばん大事なのは「学習データに送られる画像と座標が一致しているか」。
ここがズレると、間違った正解データが静かに混ざって再学習でモデルが悪化する。
エラーも出ず画面も正常に見えるので、機械で見張るしかない。

あわせて確認すること:
  - 元画像(_orig)を書き換えていないか（回すほど画質が落ちるのを防ぐ設計）
  - 何周回しても結果が同じバイト列か（劣化が積み重ならないこと）
  - 回転で枠と確認済みが解除されるか
  - 縦横が混ざったPDFが出せるか

使い方（リポジトリのルートで）:
    python tools/selfcheck_rotation.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import json
import os
import shutil
import sys
import uuid

from PIL import Image

import app
import training_data

fails = []


def eq(name, got, want):
    if got != want:
        fails.append(f"{name}: got={got!r} want={want!r}")
        print(f"  NG {name}: got={got!r} want={want!r}")
    else:
        print(f"  ok {name}: {got!r}")


# 学習データ送信を監視（実際にHubへは送らない）
captured = []
training_data.save_page = lambda **kw: captured.append(kw) or True

W, H = 400, 600            # 縦長の元ページ
job = "rot-" + uuid.uuid4().hex[:8]
out = app.DIR_OUTPUT
for sub in ("_orig", "_pages", "_thumbs"):
    os.makedirs(os.path.join(out, job + sub), exist_ok=True)

# 向きが分かるように、上半分を赤・左半分に緑の帯を入れた画像を作る
orig = Image.new("RGB", (W, H), "white")
for y in range(H // 2):
    for x in range(0, W, 7):
        orig.putpixel((x, y), (255, 0, 0))
for y in range(0, H, 7):
    for x in range(W // 2):
        orig.putpixel((x, y), (0, 200, 0))
orig.save(os.path.join(out, job + "_orig", "page_0.jpg"), quality=95)
orig.save(os.path.join(out, job + "_pages", "page_0.jpg"), quality=95)
orig.save(os.path.join(out, job + "_thumbs", "page_0.jpg"), quality=95)

with open(os.path.join(out, job + "_band.txt"), "w", encoding="utf-8") as f:
    f.write(os.path.join(app.DIR_OWN_BANDS, app.get_default_band()))
with open(os.path.join(out, job + "_opts.json"), "w", encoding="utf-8") as f:
    json.dump({"whiteout_map": False}, f)

meta = [{
    "page_index": 0, "page_label": "1ページ目", "missed": False,
    "img_w": W, "img_h": H, "rotation": 0,
    "detections": [{"cls": "band", "x0": 0, "y0": 480, "x1": W, "y1": 520, "conf": 0.9}],
    "label_only": [],
}]
meta_path = os.path.join(out, job + "_meta.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False)

client = app.app.test_client()
orig_path = app._orig_page_path(job, 0)
orig_bytes_before = open(orig_path, "rb").read()


def load_meta():
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)[0]


print("== まず確認済みにしておく（回転で外れることを見るため） ==")
eq("confirm", client.post(f"/confirm_page/{job}/0").get_json().get("ok"), True)
eq("確認済み", sorted(app._load_confirmed(job)), [0])

print("\n== 右90度に回す ==")
r = client.post(f"/rotate_page/{job}/0", json={"delta": 90})
d = r.get_json()
eq("ok", d.get("ok"), True)
eq("回転角", d.get("rotation"), 90)
eq("縦横が入れ替わる", (d.get("img_w"), d.get("img_h")), (H, W))
eq("metaの回転角", load_meta()["rotation"], 90)
eq("metaの縦横", (load_meta()["img_w"], load_meta()["img_h"]), (H, W))
eq("枠は消える", load_meta()["detections"], [])
eq("未検出になる", load_meta()["missed"], True)
eq("確認済みが外れる", sorted(app._load_confirmed(job)), [])
print(f"  推奨位置: {len(d.get('suggestions', []))}件（合成画像なので0でも正常）")

print("\n== 元画像は書き換わっていないか（回すほど劣化するのを防ぐ設計） ==")
eq("_orig のバイト列が同一", open(orig_path, "rb").read() == orig_bytes_before, True)

print("\n== 回転が本当に効いているか（画素で確認） ==")
work = Image.open(app._work_page_path(job, 0)).convert("RGB")
eq("作業用画像のサイズ", work.size, (H, W))
expected = orig.transpose(Image.ROTATE_270)   # 時計回り90度
# 元画像の左上(赤い領域)は、右90度回転で右上へ移動する
eq("右上が赤系（元の左上）", work.getpixel((H - 10, 10))[0] > work.getpixel((H - 10, 10))[2], True)
eq("PILの時計回り90度と一致するサイズ", expected.size, work.size)

print("\n== エディタの背景も回転後になっているか ==")
r = client.get(f"/orig_img/{job}/0")
eq("status", r.status_code, 200)
import io as _io
eq("配信される画像のサイズ", Image.open(_io.BytesIO(r.data)).size, (H, W))

print("\n== 出力(_pages)とサムネも回転後になっているか ==")
eq("_pages", Image.open(os.path.join(out, job + "_pages", "page_0.jpg")).size, (H, W))
th = Image.open(os.path.join(out, job + "_thumbs", "page_0.jpg"))
eq("サムネが横長になる", th.width > th.height, True)

print("\n== ★最重要: 学習データの画像と座標が一致するか ==")
captured.clear()
payload = {"rects": [
    {"cls": "band", "x0_ratio": 0.0, "y0_ratio": 0.80, "x1_ratio": 1.0, "y1_ratio": 0.90},
]}
d = client.post(f"/edit_page/{job}/0", json=payload).get_json()
eq("貼り直し成功", d.get("ok"), True)
eq("学習データを送った", d.get("saved"), True)
sent = captured[-1]
eq("送った画像は作業用(_work)", os.path.basename(os.path.dirname(sent["orig_path"])), job + "_work")
sent_img = Image.open(sent["orig_path"])
eq("送った画像のサイズ＝回転後", sent_img.size, (H, W))
eq("一緒に送った縦横も回転後", (sent["img_w"], sent["img_h"]), (H, W))
eq("★画像の縦横と座標系の縦横が一致", sent_img.size, (sent["img_w"], sent["img_h"]))
b = sent["boxes"][0]
eq("帯のy座標が回転後の高さに収まる", 0 <= b["y0"] < b["y1"] <= sent["img_h"], True)
eq("帯のx座標が回転後の幅いっぱい", (b["x0"], b["x1"]), (0, sent["img_w"]))
lines = training_data._yolo_lines(sent["boxes"], sent["img_w"], sent["img_h"])
print(f"  生成されるYOLOラベル: {lines}")
eq("ラベルが1行", len(lines), 1)

print("\n== 何度回しても劣化が積み重ならないか ==")
# 設計上の保証は「毎回 _orig から作り直すので、エンコードは常に1回」。
# よって同じ角度に戻れば、何周しても中身は完全に同じバイト列になるはず。
# （在庫の画像を上書きしていく実装だと、回すたびに再エンコードで劣化が積もる）
for _ in range(3):
    client.post(f"/rotate_page/{job}/0", json={"delta": 90})
eq("回転角が0に戻る", load_meta()["rotation"], 0)
eq("縦横も戻る", (load_meta()["img_w"], load_meta()["img_h"]), (W, H))
after_1turn = open(app._work_page_path(job, 0), "rb").read()

for _ in range(4):
    client.post(f"/rotate_page/{job}/0", json={"delta": 90})
eq("2周目も回転角0", load_meta()["rotation"], 0)
after_2turns = open(app._work_page_path(job, 0), "rb").read()
eq("★2周しても1周目とバイト単位で同一（劣化が積もらない）",
   after_1turn == after_2turns, True)

import numpy as np
back = Image.open(app._work_page_path(job, 0)).convert("RGB")
diff = np.abs(np.asarray(orig, dtype=int) - np.asarray(back, dtype=int))
print(f"  参考: 元画像との画素差 平均 {diff.mean():.2f}"
      f"（テスト画像が1画素おきの点でJPEGに最も不利な模様のため大きく出る。"
      f"何周しても増えないことが上の判定で確認できている）")

print("\n== 180度・270度 ==")
client.post(f"/rotate_page/{job}/0", json={"delta": 180})
eq("180度で縦横は変わらない", (load_meta()["img_w"], load_meta()["img_h"]), (W, H))
client.post(f"/rotate_page/{job}/0", json={"delta": 90})
eq("270度で縦横が入れ替わる", (load_meta()["img_w"], load_meta()["img_h"]), (H, W))
eq("累積角度", load_meta()["rotation"], 270)

print("\n== 不正な角度は弾くか ==")
r = client.post(f"/rotate_page/{job}/0", json={"delta": 45})
eq("45度は400", r.status_code, 400)
eq("角度は変わっていない", load_meta()["rotation"], 270)

print("\n== 縦横が混ざったPDFが出せるか ==")
# 2ページ目（回転なし・縦長）を足す
orig.save(os.path.join(out, job + "_pages", "page_1.jpg"), quality=95)
m = json.load(open(meta_path, encoding="utf-8"))
m.append({"page_index": 1, "page_label": "2ページ目", "missed": False,
          "img_w": W, "img_h": H, "rotation": 0, "detections": [], "label_only": []})
json.dump(m, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False)
r = client.get(f"/download/{job}")
eq("status", r.status_code, 200)
eq("PDFになっている", r.data[:5], b"%PDF-")
import fitz
doc = fitz.open(stream=r.data, filetype="pdf")
sizes = [(round(p.rect.width), round(p.rect.height)) for p in doc]
doc.close()
print(f"  ページの寸法: {sizes}")
eq("2ページある", len(sizes), 2)
eq("向きが混在しても出せる", sizes[0][0] > sizes[0][1] and sizes[1][0] < sizes[1][1], True)

print("\n== 後片付け ==")
for sub in ("_orig", "_pages", "_thumbs", "_work"):
    shutil.rmtree(os.path.join(out, job + sub), ignore_errors=True)
for suf in ("_meta.json", "_band.txt", "_confirmed.json", "_opts.json"):
    p = os.path.join(out, job + suf)
    if os.path.exists(p):
        os.remove(p)
print("  削除しました")

print()
if fails:
    print(f"NG が {len(fails)} 件:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("すべてOK")
