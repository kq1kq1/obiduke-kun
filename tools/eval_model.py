"""凍結検証セットでモデルを採点する（新旧を同じ物差しで比べるための道具）。

なぜ独自の指標を出すか:
  一般的な mAP は「枠がどれだけ重なったか」を測るが、帯付けくんは検出した帯を
  **必ず全幅に広げてから白塗りする**（app.py の paste_rect）。つまり band の x座標は
  捨てられていて、実際に効くのは y方向だけ。mAPだけ見ていると、実務では関係ない
  x方向のズレで良し悪しを判断してしまう。

  そこでこのスクリプトは、アプリが実際に困っている2つを直接測る:
    - 検出漏れ  (再現率)   … 赤い「未検出」になって手作業が増える
    - 誤検出    (適合率)   … 関係ない場所が白塗りされる
  band は y方向だけのIoU、logo/map は通常のIoUで判定する。

使い方:
    python tools/eval_model.py best.pt
    python tools/eval_model.py best.pt --labels fullwidth   # 案B（band全幅）の規約で採点
    python tools/eval_model.py best.pt --save eval/baseline_best_pt.json
    python tools/eval_model.py best.pt --sweep              # 確信度しきい値を振る
"""
import argparse
import json
import sys
from pathlib import Path

NAMES = ["band", "logo", "map"]
FROZEN = Path("datasets/frozen_val")
# アプリと同じ条件で測る（detect.py の CONF_THRESHOLD / IMGSZ と揃えること）
APP_CONF = 0.30
IMGSZ = 640
IOU_MATCH = 0.5


def load_gt(txt_path, w, h):
    """YOLOラベルを絶対座標の枠に直す。"""
    out = []
    if not txt_path.exists():
        return out
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        try:
            cid = int(float(p[0]))
            xc, yc, bw, bh = (float(v) for v in p[1:])
        except ValueError:
            continue
        out.append({
            "cls": cid,
            "x0": (xc - bw / 2) * w, "x1": (xc + bw / 2) * w,
            "y0": (yc - bh / 2) * h, "y1": (yc + bh / 2) * h,
        })
    return out


def iou(a, b, y_only=False):
    """重なり具合。y_only=Trueならy方向だけで測る（帯は全幅に広げるためxは無意味）。"""
    iy = max(0.0, min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]))
    if y_only:
        uy = (a["y1"] - a["y0"]) + (b["y1"] - b["y0"]) - iy
        return iy / uy if uy > 0 else 0.0
    ix = max(0.0, min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]))
    inter = ix * iy
    union = ((a["x1"] - a["x0"]) * (a["y1"] - a["y0"])
             + (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) - inter)
    return inter / union if union > 0 else 0.0


def match(gts, preds, y_only):
    """確信度の高い予測から順に、いちばん重なる正解に割り当てる。"""
    used, pairs = set(), []
    for p in sorted(preds, key=lambda d: -d["conf"]):
        best, best_i = 0.0, -1
        for i, g in enumerate(gts):
            if i in used:
                continue
            v = iou(g, p, y_only)
            if v > best:
                best, best_i = v, i
        if best >= IOU_MATCH and best_i >= 0:
            used.add(best_i)
            pairs.append((best_i, p, best))
        else:
            pairs.append((None, p, best))
    return pairs


def evaluate(model, images, label_dir, conf):
    from PIL import Image
    stats = {n: {"tp": 0, "fp": 0, "fn": 0, "ious": []} for n in NAMES}
    pages_missing_band = 0
    for img_path in images:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            r = model.predict(im, imgsz=IMGSZ, conf=conf, verbose=False)[0]
        preds = []
        for b in r.boxes:
            x0, y0, x1, y1 = (float(v) for v in b.xyxy[0])
            preds.append({"cls": int(b.cls[0]), "conf": float(b.conf[0]),
                          "x0": x0, "y0": y0, "x1": x1, "y1": y1})
        gts = load_gt(label_dir / (img_path.stem + ".txt"), w, h)

        for cid, name in enumerate(NAMES):
            g = [x for x in gts if x["cls"] == cid]
            p = [x for x in preds if x["cls"] == cid]
            pairs = match(g, p, y_only=(name == "band"))
            hit = {i for i, _, _ in pairs if i is not None}
            stats[name]["tp"] += len(hit)
            stats[name]["fp"] += sum(1 for i, _, _ in pairs if i is None)
            stats[name]["fn"] += len(g) - len(hit)
            stats[name]["ious"] += [v for i, _, v in pairs if i is not None]

        # 実務でいちばん困るケース: 正解に帯があるのに1つも出せなかったページ
        if any(x["cls"] == 0 for x in gts) and not any(x["cls"] == 0 for x in preds):
            pages_missing_band += 1
    return stats, pages_missing_band


def summarize(stats, pages, missing):
    rows = []
    for n in NAMES:
        s = stats[n]
        rec = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0.0
        pre = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0
        f1 = 2 * rec * pre / (rec + pre) if (rec + pre) else 0.0
        mi = sum(s["ious"]) / len(s["ious"]) if s["ious"] else 0.0
        rows.append({"cls": n, "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
                     "recall": round(rec, 4), "precision": round(pre, 4),
                     "f1": round(f1, 4), "mean_iou": round(mi, 4)})
    return {"per_class": rows, "pages": pages,
            "pages_missing_band": missing,
            "pages_missing_band_rate": round(missing / pages, 4) if pages else 0.0}


def main():
    ap = argparse.ArgumentParser(description="凍結検証セットでモデルを採点する")
    ap.add_argument("model", help="best.pt か best_openvino_model のパス")
    ap.add_argument("--labels", choices=["orig", "fullwidth"], default="orig",
                    help="採点に使うラベルの規約（既定: orig＝Roboflowのまま）")
    ap.add_argument("--save", default=None, help="結果をJSONで保存するパス")
    ap.add_argument("--sweep", action="store_true", help="確信度しきい値を振って比べる")
    args = ap.parse_args()

    img_dir = FROZEN / "images"
    label_dir = FROZEN / ("labels" if args.labels == "orig" else "labels_fullwidth")
    if not img_dir.is_dir():
        print("エラー: 凍結検証セットがありません。先に tools/freeze_val_set.py を実行してください。",
              file=sys.stderr)
        return 1
    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    from ultralytics import YOLO
    mp = Path(args.model)
    model = YOLO(str(mp), task="detect") if mp.is_dir() else YOLO(str(mp))

    conv = "band=全幅・案B" if args.labels == "fullwidth" else "帯にぴったり・現行"
    print(f"モデル      : {args.model}")
    print(f"ラベル規約  : {args.labels}（{conv}）")
    print(f"検証セット  : {len(images)}枚  conf={APP_CONF} imgsz={IMGSZ}（アプリと同条件）")
    print()

    stats, missing = evaluate(model, images, label_dir, APP_CONF)
    res = summarize(stats, len(images), missing)

    header = "クラス      正解  誤検出  見逃し    再現率    適合率      F1   平均IoU"
    print(header)
    print("-" * len(header))
    for r in res["per_class"]:
        note = "  ← y方向のみで判定" if r["cls"] == "band" else ""
        print(f"{r['cls']:<8}{r['tp']:>6}{r['fp']:>8}{r['fn']:>8}"
              f"{r['recall']:>10.3f}{r['precision']:>10.3f}"
              f"{r['f1']:>8.3f}{r['mean_iou']:>10.3f}{note}")
    print()
    print(f"帯を1つも出せなかったページ: {res['pages_missing_band']} / {res['pages']} "
          f"({res['pages_missing_band_rate'] * 100:.1f}%)  ← 赤い「未検出」になる分")

    if args.sweep:
        print()
        print("確信度しきい値を振ったときの band:")
        print("    conf    再現率    適合率  誤検出")
        for c in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
            st, _ = evaluate(model, images, label_dir, c)
            s = st["band"]
            rec = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0.0
            pre = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0
            mark = "  ← 今の設定" if abs(c - APP_CONF) < 1e-9 else ""
            print(f"  {c:>6.2f}{rec:>10.3f}{pre:>10.3f}{s['fp']:>8}{mark}")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        res["model"] = args.model
        res["labels"] = args.labels
        res["conf"] = APP_CONF
        res["imgsz"] = IMGSZ
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
