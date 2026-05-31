"""検出モジュール（YOLO物体検出）。

Roboflowでラベル付けした画像でfine-tuningしたYOLOv8モデル(best.pt)を使う。
- クラス: band(他社の会社情報帯) / logo(他社ロゴ) / map(地図)
- マイソクの画像フォーマットはバラバラなので、見た目を学習した物体検出で頑健に当てる。
- 取りこぼしはレビュー画面の手動ドラッグで補う前提。
"""
import os
import threading

from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "best.pt")

# これ未満の確信度は無視（誤検出＝関係ない場所の白塗り事故を抑える。precision重視）
CONF_THRESHOLD = 0.30

_model = None
_model_lock = threading.Lock()


def _get_model():
    """モデルを一度だけ読み込む（スレッドセーフ・初回のみ）。"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = YOLO(MODEL_PATH)
    return _model


def detect_objects(pil_img):
    """画像から検出物を返す。

    返り値: [{"cls": str, "x0","y0","x1","y1": int, "conf": float}]
    座標は入力画像のピクセル。clsは 'band' / 'logo' / 'map'。
    """
    model = _get_model()
    results = model.predict(pil_img, conf=CONF_THRESHOLD, verbose=False)
    out = []
    if not results:
        return out

    r = results[0]
    names = r.names  # {0:'band', 1:'logo', 2:'map'}
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x0, y0, x1, y1 = (float(v) for v in box.xyxy[0])
        out.append({
            "cls": names.get(cls_id, str(cls_id)),
            "x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1),
            "conf": conf,
        })
    return out
