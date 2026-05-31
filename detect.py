"""検出モジュール（YOLO物体検出）。

Roboflowでラベル付けした画像でfine-tuningしたYOLOv8モデルを使う。
- クラス: band(他社の会社情報帯) / logo(他社ロゴ) / map(地図)
- マイソクの画像フォーマットはバラバラなので、見た目を学習した物体検出で頑健に当てる。
- 取りこぼしはレビュー画面の手動ドラッグで補う前提。

高速化:
- Intel CPU(HF Spaces含む)では OpenVINO 形式の方が PyTorch より速いので、
  変換済みの best_openvino_model/ があれば優先して使う（無ければ best.pt にフォールバック）。
- 起動時に warmup() を呼ぶと初回のモデルコンパイル(数十秒)を先に済ませられる。
"""
import os
import threading

from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
OPENVINO_DIR = os.path.join(BASE, "best_openvino_model")
PT_PATH = os.path.join(BASE, "best.pt")

# 推論解像度（学習・エクスポート時と揃える）
IMGSZ = 640
# これ未満の確信度は無視（誤検出＝関係ない場所の白塗り事故を抑える。precision重視）
CONF_THRESHOLD = 0.30

_model = None
_model_lock = threading.Lock()


def _get_model():
    """モデルを一度だけ読み込む（スレッドセーフ・初回のみ）。

    OpenVINO形式があればそれを、無ければ .pt を使う。
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if os.path.isdir(OPENVINO_DIR):
                    # task を明示しておくと推論時の警告が出ない
                    _model = YOLO(OPENVINO_DIR, task="detect")
                else:
                    _model = YOLO(PT_PATH)
    return _model


def warmup():
    """起動時にモデルを読み込み、ダミー推論でコンパイルを済ませておく。

    OpenVINOは初回推論時にモデルをコンパイルするため数十秒かかる。
    先に済ませておけば、ユーザーの1枚目から速くなる。
    """
    from PIL import Image
    model = _get_model()
    dummy = Image.new("RGB", (IMGSZ, IMGSZ), "white")
    model.predict(dummy, imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False)


def detect_objects(pil_img):
    """画像から検出物を返す。

    返り値: [{"cls": str, "x0","y0","x1","y1": int, "conf": float}]
    座標は入力画像のピクセル。clsは 'band' / 'logo' / 'map'。
    """
    model = _get_model()
    results = model.predict(pil_img, imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False)
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
