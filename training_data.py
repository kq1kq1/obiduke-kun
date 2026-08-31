"""手修正を学習データとして貯めるモジュール（再学習で精度を上げるための土台）。

レビュー画面で人が「直した」「これでOK」と確認したページを、帯付け前の元画像＋
枠の座標として private な Hugging Face Dataset リポジトリに送る。溜まったものを
定期的に再学習に回すと、使うほど精度が上がる。

設計の要点:
- Spacesのディスクは揮発性なので、外部(Datasetリポジトリ)に逃がさないと再起動で消える。
- マイソクには他社担当者名・連絡先・物件情報が載る。リポジトリは必ず private にする。
- CommitScheduler は「追記のみ」が前提（既存ファイルの上書きはリポジトリを壊す恐れあり）。
  そこで
    画像  = 内容のSHA256をファイル名にして一度だけ書く（同じページは二度書かない）
    ラベル = JSONL に追記するだけ（同じ画像を再修正したら行が増える）
  という形にした。学習時は画像ごとに「最後の行」を採用する（＝最新の修正が勝つ）。
- 送り先が未設定なら黙って何もしない（ローカル開発で誤ってpushしないため）。
- ここで例外が出ても本来のPDF処理は絶対に止めない（学習データはおまけ）。

必要な環境変数（HF Spacesでは Settings > Variables and secrets に設定）:
  TRAINING_DATA_REPO   例: kq1kq1/obiduke-training-data （未設定なら機能OFF）
  HF_TOKEN             書き込み権限のあるHFトークン（Secretとして登録）
  TRAINING_DATA_EVERY  タイマーで何分ごとにHubへ送るか（省略時30分）

送信のタイミング:
- PDFをダウンロードした瞬間に flush() で即送信する（＝その回の作業が終わった合図）。
  通常はこれで送り終わるので、失われる分はほぼ無い。
- 上記のタイマーは「確認したがダウンロードせずに閉じた」ときだけの保険。間隔を短くすると
  コミットが増えてHub上の動作が重くなる（公式に数千コミットで劣化するとある）ので長めにする。
- 送るものが無ければ何もコミットしない（空コミットは捨てられる）。使わない日は通信も起きない。
"""
import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
# CommitSchedulerが監視するローカルフォルダ（ここに置いたものが定期的にHubへ行く）
LOCAL_DIR = BASE / "training_data"
IMAGES_SUBDIR = "images"
RECORDS_SUBDIR = "records"

# 学習ラベルに使うクラス。detect.py の names（0:band 1:logo 2:map）と必ず同じ順にする。
#
# "wo"（人が「白塗り」ボタンで足した枠）は logo と同じ 1 に寄せる。
# アプリでの logo の実用上の意味は「白塗りすべき局所領域」で、人が白塗りしたものも
# まさにそれなので、同じクラスとして学習させる。
# ただし記録の boxes には cls="wo" がそのまま残るので、あとから
# 「人が足した分は学習に使わない」と方針を変えることもできる（判断を先送りできる形）。
CLASS_IDS = {"band": 0, "logo": 1, "map": 2, "wo": 1}

_repo_id = os.environ.get("TRAINING_DATA_REPO", "").strip()
_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "").strip()

_scheduler = None
_records_path = None
_init_error = None
_saved_count = 0
_start_lock = threading.Lock()


def is_configured():
    """送り先が設定されているか（ネットワークを触らない軽い判定）。"""
    return bool(_repo_id and _token)


def status():
    """レビュー画面に出す状態。ネットワークは触らない。"""
    if not _repo_id:
        reason = "TRAINING_DATA_REPO が未設定"
    elif not _token:
        reason = "HF_TOKEN が未設定"
    else:
        reason = _init_error
    return {
        "enabled": is_configured() and _init_error is None,
        "repo": _repo_id or None,
        "saved": _saved_count,
        "reason": reason,
    }


def _ensure_started():
    """CommitSchedulerを一度だけ起動する。失敗したら以後あきらめる（アプリは止めない）。"""
    global _scheduler, _records_path, _init_error
    if _scheduler is not None or _init_error is not None:
        return _scheduler
    if not is_configured():
        _init_error = "送り先が未設定のため学習データの蓄積は無効です"
        return None

    with _start_lock:
        if _scheduler is not None or _init_error is not None:
            return _scheduler
        try:
            from huggingface_hub import CommitScheduler, HfApi

            (LOCAL_DIR / IMAGES_SUBDIR).mkdir(parents=True, exist_ok=True)
            (LOCAL_DIR / RECORDS_SUBDIR).mkdir(parents=True, exist_ok=True)

            # 個人情報が載るので private を明示して先に作る。
            # CommitScheduler任せにせず自分で作るのは、privateが確実になるため。
            HfApi(token=_token).create_repo(
                repo_id=_repo_id, repo_type="dataset", private=True, exist_ok=True
            )

            # プロセスごとに別のJSONLへ書く。再起動や複数レプリカでも行を上書きしない。
            _records_path = LOCAL_DIR / RECORDS_SUBDIR / f"records_{uuid.uuid4().hex}.jsonl"
            _records_path.touch()

            # 実送信はダウンロード時の flush() が主役。これは取りこぼし用の保険なので長めでよい。
            try:
                every = float(os.environ.get("TRAINING_DATA_EVERY", "30"))
            except ValueError:
                every = 30.0

            _scheduler = CommitScheduler(
                repo_id=_repo_id,
                repo_type="dataset",
                folder_path=LOCAL_DIR,
                path_in_repo="data",
                every=every,
                token=_token,
            )
            print(f"[info] 学習データの蓄積を有効化: {_repo_id}（{every}分ごとに送信）")
        except Exception as e:
            _init_error = str(e)
            print(f"[warn] 学習データの蓄積を無効にしました: {e}")
            return None
    return _scheduler


def init():
    """起動時に呼ぶ。ネットワークを使うので呼び出し側は別スレッドにすること。"""
    _ensure_started()


def flush():
    """溜まっている分を今すぐHubへ送る（タイマーを待たない）。

    PDFのダウンロード＝その回の作業が終わった合図なので、そこで呼ぶ。
    タイマーを待つと、その間にSpaceが再起動したぶんが失われる。

    trigger() は Future を返す非同期呼び出しなので、ダウンロードの応答は遅くならない。
    送るものが無ければ何もコミットされない。まだ1件も保存していなければ
    スケジューラ自体が動いていないので、ここでは起動もしない（無駄な通信を避ける）。
    """
    sched = _scheduler
    if sched is None:
        return False
    try:
        sched.trigger()
        return True
    except Exception as e:
        print(f"[warn] 学習データの送信を要求できませんでした: {e}")
        return False


def _yolo_lines(boxes, img_w, img_h):
    """枠(ピクセル)をYOLO形式の行に変換する。

    YOLO形式: <class_id> <x_center> <y_center> <width> <height>（すべて0..1に正規化）
    学習対象外のクラス（other）と、つぶれた枠は捨てる。
    """
    lines = []
    if img_w <= 0 or img_h <= 0:
        return lines
    for b in boxes:
        cid = CLASS_IDS.get(b.get("cls"))
        if cid is None:
            continue
        try:
            x0 = max(0, min(img_w, int(b["x0"])))
            x1 = max(0, min(img_w, int(b["x1"])))
            y0 = max(0, min(img_h, int(b["y0"])))
            y1 = max(0, min(img_h, int(b["y1"])))
        except (KeyError, TypeError, ValueError):
            continue
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        lines.append("%d %.6f %.6f %.6f %.6f" % (
            cid,
            (x0 + x1) / 2 / img_w,
            (y0 + y1) / 2 / img_h,
            (x1 - x0) / img_w,
            (y1 - y0) / img_h,
        ))
    return lines


def save_page(orig_path, boxes, img_w, img_h, edited):
    """1ページ分を学習データとして記録する。

    orig_path: 帯付け前の元画像（これがそのまま学習画像になる）
    boxes:     [{"cls","x0","y0","x1","y1"}] ピクセル。**貼り付け用に膨らませる前**の座標。
    edited:    人が枠を直したか（Falseは「見て、直す必要が無いと確認した」）

    枠が0個でも記録する。「このページには帯もロゴも無い」という人の確認は、
    誤検出（関係ない場所への白塗り）を減らすのに一番効く教師データになる。
    """
    global _saved_count
    sched = _ensure_started()
    if sched is None:
        return False
    try:
        data = Path(orig_path).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        # 1フォルダ1万ファイルの上限があるので先頭2文字で分ける
        rel = f"{IMAGES_SUBDIR}/{sha[:2]}/{sha}.jpg"
        dest = LOCAL_DIR / rel

        record = {
            "image": rel,
            "img_w": int(img_w),
            "img_h": int(img_h),
            "edited": bool(edited),
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "boxes": [
                {
                    "cls": b.get("cls"),
                    "x0": int(b["x0"]), "y0": int(b["y0"]),
                    "x1": int(b["x1"]), "y1": int(b["y1"]),
                }
                for b in boxes
                if all(k in b for k in ("x0", "y0", "x1", "y1"))
            ],
            # 学習スクリプトがそのまま .txt に書けるようYOLO形式も入れておく
            "label": "\n".join(_yolo_lines(boxes, img_w, img_h)),
        }

        with sched.lock:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # 追記のみが前提なので、同じ内容の画像は二度書かない
            if not dest.exists():
                dest.write_bytes(data)
            with _records_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        _saved_count += 1
        return True
    except Exception as e:
        # 学習データはおまけ。失敗しても利用者の作業は絶対に止めない。
        print(f"[warn] 学習データを保存できませんでした: {e}")
        return False
