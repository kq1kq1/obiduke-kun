---
title: 帯付けくん
emoji: 🏷️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 帯付けくん

マイソク（不動産チラシPDF）の **他社帯** を自動で白塗りし、登録した **自社帯** に差し替えるWebツール。社内数人での利用を想定。

## 何をするか

1. PDFをアップロード（複数可・ドラッグ＆ドロップ対応）
2. 各ページから他社の会社情報帯を自動検出して白塗り
3. その位置に自社帯を貼り付け
4. レビュー画面で確認・手動調整し、PDFとしてダウンロード

## 検出のしくみ（YOLO物体検出）

`detect.py` が学習済みYOLOモデル（`best.pt`）で検出する。マイソクの画像フォーマットがバラバラでも見た目を学習しているため頑健。

- **band（他社の会社情報帯）**: 白塗りして自社帯に差し替え（全幅に広げて確実に隠す）
- **logo（他社ロゴ）**: 白塗りのみ
- **map（案内図）**: アップロード時のチェックで「案内図も白塗りする」を選んだ場合のみ白塗り（区画図は対象外）
- 取りこぼし分はレビュー画面のドラッグ選択で手動補完。

### モデルの再学習（精度を上げたいとき）

1. `python export_images.py` で `samples/` のPDFを `training_images/` にPNG化
2. [Roboflow](https://roboflow.com)（無料Publicプラン）に画像をアップし、band/logo/map を四角で囲む
3. データセットをYOLOv8形式でExport → Google Colab（無料GPU）で `ultralytics` で学習
4. 出来た `best.pt` をプロジェクト直下に上書きして差し替え

苦手なフォーマットが出たらその画像をラベル付けして足し、再学習すれば改善する。

## 環境構築（ローカル）

Python 3.13。

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

ブラウザで http://localhost:5000 を開く。

## 使い方

1. `/bands`（自社帯の管理）で全幅の横帯画像（PNG/JPG）を登録
2. トップページで自社帯を選び、PDFをアップロードして「処理開始」
3. 処理完了後、レビュー画面で各ページを確認
   - 検出漏れページは赤く表示される。画像上でドラッグして帯範囲を指定すると自社帯が貼られる
4. 「ダウンロード」でPDFを取得

## デプロイ（Hugging Face Spaces）

- SDK: **Docker**（`Dockerfile` 同梱、ポート7860）
- 進捗をメモリ上で管理するため **gunicorn worker は1つ固定**（複数workerだと進捗共有が壊れる）。並行アクセスはthreadsで処理。
- 注意: Spacesのファイルシステムは再起動で初期化されるため、登録した自社帯は再起動後に消える。永続化が必要な場合はHFのPersistent Storageを利用する。

## 構成

| ファイル | 役割 |
|---|---|
| `app.py` | Flaskアプリ本体（ルーティング・ジョブ管理・帯付け） |
| `detect.py` | 帯検出ロジック |
| `templates/` | 画面（index / processing / review / bands） |
| `static/` | CSS |
| `Dockerfile` | HF Spacesデプロイ用 |
