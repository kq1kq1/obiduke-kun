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

**公開URL（Hugging Face Spaces）**: https://huggingface.co/spaces/kq1kq1/obiduke-kun
（Publicなら誰でもブラウザから利用可。HFアカウント不要）

## 何をするか

1. PDFをアップロード（複数可・ドラッグ＆ドロップ対応／アップロード進捗バー表示）
2. 各ページから他社の会社情報帯・ロゴを自動検出して白塗り
3. その位置に自社帯を貼り付け
4. レビュー画面で確認・手動調整し、PDFとしてダウンロード
   - 処理中は **完了までの推定残り時間（ETA）** を表示
   - レビュー画面でページを **ドラッグして並べ替え**（出力PDFの順序に反映）
   - 取りこぼしは手動で「帯を貼る」または「白塗りのみ」で補完できる

## 検出のしくみ（YOLO物体検出）

`detect.py` が学習済みYOLOモデル（`best.pt`）で検出する。マイソクの画像フォーマットがバラバラでも見た目を学習しているため頑健。

- **band（他社の会社情報帯）**: 白塗りして自社帯に差し替え（全幅に広げて確実に隠す）
- **logo（他社ロゴ）**: 白塗りのみ
- **map（案内図）**: アップロード時のチェックで「案内図も白塗りする」を選んだ場合のみ白塗り（区画図は対象外）
- 取りこぼし分はレビュー画面のドラッグ選択で手動補完。

### 推論の高速化（OpenVINO）

Intel CPU（HF Spaces含む）では PyTorch より OpenVINO の方が速いため、`best.pt` を
OpenVINO形式に変換した `best_openvino_model/` を優先して使う（無ければ `best.pt` に
フォールバック）。起動時にウォームアップ（モデルのコンパイル）を済ませて1枚目から速くする。

### モデルの再学習（精度を上げたいとき）

苦手なフォーマット（検出漏れ・新しい帯/ロゴ/案内図）が出たら、その画像を足して再学習すれば改善する。

1. 苦手だったマイソクPDFを `samples/` に入れ、`python export_images.py` で `training_images/` にPNG化
2. [Roboflow](https://roboflow.com)（無料Publicプラン）の既存プロジェクトに画像を追加アップし、band/logo/map を四角で囲む
3. 新しいバージョンを作成 → YOLOv8形式でExport → Google Colab（無料GPU）で `ultralytics` で学習
4. 出来た `best.pt` をプロジェクト直下に上書き
5. **OpenVINO形式を作り直す**:
   ```powershell
   python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='openvino', imgsz=640)"
   ```
6. `best.pt` と `best_openvino_model/` をコミットして再デプロイ（後述の「再デプロイ手順」）

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

- Space: https://huggingface.co/spaces/kq1kq1/obiduke-kun
- SDK: **Docker**（`Dockerfile` 同梱、ポート7860）
- 進捗をメモリ上で管理するため **gunicorn worker は1つ固定**（複数workerだと進捗共有が壊れる）。並行アクセスはthreadsで処理。
- HFは**バイナリをGit LFSで管理する必要がある**（`best.pt` / `best_openvino_model/*.bin` / `own_bands/*.png`）。
- 注意: Spacesのファイルシステムは再起動で初期化されるため、画面から登録した自社帯は再起動後に消える。恒久的に増やすなら `own_bands/` に置いてコミット→再デプロイする。

### 再デプロイ手順

GitHub（`origin`）はフル履歴、HF（`hf`）はバイナリをLFS管理した単一コミットで運用している。

```powershell
# 1. 変更を GitHub にコミット＆push
git add -A
git commit -m "変更内容"
git push origin main

# 2. HF用にLFS付きの単一コミットを作って push（hf-deployブランチを作り直す）
git checkout --orphan hf-deploy
git rm -r --cached . > $null
git lfs track "*.pt" "*.png" "*.jpg" "*.jpeg" "*.bin"
git add .gitattributes; git add -A
git commit -m "HFデプロイ"
git push "https://ユーザー名:HFトークン@huggingface.co/spaces/kq1kq1/obiduke-kun" hf-deploy:main --force
git checkout main
git branch -D hf-deploy
```

HFトークンは https://huggingface.co/settings/tokens で **Write** 権限のものを発行する（使い終わったら失効可）。

## 構成

| ファイル | 役割 |
|---|---|
| `app.py` | Flaskアプリ本体（ルーティング・ジョブ管理・帯付け・並べ替え） |
| `detect.py` | 帯検出ロジック（OpenVINO優先／`best.pt`フォールバック） |
| `best.pt` / `best_openvino_model/` | 学習済みYOLOモデル（PyTorch版／OpenVINO版） |
| `export_images.py` | `samples/`のPDFを学習用PNGに書き出す（再学習の準備用） |
| `templates/` | 画面（index / processing / review / bands） |
| `static/` | CSS |
| `Dockerfile` | HF Spacesデプロイ用 |
