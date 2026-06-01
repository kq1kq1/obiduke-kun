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
4. レビュー画面で確認・修正し、PDFとしてダウンロード
   - 処理中は **完了までの推定残り時間（ETA）** を表示
   - レビュー画面でページを **ドラッグして並べ替え**（出力PDFの順序に反映）
   - 各ページの帯・白塗りを **枠の直接編集** で修正できる（後述）

### レビュー画面の枠エディタ

カードをクリックすると、帯付け前の元画像の上に検出枠が表示される。

- **橙＝自社帯 / 赤＝白塗り**
- 枠を **ドラッグで移動**、選択中に出る青ハンドルで **サイズ変更**、**×で削除**
- 「＋帯」「＋白塗り」で枠を **追加**（帯は全幅・縦のみ調整、白塗りは2D自由）
- 「適用して貼り直す」で元画像から再構成（誤検出の削除・ズレ修正・取りこぼし追加が1画面で完結）

## 検出のしくみ（YOLO物体検出）

`detect.py` が学習済みYOLOモデル（`best.pt`）で検出する。マイソクの画像フォーマットがバラバラでも見た目を学習しているため頑健。

- **band（他社の会社情報帯）**: 白塗りして自社帯に差し替え（全幅に広げて確実に隠す）
- **logo（他社ロゴ）**: 白塗りのみ
- **map（案内図）**: アップロード時のチェックで「案内図も白塗りする」を選んだ場合のみ白塗り（区画図は対象外）
- 取りこぼし・誤検出はレビュー画面の枠エディタで手動修正。

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
   - 複数登録できる。「デフォルトにする」で、トップページで最初に選ばれる帯を変更できる（`own_bands_default.txt` に記録）
2. トップページで自社帯を選び、PDFをアップロードして「処理開始」
3. 処理完了後、レビュー画面で各ページを確認
   - 検出漏れページは赤く表示される。カードをクリックして枠エディタで帯・白塗りを追加／修正
4. 「ダウンロード」でPDFを取得

### 自社帯について

- 対応形式は **PNG / JPG**（透過させたいときはPNG）。ファイル名は **英数字（アンダースコア区切り）推奨**（日本語名は文字化けの素）。
- 画面の「アップロード」から足した帯は **再起動で消える**（HFのファイルシステムが揮発性のため）。**恒久的に残す帯は `own_bands/` に置いてコミット→再デプロイ**する。
- 同時アクセス時の安全性: ジョブはUUIDで分離され、共有モデルの推論はロックで直列化しているため、社内数人が同時に処理しても結果は混ざらない。

## デプロイ（Hugging Face Spaces）

- Space: https://huggingface.co/spaces/kq1kq1/obiduke-kun
- SDK: **Docker**（`Dockerfile` 同梱、ポート7860）
- 進捗をメモリ上で管理するため **gunicorn worker は1つ固定**（複数workerだと進捗共有が壊れる）。並行アクセスはthreadsで処理。
- HFは**バイナリをGit LFSで管理する必要がある**（`best.pt` / `best_openvino_model/*.bin` / `own_bands/*.png` / `*.jpg`）。
- 注意: Spacesのファイルシステムは再起動で初期化されるため、画面から登録した自社帯は再起動後に消える。恒久的に増やすなら `own_bands/` に置いてコミット→再デプロイする。

- 画像・モデルは **GitHub にも入っている**（GitHubはLFSなしで100MBまでのバイナリを持てる）。GitHubがフル履歴・バックアップ、HFが公開・実行先。

### 再デプロイ手順

**いちばん簡単な方法**: まず変更を main にコミットしてから、付属スクリプトを実行する。

```powershell
git add -A
git commit -m "変更内容"
.\redeploy_hf.ps1   # GitHub push → HFへLFS付き単一コミットをforce push まで自動
```

実行するとHFの **Write トークン** を聞かれるので貼る（画面に表示されず保存もされない）。
トークンは https://huggingface.co/settings/tokens で発行（使い終わったら失効可）。

<details>
<summary>スクリプトを使わず手動でやる場合</summary>

```powershell
git push origin main
git checkout --orphan hf-deploy
git rm -r --cached . > $null
git lfs track "*.pt" "*.png" "*.jpg" "*.jpeg" "*.bin"
git add .gitattributes; git add -A
git commit -m "HFデプロイ"
git push "https://ユーザー名:HFトークン@huggingface.co/spaces/kq1kq1/obiduke-kun" hf-deploy:main --force
git checkout main
git branch -D hf-deploy
```
</details>

## 構成

| ファイル | 役割 |
|---|---|
| `app.py` | Flaskアプリ本体（ルーティング・ジョブ管理・帯付け・並べ替え・枠編集） |
| `detect.py` | 帯検出ロジック（OpenVINO優先／`best.pt`フォールバック・推論ロックで直列化） |
| `best.pt` / `best_openvino_model/` | 学習済みYOLOモデル（PyTorch版／OpenVINO版） |
| `export_images.py` | `samples/`のPDFを学習用PNGに書き出す（再学習の準備用） |
| `own_bands/` | 自社帯画像（恒久的に残す帯はここに置いてコミット） |
| `own_bands_default.txt` | デフォルトの自社帯ファイル名 |
| `redeploy_hf.ps1` | GitHub push＋HF再デプロイを自動化するスクリプト |
| `templates/` | 画面（index / processing / review / bands） |
| `static/` | CSS |
| `Dockerfile` | HF Spacesデプロイ用 |
