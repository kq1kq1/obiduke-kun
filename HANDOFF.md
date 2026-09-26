# 引き継ぎ（2026-09-25 時点）

このファイルは「**今どうなっていて、次に何をするか**」だけを書く。
仕組みの説明は [README.md](README.md) にある。

---

## これは何か

マイソク（不動産チラシPDF）の**他社帯を白塗りして自社帯に差し替える**社内Webツール。
YOLOで検出 → 白塗り → 自社帯を貼る → レビュー画面で人が確認・修正 → PDF出力。

- 本番: https://huggingface.co/spaces/kq1kq1/obiduke-kun （HF Spaces / 無料CPU）
- コード: https://github.com/kq1kq1/obiduke-kun （public）
- 学習データ: `kq1kq1/obiduke-training-data`（HF Dataset / **private**）

---

## 今の状態

### 動いているもの

| | 状態 |
|---|---|
| 本番アプリ | **稼働中**。最新コードをデプロイ済み |
| 学習データの自動蓄積 | **動いている**。321ページ貯まっている |
| ページ回転（90度単位） | 実装済み・デプロイ済み |
| 枠の追加ボタン | 帯 / 案内図 / 白塗り の3つ |

### 使っているモデル

`best.pt`（YOLOv8n）＋ `best_openvino_model/`（本番はこちらを使う）。
2026-08-31に「Roboflowのリサイズを外したv4」で学習し直したもの。

**現在の基準スコア**（凍結検証セット29枚・`conf=0.30 / imgsz=640`）:

| クラス | 再現率 | 実害誤検出 | 重複 | 見逃し |
|---|---|---|---|---|
| band | **1.000** | 2 | 0 | 0 |
| logo | **1.000** | 3 | 0 | 0 |
| map | 0.870 | 3 | 0 | 3 |

帯を1つも出せなかったページ: **0 / 29**

`eval/baseline_best_pt.json` に保存済み。**新しいモデルはこれを下回ったら採用しない。**

### 蓄積データ（2026-09-25 時点）

```
321ページ（人が修正 51 / 自動で抜き取り 270）
band=327 logo=114 map=180 / 枠が0個のページ 13
本番の手直し率: 約3.6%（51 / 推定1401ページ）
```

---

## 次にやること

### 1. デスクトップ（RTX 3060）で再学習 ← 最優先

前回はColabでやったが、切断リスクと246MBの再ダウンロードがあるのでデスクトップに移す。
手順は下の「別のPCで始める」を参照。

```powershell
python tools\train_new_model.py kq1kq1/obiduke-training-data --epochs 150
```

判定が「採用してよい」なら、出てくる手順に従ってモデルを差し替える。

### 2. INT8量子化（検証済み・未適用）

**検出が1.74倍速くなり、凍結検証セットでの精度は完全に同じ**だった（実測済み）。
ただし検出枠がわずかに小さく出るので、`paste_rect()` のパディングを
`H * 0.006` → `0.010` に増やす必要がある。詳細はREADMEの該当節。

**再学習が終わってから**新しいモデルに対して適用する（先にやると作り直しになる）。

### 3. トークンの整理（未着手）

前回Colabにトークンを直書きしたノートブックがGoogleドライブに残っている可能性がある。
用途ごとにfine-grainedトークンを作り直すのが望ましい。

| 用途 | あるべき権限 |
|---|---|
| Space（本番アプリ） | `obiduke-training-data` に Write |
| 学習・採点 | `obiduke-training-data` に Read |

**順番が大事**: 新しいトークンに差し替えて動作確認してから、古いものを失効させる。
先に消すと本番の蓄積が止まる。

### 4. 検証セットの拡充（保留中・優先度低）

凍結検証セット29枚は **band も logo も再現率1.000（天井）** で、
**悪化は検出できるが改善は測れない**状態。

蓄積データで広げようとしたが詰まっている:
- 「人が修正した51ページ」だけがラベルを信頼できる。だが検証に回すと学習から
  最も価値のあるデータが消える
- 「自動で抜き取った270ページ」のラベルは**今のモデルの出力そのもの**なので、
  新モデルの採点に使うと「今のモデルにどれだけ似ているか」を測ることになる

**正攻法は「本番ページを100枚ほど無作為に選んでラベルを人の手で付け直す」。**
モデルの予測を下書きとしてRoboflowに読み込ませれば「描く」ではなく「直す」だけで済む。
当面は `tools/correction_rate.py`（本番の手直し率）で代用する。

---

## 別のPC（デスクトップ）で始める

### Git管理外だが必要なもの

| もの | ノートPCでのサイズ | 要否 | 入手方法 |
|---|---|---|---|
| `datasets/frozen_val/` | 22MB | **採点に必要** | `tools/restore_frozen_val.py` で復元 |
| `datasets/roboflow/v4/` | 271MB | 再学習には**不要** | 中身はHubの `base/train/` にある |
| `datasets/hub/` | 239MB | 自動で作られる | fetch / train が落とす |
| `venv/` | 1.6GB | 必要 | `tools/setup_train_env.ps1` が作る |
| `collected/` | 124MB | 不要 | fetch の出力。使い捨て |
| HFトークン | — | 必要 | `hf auth login` |
| `~/.ssh/id_ed25519` | — | **デプロイにのみ**必要 | HFに公開鍵を登録 |
| `outputs/` `uploads/` | 約100MB | 不要 | アプリ実行時の一時ファイル |

**コピーするものはゼロ。`git clone` ＋ `hf auth login` で足りる。**
ノートPCには合計2.3GBほどあるが、必要なのは22MBの凍結検証セットだけで、
それもHubから復元できる。Roboflowのエクスポート（271MB）も持ち回らなくてよい。

### 手順

```powershell
git clone https://github.com/kq1kq1/obiduke-kun
cd obiduke-kun
```

```powershell
.\tools\setup_train_env.ps1
```

GPU確認 → venv作成 → **CUDA版のtorch** → 依存 → 動作確認 まで一括。
失敗したら `-Cuda cu124` を試す。

> `requirements.txt` は **torchのCPU版を明示している**（HF Spacesの無料CPU環境向け）。
> そのまま入れるとGPUがあっても使われない。学習用は必ず上のスクリプトを使う。

```powershell
.\venv\Scripts\Activate.ps1
hf auth login
python tools\restore_frozen_val.py kq1kq1/obiduke-training-data
```

確認:

```powershell
python tools\eval_model.py best.pt --labels fullwidth
```

上の「現在の基準スコア」と同じ数字が出れば環境は正しい。

### デプロイもデスクトップからやるなら

`redeploy_hf.ps1` はSSH鍵で認証する。デスクトップで鍵を作ってHFに登録する。

```powershell
ssh-keygen -t ed25519 -C "your.email@example.com"
type $env:USERPROFILE\.ssh\id_ed25519.pub
```

出てきた公開鍵を https://huggingface.co/settings/keys に登録してから:

```powershell
ssh -T git@hf.co        # 「Hi kq1kq1」と出れば成功
```

---

## ハマりどころ（実際に踏んだもの）

| 症状 | 原因と対処 |
|---|---|
| 学習後に「学習結果が見つかりません」 | ultralyticsは `project` が相対パスだと `runs/detect/` の下に置く。修正済み（`--weights` で直接指定もできる） |
| ダウンロードが `e69de29bb2d1...` で失敗 | **空ファイル**のハッシュ。Windowsでシンボリックリンクが使えないとキャッシュ方式が0バイトファイルで落ちる。`local_dir` 方式に変更済み |
| `redeploy_hf.ps1` が最初のgitコマンドで止まる | `$ErrorActionPreference = "Stop"` ＋ PS5.1 では git が標準エラーに書くだけで止まる。修正済み |
| PowerShellスクリプトの日本語が化ける | **UTF-8 BOM が必要**（PS5.1はBOMが無いとANSIとして読む） |
| Colabが切断されて学習が消える | 無料Colabにバックグラウンド実行は無い。PCをスリープさせない。デスクトップならこの問題自体が無い |

---

## 設計上、変えてはいけないこと

1. **保存する枠の座標にパディングを足さない。**
   `label_rect()` が保存用、`paste_rect()` が貼り付け用。膨らんだ枠をラベルにすると
   「帯は実際より少し大きい」とモデルが覚えて精度が落ちる。

2. **bandのラベルは常に全幅**（`x_center=0.5, width=1.0`）。
   検出枠と手修正枠でx方向の規約が混ざると学習が濁る。

3. **学習データに送る画像は `_work`（回転後）を見る。**
   `_orig` に向けると回転したページで「画像は回転前・座標は回転後」になり、
   間違った正解データが静かに混ざる。`tools/selfcheck_rotation.py` が見張っている。

4. **凍結検証セットは学習に絶対に使わない。**
   `train_new_model.py` は train / val / frozen の3つに分け、`frozen` を
   `data.yaml` に書かない（ultralyticsが `best.pt` を選ぶのに使わせないため）。

5. **`eval/frozen_val_manifest.json` を作り直さない。**
   物差しが変わって過去のスコアと比較できなくなる。

---

## デプロイ前に必ず流すもの

```powershell
python tools\selfcheck_rotation.py
python tools\selfcheck_sampling.py
```

両方 `すべてOK` を確認してから `.\redeploy_hf.ps1`。

---

## 戻し方

| タグ | 内容 |
|---|---|
| `v1-prod-26.08.31` | 再学習機能を入れる前の本番状態 |
| `v1-backup-26.08.04` | 同上（3コミット手前を指している） |

```powershell
git show v1-prod-26.08.31           # 内容を確認
git checkout cd351bd~1 -- best.pt best_openvino_model   # モデルだけ前に戻す
```
