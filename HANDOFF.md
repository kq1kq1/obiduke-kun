# 引き継ぎ（2026-09-26 時点）

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
| 学習データの自動蓄積 | **動いている**。338ページ貯まっている |
| 再学習の環境 | **デスクトップ（RTX 3060）に構築済み**。150エポックで約9分 |
| ページ回転（90度単位） | 実装済み・デプロイ済み |
| 枠の追加ボタン | 帯 / 案内図 / 白塗り の3つ |

### 使っているモデル

`best.pt`（YOLOv8n）＋ `best_openvino_model/`（本番はこちらを使う・FP32）。
**2026-09-26 にデスクトップで再学習したもの**。前のモデル（2026-08-31・Roboflow v4 のみ）から
続きを150エポック。学習525枚のうち蓄積データ338ページを初めて入れた。

**現在の基準スコア**（凍結検証セット29枚・`conf=0.30 / imgsz=640`・`--labels fullwidth`）:

| クラス | 再現率 | 実害誤検出 | 重複 | 見逃し | 前のモデルとの差 |
|---|---|---|---|---|---|
| band | **1.000** | 2 | 0 | 0 | 同じ |
| logo | **1.000** | 4 | 0 | 0 | 誤検出+1（帯の枠の中なので実害なし） |
| map | 0.870 | 2 | 1 | 3 | 誤検出-1 |

帯を1つも出せなかったページ: **0 / 29**

`eval/baseline_best_pt.json` に保存済み。**新しいモデルはこれを下回ったら採用しない。**

**本番形式（OpenVINO）では map の見逃しが 3 → 4 に増えている**
（`5-30-2-pdf_page_1` の案内図。前のモデルは確信度0.95で出せていたが、新モデルは出せない）。
案内図の白塗りは既定オフで、実際に誰も使っていないため**実害なしと判断して採用した**。
案内図の白塗りを使うようになったら要注意。→「次にやること」の3

### 蓄積データ（2026-09-26 時点）

```
338ページ（人が修正 54 / 自動で抜き取り 284）
band=340 logo=117 map=195（案内図のあるページ 176）/ 枠が0個のページ 16
本番の手直し率: 約3.66%（54 / 推定1474ページ）← 前のモデルでの値
```

---

## 次にやること

### 1. INT8量子化（検証済み・未適用）← 次

**検出が1.74倍速くなり、凍結検証セットでの精度は完全に同じ**だった（前のモデルで実測）。
ただし検出枠がわずかに小さく出るので、`paste_rect()` のパディングを
`H * 0.006` → `0.010` に増やす必要がある。詳細はREADMEの該当節。

再学習が終わったので、**2026-09-26 のモデルに対して検証をやり直してから**適用する。
キャリブレーションには `training_run/dataset/data.yaml`（学習データ）が使える。

### 2. 差し替えの効果を本番で測る（数週間後）

凍結検証セットは天井なので、良くなったかは本番の手直し率でしか分からない。
数百ページ貯まってから:

```powershell
.\venv\Scripts\python.exe tools\fetch_training_data.py kq1kq1/obiduke-training-data --cache training_run\hub   # 増えた分だけ落とす
.\venv\Scripts\python.exe tools\correction_rate.py --data training_run\hub\data --since 2026-09-26
```

前のモデルでの値は **3.66%**。これより下がっていれば本当に良くなっている。

### 3. 学習データの品質（保留中）

今回の再学習で、前のモデルが確信度0.95で出せていた案内図を新モデルが出せなくなった。
**蓄積データに案内図のラベルは入っている**（338ページ中176ページ・195個。
アプリは白塗りしない案内図もラベルとして記録する）ので、「塗っていないから学習されない」
ではない。原因は未調査。疑わしいもの:

- 自動で抜き取った284ページのラベルは前のモデルの出力そのもの。前のモデルの
  見逃し・誤検出がそのまま「正解」として入っている
- レビュー画面で案内図の枠を消したページがあると「ここは案内図ではない」と教えることになる

**正攻法は、落としたデータを人が手直ししてから学習に使うこと**（下の4と同じ作業で兼ねられる）。

あわせて、判定ツール（`train_new_model.py` の `judge()`）の弱点:
- **採点が PyTorch 版だけで、本番の OpenVINO 版を見ていない**。今回、本番形式でだけ
  map の見逃しが増えていた。OpenVINO 版でも採点するようにしたい
- 「どれか1つでも良くなれば採用」なので、良くなったものと悪くなったものが混ざっていても
  「採用してよい」と出る

### 4. トークンの整理（一部済み）

用途ごとにfine-grainedトークンを作り直す。

| 用途 | あるべき権限 | 状態 |
|---|---|---|
| Space（本番アプリ） | `obiduke-training-data` に Write | 未 |
| 学習・採点 | `obiduke-training-data` に Read | **デスクトップで作成・ログイン済み** |

古いトークンで失効させるべきもの:
- 前回Colabに直書きしたノートブック（Googleドライブに残っている可能性）
- **ノートPCの `.git/config`（LFS設定のURL）に平文で埋め込まれていたトークン**。
  デスクトップ側では削除済み。**ノートPC側の `.git/config` にはまだ残っている**

**順番が大事**: Spaceのトークンを新しいものに差し替えて動作確認してから、古いものを失効させる。
先に消すと本番の蓄積が止まる。

### 5. 検証セットの拡充（保留中・優先度低）

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
powershell -ExecutionPolicy Bypass -File .\tools\setup_train_env.ps1
```

GPU確認 → venv作成 → **CUDA版のtorch** → 依存 → 動作確認 まで一括。
失敗したら `-Cuda cu124` を試す。

> Windowsの初期設定では `.ps1` の実行が禁止されているので、`-ExecutionPolicy Bypass` を付けて呼ぶ。
> 同じ理由で `.\venv\Scripts\Activate.ps1` も動かないので、venv は有効化せず
> `.\venv\Scripts\python.exe` / `.\venv\Scripts\hf.exe` とパスで呼ぶ（以下すべてこの形）。
>
> `requirements.txt` は **torchのCPU版を明示している**（HF Spacesの無料CPU環境向け）。
> そのまま入れるとGPUがあっても使われない。学習用は必ず上のスクリプトを使う。

```powershell
.\venv\Scripts\hf.exe auth login
.\venv\Scripts\python.exe tools\restore_frozen_val.py kq1kq1/obiduke-training-data
```

確認:

```powershell
.\venv\Scripts\python.exe tools\eval_model.py best.pt --labels fullwidth
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
| `setup_train_env.ps1` が「CUDAが使えません」で終わる | torchは正しく入っていた。PS5.1 が `python -c` に渡した文字列の `"` を壊して SyntaxError になっていた。修正済み（標準入力から渡す） |
| ローカルで学習した後のデプロイで、非公開の学習データが公開Spaceに上がりかけた | `training_run/`（学習データ一式が入る）が `.gitignore` に無く、`redeploy_hf.ps1` の `git add -A` で拾われる。追加済み（`weights/` も） |
| PyTorch版では良いのに本番形式では悪い | OpenVINO版は入力の余白の付け方が違い、しきい値付近の検出が変わる。**採用前に `eval_model.py best_openvino_model --labels fullwidth` でも採点する** |
| 採用手順どおりに基準を更新したら物差しがずれる | `--labels fullwidth` が抜けていた。`labels` と `labels_fullwidth` は中身が違う。修正済み |

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
.\venv\Scripts\python.exe tools\selfcheck_rotation.py
.\venv\Scripts\python.exe tools\selfcheck_sampling.py
```

両方 `すべてOK` を確認してから、コミットして:

```powershell
powershell -ExecutionPolicy Bypass -File .\redeploy_hf.ps1
```

---

## 戻し方

| タグ | 内容 |
|---|---|
| `v1-prod-26.08.31` | 再学習機能を入れる前の本番状態 |
| `v1-backup-26.08.04` | 同上（3コミット手前を指している） |

```powershell
git show v1-prod-26.08.31           # 内容を確認
git checkout 2c21e7e -- best.pt best_openvino_model eval/baseline_best_pt.json   # 2026-08-31 のモデルに戻す
git checkout cd351bd~1 -- best.pt best_openvino_model   # さらに前（08-31の再学習前）に戻す
```

モデルを戻すときは、基準スコア（`eval/baseline_best_pt.json`）も一緒に戻すこと。
