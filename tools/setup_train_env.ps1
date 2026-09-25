# GPUを積んだPCに、再学習だけできる環境を作るスクリプト。
#
# やること:
#   1. GPUとドライバを確認
#   2. venv を作る（無ければ）
#   3. CUDA版の torch を入れる（ここが requirements.txt と違うところ）
#   4. 残りの依存を入れる
#   5. GPUが本当に使えるか確かめる
#
# 使い方（リポジトリのルートで）:
#   .\tools\setup_train_env.ps1
#   .\tools\setup_train_env.ps1 -Cuda cu124     # ドライバが古くて動かないとき
#
# 注意: これは学習用。アプリを動かすだけなら requirements.txt を使う。
#   requirements.txt は torch のCPU版を明示している（HF Spacesの無料CPU環境向け）ので、
#   そのまま入れるとGPUがあっても使われない。

param(
    # PyTorchのCUDAビルド。ドライバが古くて動かないときは cu124 を試す。
    [string]$Cuda = "cu128"
)

$ErrorActionPreference = "Continue"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

function Assert-Ok {
    param([string]$What)
    if ($LASTEXITCODE -ne 0) { throw "$What に失敗しました（終了コード $LASTEXITCODE）" }
}

try {
    Write-Host "`n[1/5] GPUを確認..." -ForegroundColor Cyan
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) {
        Write-Host "  nvidia-smi が見つかりません。" -ForegroundColor Yellow
        Write-Host "  NVIDIAのGPUとドライバが入っているか確認してください。" -ForegroundColor Yellow
        Write-Host "  （GPUが無いPCなら、このスクリプトは不要です）" -ForegroundColor Yellow
        exit 1
    }
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    Assert-Ok "GPUの確認"

    Write-Host "`n[2/5] venv を用意..." -ForegroundColor Cyan
    $py = Join-Path (Get-Location) "venv\Scripts\python.exe"
    if (-not (Test-Path $py)) {
        python -m venv venv
        Assert-Ok "venvの作成"
        Write-Host "  作成しました"
    } else {
        Write-Host "  既にあります"
    }

    Write-Host "`n[3/5] CUDA版のtorchを入れる（$Cuda・数GBあるので数分かかります）..." -ForegroundColor Cyan
    & $py -m pip install --upgrade pip --quiet
    & $py -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
    Assert-Ok "torchのインストール"

    Write-Host "`n[4/5] 残りの依存を入れる..." -ForegroundColor Cyan
    & $py -m pip install -r requirements-train.txt
    Assert-Ok "依存のインストール"

    Write-Host "`n[5/5] GPUが使えるか確認..." -ForegroundColor Cyan
    $check = @'
import sys
import torch
print("  torch      :", torch.__version__)
print("  CUDA usable:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  GPU        :", torch.cuda.get_device_name(0))
    print("  VRAM       : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
else:
    sys.exit(1)
'@
    & $py -c $check
    if ($LASTEXITCODE -ne 0) {
        throw "CUDAが使えません。-Cuda cu124 で試すか、GPUドライバを更新してください"
    }

    Write-Host "`n準備できました。次はこれ:" -ForegroundColor Green
    Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Green
    Write-Host "  hf auth login" -ForegroundColor Green
    Write-Host "  python tools\train_new_model.py kq1kq1/obiduke-training-data --epochs 150" -ForegroundColor Green
}
catch {
    Write-Host "`n失敗しました: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "CUDAのビルドが合っていない可能性があります。-Cuda cu124 も試してください。" -ForegroundColor Yellow
    exit 1
}
