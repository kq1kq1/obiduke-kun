# 帯付けくん 再デプロイスクリプト（Hugging Face Spaces）
#
# やること:
#   1. 今の main を GitHub(origin) に push
#   2. バイナリを Git LFS で管理した単一コミット(orphan)を作って HF Space に force push
#   3. main に戻して一時ブランチを削除
#
# 使い方（PowerShellで）:
#   .\redeploy_hf.ps1
#   （HFの Write トークンを聞かれるので貼る。トークンは画面に表示されず、保存もされない）
#
# 事前に: 変更を main にコミット済みにしておくこと（このスクリプトはコミットはしない）。

$ErrorActionPreference = "Stop"
$Space = "https://huggingface.co/spaces/kq1kq1/obiduke-kun"
$HfUser = "kq1kq1"
$DeployBranch = "hf-deploy"

# リポジトリのルートへ移動（スクリプトのある場所）
Set-Location -Path $PSScriptRoot

# 未コミットの変更があれば警告して中断（事故防止）
$dirty = git status --porcelain
if ($dirty) {
    Write-Host "未コミットの変更があります。先に main へコミットしてください:" -ForegroundColor Yellow
    Write-Host $dirty
    exit 1
}

# HFトークンを安全に入力（画面非表示）
$secure = Read-Host -Prompt "HFのWriteトークンを貼ってEnter" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$token = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
if (-not $token) { Write-Host "トークンが空です。中断します。" -ForegroundColor Red; exit 1 }

try {
    Write-Host "`n[1/3] GitHub(origin) に push..." -ForegroundColor Cyan
    git push origin main
    if ($LASTEXITCODE -ne 0) { throw "GitHubへのpushに失敗しました" }

    Write-Host "`n[2/3] HFデプロイ用の単一コミットを作成..." -ForegroundColor Cyan
    # 既存の一時ブランチがあれば消す
    git branch -D $DeployBranch 2>$null
    git checkout --orphan $DeployBranch
    if ($LASTEXITCODE -ne 0) { throw "orphanブランチ作成に失敗しました" }

    git rm -r --cached . | Out-Null
    git lfs track "*.pt" "*.png" "*.jpg" "*.jpeg" "*.bin" | Out-Null
    git add .gitattributes
    git add -A
    git commit -m "HFデプロイ" | Out-Null

    Write-Host "`n[3/3] HF Space に force push..." -ForegroundColor Cyan
    $remote = "https://$($HfUser):$($token)@huggingface.co/spaces/kq1kq1/obiduke-kun"
    git push $remote "$($DeployBranch):main" --force
    if ($LASTEXITCODE -ne 0) { throw "HFへのpushに失敗しました" }

    Write-Host "`n✅ デプロイ完了: $Space" -ForegroundColor Green
}
finally {
    # 必ず main に戻して一時ブランチを片付ける（失敗時も）
    git checkout main 2>$null
    git branch -D $DeployBranch 2>$null
}
