# 帯付けくん 再デプロイスクリプト（Hugging Face Spaces）
#
# やること:
#   1. 今の main を GitHub(origin) に push
#   2. バイナリを Git LFS で管理した単一コミット(orphan)を作って HF Space に force push
#   3. 元のブランチに戻して一時ブランチを削除
#
# 使い方（PowerShellで）:
#   .\redeploy_hf.ps1
#   （HFの Write トークンを聞かれるので貼る。トークンは画面に表示されず、保存もされない）
#
# 事前に: 変更を main にコミット済みにしておくこと（このスクリプトはコミットはしない）。
#
# 注意: $ErrorActionPreference を "Stop" にしないこと。
#   Windows PowerShell 5.1 では、gitが標準エラーに1行書くだけで終了扱いになる。
#   「消すブランチが無い」程度のことでスクリプト全体が止まり、しかも後始末の
#   エラーが元のエラーを覆い隠して原因が分からなくなる。
#   代わりに $LASTEXITCODE を都度見る。

$ErrorActionPreference = "Continue"
$Space = "https://huggingface.co/spaces/kq1kq1/obiduke-kun"
$HfUser = "kq1kq1"
$DeployBranch = "hf-deploy"

# リポジトリのルートへ移動（スクリプトのある場所）
Set-Location -Path $PSScriptRoot

function Assert-Git {
    param([string]$What)
    if ($LASTEXITCODE -ne 0) { throw "$What に失敗しました（git 終了コード $LASTEXITCODE）" }
}

function Remove-DeployBranch {
    # 一時ブランチが在れば消す。無いときにエラーにしない（ここで止まるのが以前の不具合）
    git show-ref --verify --quiet "refs/heads/$DeployBranch"
    if ($LASTEXITCODE -eq 0) { git branch -D $DeployBranch 2>&1 | Out-Null }
}

# 未コミットの変更があれば警告して中断（事故防止）
$dirty = git status --porcelain
if ($dirty) {
    Write-Host "未コミットの変更があります。先に main へコミットしてください:" -ForegroundColor Yellow
    Write-Host $dirty
    exit 1
}

# 失敗しても必ずここへ戻れるように、今のブランチを覚えておく
$startBranch = git rev-parse --abbrev-ref HEAD
Assert-Git "現在のブランチの取得"
if ($startBranch -ne "main") {
    Write-Host "注意: 今いるブランチは '$startBranch' です。" -ForegroundColor Yellow
    Write-Host "      このスクリプトは main の内容をデプロイする前提です。" -ForegroundColor Yellow
    $ans = Read-Host "このまま '$startBranch' の内容をデプロイしますか？ (y/N)"
    if ($ans -ne "y") { Write-Host "中断しました。"; exit 1 }
}

# HFトークンを安全に入力（画面非表示）
$secure = Read-Host -Prompt "HFのWriteトークンを貼ってEnter" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$token = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
if (-not $token) { Write-Host "トークンが空です。中断します。" -ForegroundColor Red; exit 1 }

$ok = $false
try {
    Write-Host "`n[1/3] GitHub(origin) に push..." -ForegroundColor Cyan
    git push origin $startBranch
    Assert-Git "GitHubへのpush"

    Write-Host "`n[2/3] HFデプロイ用の単一コミットを作成..." -ForegroundColor Cyan
    Remove-DeployBranch
    git checkout --orphan $DeployBranch
    Assert-Git "orphanブランチの作成"

    git rm -r --cached . | Out-Null
    git lfs track "*.pt" "*.png" "*.jpg" "*.jpeg" "*.bin" | Out-Null
    git add .gitattributes
    git add -A
    Assert-Git "デプロイ内容の追加"
    git commit -m "HFデプロイ" | Out-Null
    Assert-Git "デプロイ用コミットの作成"

    Write-Host "`n[3/3] HF Space に force push..." -ForegroundColor Cyan
    $remote = "https://$($HfUser):$($token)@huggingface.co/spaces/kq1kq1/obiduke-kun"
    git push $remote "$($DeployBranch):main" --force
    Assert-Git "HFへのpush"
    $ok = $true
}
catch {
    Write-Host "`nデプロイに失敗しました: $($_.Exception.Message)" -ForegroundColor Red
}
finally {
    # 必ず元のブランチに戻して一時ブランチを片付ける（失敗時も）
    git checkout $startBranch 2>&1 | Out-Null
    Remove-DeployBranch
}

if ($ok) {
    Write-Host "`nデプロイ完了: $Space" -ForegroundColor Green
    Write-Host "  反映まで数分かかります。Space の画面上部が Building から Running に" -ForegroundColor Green
    Write-Host "  変わるまで待ってから動作確認してください。" -ForegroundColor Green
} else {
    Write-Host "`nHFには何も送られていません。リポジトリは元のままです。" -ForegroundColor Yellow
    exit 1
}
