param([switch]$Restart)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runtime = Join-Path $project 'data/runtime'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$python = Join-Path $project '.venv-local/Scripts/python.exe'
$application = 'http://127.0.0.1:8000/health'

function Get-LocalHealth($url) {
    try { Invoke-RestMethod -Uri $url -TimeoutSec 1 }
    catch { $null }
}

function Wait-LocalHealth($url, $process, $errorLog) {
    for ($attempt = 0; $attempt -lt 35; $attempt++) {
        if ($process.HasExited) { throw "服务启动失败，日志位于 $errorLog" }
        $health = Get-LocalHealth $url
        if ($health.status -eq 'ok') { return $health }
        Start-Sleep -Milliseconds 400
    }
    throw "服务启动超时，日志位于 $errorLog"
}

function Start-LocalProcess($file, $arguments, $logName) {
    $options = @{
        FilePath = $file
        ArgumentList = $arguments
        WorkingDirectory = $project
        WindowStyle = 'Hidden'
        PassThru = $true
        RedirectStandardOutput = (Join-Path $runtime "$logName.log")
        RedirectStandardError = (Join-Path $runtime "$logName-error.log")
    }
    Start-Process @options
}

$status = Get-LocalHealth $application
if ($Restart -and $status) {
    $appLog = Join-Path $runtime 'app-error.log'
    $started = Get-Content -LiteralPath $appLog -TotalCount 8 |
        Select-String 'Started server process \[(\d+)\]' | Select-Object -First 1
    if (-not $started) { throw '找不到本脚本启动的应用进程。请先关闭当前应用进程。' }
    $processId = $started.Matches[0].Groups[1].Value
    $running = Get-CimInstance Win32_Process -Filter "ProcessId=$processId"
    if (-not $running -or $running.CommandLine -notmatch ' -m app\.main\s*$') {
        throw '当前进程与应用启动日志不符。请先关闭当前应用进程。'
    }
    Stop-Process -Id $processId
    Wait-Process -Id $processId -Timeout 10 -ErrorAction SilentlyContinue
    $status = $null
}
if (-not $status) {
    $process = Start-LocalProcess $python @('-m', 'app.main') 'app'
    $status = Wait-LocalHealth $application $process (Join-Path $runtime 'app-error.log')
}
$status | ConvertTo-Json -Compress
Write-Output 'http://127.0.0.1:8000/'
