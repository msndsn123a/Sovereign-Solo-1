param(
    [int]$Port = 5556,
    [string]$QemuAccel = "",
    [string]$QemuCpu = "Skylake-Server,+avx512f,+avx512dq",
    [string]$ShardPath = "dist/trained_pure_solo_shard.bin",
    [string]$ExpectedPath = "dist/trained_pure_solo_expected.json",
    [string]$EfiPath = "target/x86_64-unknown-uefi/release/neural_box_core.efi"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
$qemuProcess = $null
$client = $null

try {
    if (-not (Test-Path $ShardPath)) {
        throw "Shard not found: $ShardPath"
    }
    if (-not (Test-Path $EfiPath)) {
        throw "UEFI binary not found: $EfiPath"
    }
    if (-not (Test-Path $ExpectedPath)) {
        throw "Expected-vector sidecar not found: $ExpectedPath"
    }

    $shardBytes = [System.IO.File]::ReadAllBytes((Resolve-Path $ShardPath))
    if ($shardBytes.Length -lt 96 -or [System.Text.Encoding]::ASCII.GetString($shardBytes, 0, 4) -ne "NEUR") {
        throw "Shard does not contain a complete signed native Solo record"
    }
    $version = [BitConverter]::ToUInt32($shardBytes, 4)
    $inputDim = [BitConverter]::ToUInt32($shardBytes, 8)
    $modelType = $shardBytes[12]
    $outputDim = [BitConverter]::ToUInt16($shardBytes, 13)
    if ($version -ne 2 -or $inputDim -ne 64 -or $modelType -ne 2 -or $outputDim -ne 1 -or $shardBytes[15] -ne 0) {
        throw "Shard header is not the native signed v2 64->1 Solo format"
    }
    $expected = Get-Content -Raw -Path $ExpectedPath | ConvertFrom-Json
    $testInputs = @($expected.test_batch_i8)
    $testOutputs = @($expected.test_batch_solo_i32)
    if ($testInputs.Count -lt 1 -or $testOutputs.Count -ne $testInputs.Count) {
        throw "Expected-vector sidecar dimensions do not match the NEUR shard"
    }
    foreach ($vector in $testInputs) {
        if (@($vector).Count -ne 64) { throw "Each QEMU input vector must have 64 values" }
    }
    foreach ($vector in $testOutputs) {
        if (@($vector).Count -ne 1) { throw "Each expected Solo output must contain one scalar" }
    }
    $shardHash = (Get-FileHash -Algorithm SHA256 -Path $ShardPath).Hash.ToLowerInvariant()
    if ($expected.shard_sha256 -ne $shardHash) {
        throw "Expected-vector sidecar hash does not match the shard being booted"
    }

    New-Item -ItemType Directory -Force -Path "esp/EFI/BOOT", "dist" | Out-Null
    Copy-Item $EfiPath "esp/EFI/BOOT/BOOTX64.EFI" -Force

    $qemu = (Get-Command qemu-system-x86_64 -ErrorAction Stop).Source
    $serialLog = "dist/qemu-uart-com1.log"
    $qemuArgs = @(
        "-bios", "assets/OVMF.fd",
        "-drive", "format=raw,file=fat:rw:esp",
        "-drive", "file=$ShardPath,if=none,id=nvm1,format=raw",
        "-device", "nvme,serial=deadbeef,drive=nvm1",
        "-cpu", $QemuCpu,
        "-net", "none",
        "-display", "none",
        "-monitor", "none",
        "-serial", "file:$serialLog",
        "-serial", "tcp:127.0.0.1:$Port,server=on,wait=off"
    )
    if ($QemuAccel) {
        $qemuArgs = @("-accel", $QemuAccel) + $qemuArgs
    }
    $qemuProcess = Start-Process -FilePath $qemu -ArgumentList $qemuArgs -PassThru -WindowStyle Hidden

    $connectDeadline = [DateTime]::UtcNow.AddSeconds(15)
    while (-not $client -and [DateTime]::UtcNow -lt $connectDeadline) {
        $candidate = [System.Net.Sockets.TcpClient]::new()
        try {
            $connect = $candidate.BeginConnect("127.0.0.1", $Port, $null, $null)
            if ($connect.AsyncWaitHandle.WaitOne(250)) {
                $candidate.EndConnect($connect)
                $client = $candidate
            } else {
                $candidate.Dispose()
            }
        } catch {
            $candidate.Dispose()
        }
        if ($qemuProcess.HasExited) {
            throw "QEMU exited before opening COM2 socket (exit code $($qemuProcess.ExitCode))"
        }
    }
    if (-not $client) {
        throw "Timed out connecting to QEMU COM2 socket on port $Port"
    }
    $stream = $client.GetStream()
    $stream.ReadTimeout = 30000
    $stream.WriteTimeout = 10000

    $readyBytes = [System.Collections.Generic.List[byte]]::new()
    $readyToken = [System.Text.Encoding]::ASCII.GetBytes("UART_READY`n")
    $readyFound = $false
    while ($readyBytes.Count -lt 4096 -and -not $readyFound) {
        $next = $stream.ReadByte()
        if ($next -lt 0) {
            throw "COM2 disconnected before UART_READY"
        }
        $readyBytes.Add([byte]$next)
        if ($readyBytes.Count -ge $readyToken.Length) {
            $match = $true
            $start = $readyBytes.Count - $readyToken.Length
            for ($index = 0; $index -lt $readyToken.Length; $index++) {
                if ($readyBytes[$start + $index] -ne $readyToken[$index]) {
                    $match = $false
                    break
                }
            }
            $readyFound = $match
        }
    }
    if (-not $readyFound) {
        throw "UART_READY handshake was not found in the initial COM2 stream"
    }
    Write-Host "[ROUNDTRIP]: Received COM2 readiness handshake."

    $expectedLength = 6
    for ($frameIndex = 0; $frameIndex -lt $testInputs.Count; $frameIndex++) {
        $inputFrame = New-Object byte[] 66
        $inputFrame[0] = 0x53
        $inputFrame[1] = 0x4F
        for ($index = 0; $index -lt 64; $index++) {
            $value = [int]$testInputs[$frameIndex][$index]
            if ($value -lt -128 -or $value -gt 127) { throw "Test input is outside int8 range" }
            $inputFrame[$index + 2] = [byte]($value -band 0xFF)
        }
        $stream.Write($inputFrame, 0, $inputFrame.Length)
        $stream.Flush()

        $response = New-Object byte[] $expectedLength
        $offset = 0
        while ($offset -lt $response.Length) {
            $received = $stream.Read($response, $offset, $response.Length - $offset)
            if ($received -le 0) {
                throw "COM2 disconnected before output frame $frameIndex was complete"
            }
            $offset += $received
        }

        if ($response[0] -ne 0x53 -or $response[1] -ne 0x52) {
            throw "Output frame $frameIndex did not contain the SR preamble"
        }
        $actual = [BitConverter]::ToInt32($response, 2)
        $expectedValue = [int]$testOutputs[$frameIndex][0]
        if ($actual -ne $expectedValue) {
            throw "Frame $frameIndex scalar mismatch: QEMU=$actual, Solo reference=$expectedValue"
        }
        $expectedBytes = [BitConverter]::GetBytes($expectedValue)
        for ($index = 0; $index -lt 4; $index++) {
            if ($response[$index + 2] -ne $expectedBytes[$index]) {
                throw "Frame $frameIndex scalar byte $index differs from little-endian reference bytes"
            }
        }
    }
    Write-Host "[ROUNDTRIP]: QEMU returned $($testInputs.Count) output frames matching PyTorch predictions byte-for-byte."
    Write-Host ("[ROUNDTRIP]: final TX bytes: " + [System.BitConverter]::ToString($response))

    if (-not $qemuProcess.WaitForExit(15000)) {
        throw "QEMU did not exit after the configured one-frame limit"
    }
    if (Test-Path $serialLog) {
        $telemetry = Select-String -Path $serialLog -Pattern "\[GUEST CPUID\]:|\[SIMD\]:|\[SHM\]: Initialized Mailbox|\[SOLO VERIFY\]|\[SOLO MODEL\]|\[SHARD\]: MAGIC|\[SECURITY\]: Shard signature valid|\[ALLOC\]:|\[LATENCY\]: shared_mailbox_|\[TIMER\]: Calibrated TSC frequency|\[TIMER\]: Invariant|\[TIMER WARNING\]|\[CPU POWER\]|\[LATENCY\]:|\[UART\]: RX frames=$($testInputs.Count), TX frames=$($testInputs.Count)|\[RING\]: Atomic SPSC"
        $telemetry | ForEach-Object { Write-Host $_.Line }
        if (-not ($telemetry.Line -match "\[SHM\]: Initialized Mailbox at physical addr .*alignment=64")) {
            throw "COM1 log did not confirm an aligned shared mailbox allocation"
        }
        if (-not ($telemetry.Line -match "\[SOLO VERIFY\]: scalar mailbox parity=match; frames=16, drops=0")) {
            throw "COM1 log did not confirm Solo scalar parity and zero drops"
        }
        if (-not ($telemetry.Line -match "\[SHARD\]: MAGIC=0x4E455552, VERSION=2, INPUT_DIM=64, MODEL_TYPE=2, HIDDEN_OR_ATTN_DIM=0, OUTPUT_DIM=1")) {
            throw "COM1 log did not confirm the signed v2 NEUR header dimensions"
        }
        if (-not ($telemetry.Line -match "\[SECURITY\]: Shard signature valid \(Ed25519 verified\)")) {
            throw "COM1 log did not confirm Ed25519 shard authentication"
        }
        if (-not ($telemetry.Line -match "\[SOLO MODEL\]: authenticated native Solo payload ingested directly; output_dim=1, recurrent_state=disabled")) {
            throw "COM1 log did not confirm direct native Solo ingestion"
        }
        if (-not ($telemetry.Line -match "\[TIMER\]: Calibrated TSC frequency")) {
            throw "COM1 log did not contain the TSC calibration report"
        }
        if (-not ($telemetry.Line -match "\[LATENCY\]: turnaround cycles min=")) {
            throw "COM1 log did not contain end-to-end turnaround statistics"
        }
    }
} finally {
    if ($client) {
        $client.Dispose()
    }
    if ($qemuProcess -and -not $qemuProcess.HasExited) {
        Stop-Process -Id $qemuProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Pop-Location
}