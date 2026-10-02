param(
    [int]$Port = 5569,
    [ValidateRange(1, 64)][int]$CpuCount = 2,
    [string]$QemuAccel = "",
    [string]$QemuCpu = "Skylake-Server,+avx512f,+avx512dq",
    [string]$ShardPath = "dist/attention_test_shard.bin",
    [string]$EfiPath = "target/x86_64-unknown-uefi/release/neural_box_core.efi"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
$qemuProcess = $null
$client = $null

try {
    cargo +nightly build --target x86_64-unknown-uefi --release
    if ($LASTEXITCODE -ne 0) { throw "UEFI release build failed: $LASTEXITCODE" }
    & "$PSScriptRoot/payload_builder/build_payload.ps1" -outPath $ShardPath -model attention -quant pot -potScale 0 -multiStream
    if ($LASTEXITCODE -ne 0) { throw "Attention shard generation failed: $LASTEXITCODE" }
    if (-not (Test-Path $EfiPath)) { throw "UEFI binary not found: $EfiPath" }

    New-Item -ItemType Directory -Force -Path "esp/EFI/BOOT", "dist" | Out-Null
    Copy-Item $EfiPath "esp/EFI/BOOT/BOOTX64.EFI" -Force
    $qemu = (Get-Command qemu-system-x86_64 -ErrorAction Stop).Source
    $serialLog = "dist/qemu-attention-sequence-com1.log"
    $qemuArgs = @(
        "-bios", "assets/OVMF.fd",
        "-drive", "format=raw,file=fat:rw:esp",
        "-drive", "file=$ShardPath,if=none,id=nvm1,format=raw",
        "-device", "nvme,serial=deadbeef,drive=nvm1",
        "-smp", "$CpuCount",
        "-cpu", $QemuCpu,
        "-net", "none",
        "-display", "none",
        "-monitor", "none",
        "-serial", "file:$serialLog",
        "-serial", "tcp:127.0.0.1:$Port,server=on,wait=off"
    )
    if ($QemuAccel) { $qemuArgs = @("-accel", $QemuAccel) + $qemuArgs }
    $qemuProcess = Start-Process -FilePath $qemu -ArgumentList $qemuArgs -PassThru -WindowStyle Hidden

    $connectDeadline = [DateTime]::UtcNow.AddSeconds(20)
    while (-not $client -and [DateTime]::UtcNow -lt $connectDeadline) {
        $candidate = [System.Net.Sockets.TcpClient]::new()
        try {
            $connect = $candidate.BeginConnect("127.0.0.1", $Port, $null, $null)
            if ($connect.AsyncWaitHandle.WaitOne(250)) {
                $candidate.EndConnect($connect)
                $client = $candidate
            } else { $candidate.Dispose() }
        } catch { $candidate.Dispose() }
        if ($qemuProcess.HasExited) { throw "QEMU exited before opening COM2 (exit $($qemuProcess.ExitCode))" }
    }
    if (-not $client) { throw "Timed out connecting to QEMU COM2 on port $Port" }

    $stream = $client.GetStream()
    $stream.ReadTimeout = 30000
    $stream.WriteTimeout = 10000
    $ready = [System.Collections.Generic.List[byte]]::new()
    $readyToken = [System.Text.Encoding]::ASCII.GetBytes("UART_READY`n")
    $readyFound = $false
    while ($ready.Count -lt 4096 -and -not $readyFound) {
        $value = $stream.ReadByte()
        if ($value -lt 0) { throw "COM2 disconnected before UART_READY" }
        $ready.Add([byte]$value)
        if ($ready.Count -ge $readyToken.Length) {
            $offset = $ready.Count - $readyToken.Length
            $readyFound = $true
            for ($index = 0; $index -lt $readyToken.Length; $index++) {
                if ($ready[$offset + $index] -ne $readyToken[$index]) { $readyFound = $false; break }
            }
        }
    }
    if (-not $readyFound) { throw "UART_READY handshake was not found" }
    Write-Host "[ATTENTION TEST]: UART_READY received."

    $script:soloOutput = $null
    function Send-SoloFrame([byte]$streamId) {
        $request = New-Object byte[] 67
        $request[0] = 0x4E
        $request[1] = 0x53
        $request[2] = $streamId
        for ($index = 3; $index -lt $request.Length; $index++) { $request[$index] = 1 }
        $stream.Write($request, 0, $request.Length)
        $stream.Flush()

        $response = New-Object byte[] 7
        $offset = 0
        while ($offset -lt $response.Length) {
            $count = $stream.Read($response, $offset, $response.Length - $offset)
            if ($count -le 0) { throw "COM2 disconnected before Solo output frame" }
            $offset += $count
        }
        if ($response[0] -ne 0x4E -or $response[1] -ne 0x52 -or $response[2] -ne 1) {
            throw "Solo response header mismatch"
        }
        $actual = [BitConverter]::ToInt32($response, 3)
        if ($null -eq $script:soloOutput) {
            $script:soloOutput = $actual
        } elseif ($actual -ne $script:soloOutput) {
            throw "Stateless Solo output changed across calls: first=$script:soloOutput, stream=$streamId, actual=$actual"
        }
        Write-Host "[SOLO STATE TEST]: stream_tag=$streamId output=$actual (identical across calls)"
    }

    # Repeated frames with different transport stream tags must not accumulate state.
    Send-SoloFrame 2
    Send-SoloFrame 5
    Send-SoloFrame 5
    Send-SoloFrame 2

    # Selective NR is accepted as a stateless no-op and keeps the session alive.
    $targetedReset = [byte[]](0x4E, 0x52, 0x05)
    $stream.Write($targetedReset, 0, $targetedReset.Length)
    $stream.Flush()
    Send-SoloFrame 5
    Send-SoloFrame 2

    # A standalone NR remains the legacy stream-0 reset/exit control.
    $reset = [byte[]](0x4E, 0x52)
    $stream.Write($reset, 0, $reset.Length)
    $stream.Flush()
    Write-Host "[SOLO STATE TEST]: sent stateless NR+5 control and standalone NR exit."

    if (-not $qemuProcess.WaitForExit(15000)) { throw "QEMU did not exit after the NR reset event" }
    if ($qemuProcess.ExitCode -ne 0) { throw "QEMU exited with code $($qemuProcess.ExitCode)" }

    $expectedAps = $CpuCount - 1
    $required = @(
        "\[SHARD\]: .*MODEL_TYPE=1.*OUTPUT_DIM=16",
        "\[SHARD CONFIG\]: multi_stream=true, quant_type=1, pot_scale=0, block_sparse=false, activation_lut=false",
        "\[SMP\]: $expectedAps/$expectedAps Application Processors awakened and parked",
        "\[SOLO MODEL\]: authenticated shard retained; projection=row 0, output_dim=1, recurrent_state=disabled",
        "\[SOLO CONTROL\]: command=NR reset_count=1 stream=5 stateless=true",
        "\[UART\]: RX frames=6, TX frames=6, reset_commands=2, stream_events=8, ring_full_drops=0"
    )
    $matched = Select-String -Path $serialLog -Pattern $required
    $matched | ForEach-Object { Write-Host $_.Line }
    foreach ($pattern in $required) {
        if (-not (Select-String -Path $serialLog -Pattern $pattern -Quiet)) {
            throw "COM1 log did not contain required verification pattern: $pattern"
        }
    }
    Write-Host "[SOLO STATE TEST]: PASS; repeated inference remained identical across stream tags and reset controls, with $CpuCount-CPU appliance execution clean."
} finally {
    if ($client) { $client.Dispose() }
    if ($qemuProcess -and -not $qemuProcess.HasExited) {
        Stop-Process -Id $qemuProcess.Id -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $qemuProcess.Id -ErrorAction SilentlyContinue
    }
    Pop-Location
}
