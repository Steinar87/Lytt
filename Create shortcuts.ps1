# Creates a Lytt shortcut on the desktop and (optionally) in the Startup folder.
# Run:  powershell -ExecutionPolicy Bypass -File "Create shortcuts.ps1" [-Startup]
param([switch]$Startup)

$here    = Split-Path -Parent $MyInvocation.MyCommand.Path
$target  = "$env:SystemRoot\System32\wscript.exe"
$vbsArg  = "`"$here\Start Lytt.vbs`""
$icon    = "$here\ui\lytt.ico"
$sh      = New-Object -ComObject WScript.Shell

function Make-Shortcut($path) {
    $s = $sh.CreateShortcut($path)
    $s.TargetPath       = $target
    $s.Arguments        = $vbsArg
    $s.WorkingDirectory = $here
    $s.IconLocation     = $icon
    $s.Description      = "Lytt - local speech to text"
    $s.Save()
    Write-Host "Created: $path"
}

Make-Shortcut ([Environment]::GetFolderPath("Desktop") + "\Lytt.lnk")
if ($Startup) {
    Make-Shortcut ([Environment]::GetFolderPath("Startup") + "\Lytt.lnk")
    Write-Host "Lytt will now start automatically at login."
}
