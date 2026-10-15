# Creates a Lytt shortcut on the desktop.
# Run:  powershell -ExecutionPolicy Bypass -File "Create shortcuts.ps1"
# Start-with-Windows is a setting inside Lytt (Settings -> Window), off by default.

$here    = Split-Path -Parent $MyInvocation.MyCommand.Path
$target  = "$env:SystemRoot\System32\wscript.exe"
$vbsArg  = "`"$here\Start Lytt.vbs`""
$icon    = "$here\ui\lytt.ico"
$sh      = New-Object -ComObject WScript.Shell

$path = [Environment]::GetFolderPath("Desktop") + "\Lytt.lnk"
$s = $sh.CreateShortcut($path)
$s.TargetPath       = $target
$s.Arguments        = $vbsArg
$s.WorkingDirectory = $here
$s.IconLocation     = $icon
$s.Description      = "Lytt - local speech to text"
$s.Save()
Write-Host "Created: $path"
