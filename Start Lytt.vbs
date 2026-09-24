' Starts Lytt without a console window. The app lives in the system tray next to the clock.
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("WScript.Shell")
here = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = here
sh.Run """" & here & "\.venv\Scripts\pythonw.exe"" """ & here & "\app.py""", 0, False
