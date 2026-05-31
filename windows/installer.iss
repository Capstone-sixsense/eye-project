[Setup]
AppName=Eye Project
AppVersion=1.0.0
AppPublisher=Capstone-sixsense
AppPublisherURL=https://github.com/Capstone-sixsense/eye-project
DefaultDirName={autopf}\EyeProject
DefaultGroupName=Eye Project
OutputBaseFilename=eye_project_setup_v1.0.0
Compression=lzma2/ultra64
SolidCompression=yes
OutputDir=installer_output
; Minimum Windows 10
MinVersion=10.0
PrivilegesRequired=lowest

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Files]
; Launcher
Source: "dist\EyeProject.exe"; DestDir: "{app}"; Flags: ignoreversion

; Backend folder (PyInstaller output)
Source: "dist\eye_backend\*"; DestDir: "{app}\eye_backend"; Flags: ignoreversion recursesubdirs createallsubdirs

; Frontend folder (Flutter build output)
Source: "dist\eye_frontend\*"; DestDir: "{app}\eye_frontend"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Desktop shortcut
Name: "{autodesktop}\Eye Project"; Filename: "{app}\EyeProject.exe"
; Start menu
Name: "{group}\Eye Project"; Filename: "{app}\EyeProject.exe"
; Uninstall from start menu
Name: "{group}\Uninstall Eye Project"; Filename: "{uninstallexe}"

[Run]
; Run after install (optional checkbox)
Filename: "{app}\EyeProject.exe"; Description: "Eye Project 실행"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Remove AppData folder on uninstall (keep .key file to preserve encrypted data)
Type: filesandordirs; Name: "{userappdata}\EyeProject\storage"
Type: filesandordirs; Name: "{userappdata}\EyeProject\results"
