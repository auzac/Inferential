; Inno Setup 6 script: per-user installer for the one-folder PyInstaller build.
;
; Built by packaging\windows_build.ps1 (which passes /DAppVersion and /DSourceDir).
; Manual build:  iscc /DAppVersion=0.5.0 /DSourceDir=..\output\dist\InferentialModelStudio InferentialModelStudio.iss
;
; Per-user install (no administrator rights): files go to %LOCALAPPDATA%\Programs, the
; uninstaller is registered under HKCU, and .imsproj files open with the application.
; NOTE: written and reviewed on Linux; build and test it on Windows before release.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\output\dist\InferentialModelStudio"
#endif
#define AppName "Inferential Model Studio"
#define AppExe "InferentialModelStudio.exe"
#define AppId "{{6E7B1C1A-4E52-4F3B-9B7D-2C6A1E5A9D41}"

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Process Engineering (internal)
DefaultDirName={localappdata}\Programs\InferentialModelStudio
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\output
OutputBaseFilename=InferentialModelStudio-{#AppVersion}-setup-win64
SetupIconFile=..\..\assets\icons\app.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName} {#AppVersion}
LicenseFile=..\..\LICENSES.md
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ChangesAssociations=yes
CloseApplications=yes
VersionInfoVersion={#AppVersion}
VersionInfoProductName={#AppName}

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked
Name: "associate"; Description: "Open .imsproj project files with {#AppName}"; GroupDescription: "File types:"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autoprograms}\{#AppName} example project"; Filename: "{app}\{#AppExe}"; Parameters: """{app}\_internal\examples\synthetic_distillation\sample_project.imsproj"""
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.imsproj"; ValueType: string; ValueName: ""; ValueData: "InferentialModelStudio.Project"; Flags: uninsdeletevalue; Tasks: associate
Root: HKA; Subkey: "Software\Classes\InferentialModelStudio.Project"; ValueType: string; ValueName: ""; ValueData: "Inferential Model Studio project"; Flags: uninsdeletekey; Tasks: associate
Root: HKA; Subkey: "Software\Classes\InferentialModelStudio.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#AppExe},0"; Tasks: associate
Root: HKA; Subkey: "Software\Classes\InferentialModelStudio.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: associate

[Run]
Filename: "{app}\{#AppExe}"; Description: "Start {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Settings, logs, recovery copies and extracted portable data live in %LOCALAPPDATA%\InferentialModelStudio
; and are kept on uninstall (user data). Only the cache of extracted portable-project data is removed.
Type: filesandordirs; Name: "{localappdata}\InferentialModelStudio\cache"
