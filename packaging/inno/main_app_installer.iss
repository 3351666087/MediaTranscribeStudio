; Inno Setup main installer (core logic).
; This installer owns file deployment/uninstall. The PySide6 frontend is optional
; and can launch this installer silently with /DIR + /ALLUSERS or /CURRENTUSER.

#ifndef AppName
  #define AppName "MediaTranscribeStudio"
#endif

#ifndef AppVersion
  #define AppVersion "11.45.14"
#endif

#ifndef AppPublisher
  #define AppPublisher "MediaTranscribeStudio"
#endif

#ifndef AppSourceDir
  #error "AppSourceDir define is required (path to built app directory)."
#endif

#ifndef MainExeName
  #define MainExeName AppName + ".exe"
#endif

#ifndef MainAppUserModelID
  #define MainAppUserModelID "MediaTranscribeStudio.Main"
#endif

#ifndef UninstallFrontendExeName
  #define UninstallFrontendExeName "uninstall.exe"
#endif

#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif

#ifndef OutputBaseFilename
  #define OutputBaseFilename "MediaTranscribeStudio-Setup-Core"
#endif

#ifndef WizardSmallImageFile
  #define WizardSmallImageFile ""
#endif

#ifndef WizardImageFile
  #define WizardImageFile ""
#endif

#ifndef SetupIconFile
  #define SetupIconFile ""
#endif

[Setup]
AppId={{4E51595C-59C0-4C9B-99F4-9C1F674F916B}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBaseFilename}
Compression=lzma2/ultra64
SolidCompression=yes
DiskSpanning=yes
DiskSliceSize=2000000000
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline
UsePreviousAppDir=yes
UsePreviousTasks=yes
UsePreviousPrivileges=yes
CloseApplications=no
RestartApplications=no
DisableProgramGroupPage=yes
WizardStyle=modern
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} Installer Core
VersionInfoOriginalFileName={#OutputBaseFilename}.exe
UninstallDisplayIcon={app}\{#MainExeName}
ChangesAssociations=no
#if SetupIconFile != ""
SetupIconFile={#SetupIconFile}
#endif
#if WizardImageFile != ""
WizardImageFile={#WizardImageFile}
#endif
#if WizardSmallImageFile != ""
WizardSmallImageFile={#WizardSmallImageFile}
#endif

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "startmenuicon"; Description: "Create Start Menu shortcuts"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce
Name: "autolaunch"; Description: "Launch {#AppName} after install"; GroupDescription: "Post-install"; Flags: unchecked

[Files]
Source: "{#AppSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#MainExeName}"; Tasks: desktopicon; WorkingDir: "{app}"; IconFilename: "{app}\{#MainExeName}"; AppUserModelID: "{#MainAppUserModelID}"
Name: "{autoprograms}\{#AppName}\{#AppName}"; Filename: "{app}\{#MainExeName}"; Tasks: startmenuicon; WorkingDir: "{app}"; IconFilename: "{app}\{#MainExeName}"; AppUserModelID: "{#MainAppUserModelID}"
Name: "{autoprograms}\{#AppName}\Uninstall {#AppName}"; Filename: "{app}\{#UninstallFrontendExeName}"; Tasks: startmenuicon; WorkingDir: "{app}"; Check: HasCustomUninstallFrontend
Name: "{autoprograms}\{#AppName}\Uninstall {#AppName}"; Filename: "{uninstallexe}"; Tasks: startmenuicon; Check: not HasCustomUninstallFrontend

[Run]
Filename: "{app}\{#MainExeName}"; Tasks: autolaunch; WorkingDir: "{app}"; Flags: nowait

[Code]
function HasCustomUninstallFrontend(): Boolean;
begin
  Result := FileExists(ExpandConstant('{app}\{#UninstallFrontendExeName}'));
end;
