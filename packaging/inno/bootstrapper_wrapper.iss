; Inno Setup wrapper that preserves the existing Qt bootstrapper UI.
; It embeds the PyInstaller-built bootstrapper, extracts it to %TEMP%, and launches it immediately.
; Result: user-facing install experience stays the current custom UI, while distribution can use
; a signed Inno-produced Setup.exe wrapper.

#ifndef AppName
  #define AppName "MediaTranscribeStudio"
#endif

#ifndef AppVersion
  #define AppVersion "11.45.14"
#endif

#ifndef AppPublisher
  #define AppPublisher "MediaTranscribeStudio"
#endif

#ifndef BootstrapperPath
  #error "BootstrapperPath define is required (path to the Qt bootstrapper exe)."
#endif

#ifndef InnerBootstrapperName
  #define InnerBootstrapperName "MediaTranscribeStudio-Setup-Inner.exe"
#endif

#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif

#ifndef OutputBaseFilename
  #define OutputBaseFilename "MediaTranscribeStudio-Setup"
#endif

#ifndef WizardSmallImageFile
  #define WizardSmallImageFile ""
#endif

#ifndef SetupIconFile
  #define SetupIconFile ""
#endif

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={tmp}\{#AppName}
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBaseFilename}
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline
UsePreviousPrivileges=no
DisableWelcomePage=yes
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableReadyPage=yes
DisableFinishedPage=yes
DisableReadyMemo=yes
ShowLanguageDialog=no
AllowNoIcons=yes
Uninstallable=no
CreateUninstallRegKey=no
SetupLogging=no
UsedUserAreasWarning=no
WizardStyle=modern
VersionInfoProductName={#AppName} Setup
VersionInfoProductVersion={#AppVersion}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} Bootstrap Wrapper
VersionInfoOriginalFileName={#OutputBaseFilename}.exe
#if SetupIconFile != ""
SetupIconFile={#SetupIconFile}
#endif
#if WizardSmallImageFile != ""
WizardSmallImageFile={#WizardSmallImageFile}
#endif

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "{#BootstrapperPath}"; DestName: "{#InnerBootstrapperName}"; Flags: dontcopy ignoreversion

[Code]
var
  LaunchAttempted: Boolean;

function _InnerArgNeedsValue(const ArgNameLower: string): Boolean;
begin
  Result :=
    (ArgNameLower = '--url') or
    (ArgNameLower = '--sha256') or
    (ArgNameLower = '--dir') or
    (ArgNameLower = '--scope');
end;

function _IsKnownInnerArg(const ArgNameLower: string): Boolean;
begin
  Result :=
    _InnerArgNeedsValue(ArgNameLower) or
    (ArgNameLower = '--auto-start') or
    (ArgNameLower = '--show-url') or
    (ArgNameLower = '--elevated') or
    (ArgNameLower = '--no-desktop-shortcut') or
    (ArgNameLower = '--no-start-menu-shortcut') or
    (ArgNameLower = '--no-auto-launch');
end;

procedure _AppendCmdArg(var Dest: string; const Arg: string);
var
  NeedsQuotes: Boolean;
begin
  if Dest <> '' then
    Dest := Dest + ' ';

  NeedsQuotes :=
    (Pos(' ', Arg) > 0) or
    (Pos(#9, Arg) > 0) or
    (Pos('"', Arg) > 0);

  if NeedsQuotes then
    Dest := Dest + AddQuotes(Arg)
  else
    Dest := Dest + Arg;
end;

function _CollectInnerBootstrapperParams(): string;
var
  I: Integer;
  Token: string;
  TokenLower: string;
begin
  Result := '';
  I := 1;
  while I <= ParamCount do begin
    Token := ParamStr(I);
    TokenLower := Lowercase(Token);

    { Only forward arguments that belong to the inner Qt bootstrapper.
      Ignore Inno's own switches like /SL5, /SPAWNWND, /NOTIFYWND, /SP-, etc. }
    if _IsKnownInnerArg(TokenLower) then begin
      _AppendCmdArg(Result, Token);
      if _InnerArgNeedsValue(TokenLower) and (I < ParamCount) then begin
        Inc(I);
        _AppendCmdArg(Result, ParamStr(I));
      end;
    end;

    Inc(I);
  end;
end;

function _LaunchInnerBootstrapper(): Boolean;
var
  InnerPath: string;
  Params: string;
  ResultCode: Integer;
begin
  Result := False;

  ExtractTemporaryFile('{#InnerBootstrapperName}');
  InnerPath := ExpandConstant('{tmp}\{#InnerBootstrapperName}');
  if not FileExists(InnerPath) then begin
    MsgBox(
      '无法启动内置安装器（文件未提取成功）。' + #13#10 + InnerPath,
      mbCriticalError,
      MB_OK
    );
    Exit;
  end;

  Params := _CollectInnerBootstrapperParams();
  if not Exec(InnerPath, Params, ExtractFileDir(InnerPath), SW_SHOWNORMAL, ewNoWait, ResultCode) then begin
    MsgBox(
      '启动安装器失败。请尝试以管理员身份运行，或检查杀毒软件拦截。' + #13#10#13#10 +
      '错误代码: ' + IntToStr(ResultCode) + #13#10 +
      '文件: ' + InnerPath,
      mbCriticalError,
      MB_OK
    );
    Exit;
  end;

  Result := True;
end;

procedure InitializeWizard();
begin
  if LaunchAttempted then
    Exit;
  LaunchAttempted := True;

  WizardForm.Visible := False;
  if _LaunchInnerBootstrapper() then begin
    WizardForm.Close;
  end else begin
    WizardForm.Visible := True;
    WizardForm.Caption := '{#AppName} Setup Wrapper';
  end;
end;
