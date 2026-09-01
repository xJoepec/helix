@echo off
setlocal
pushd "%~dp0" || exit /b 1
where uv >nul 2>nul
if %errorlevel% equ 0 (
    uv run helix %*
) else (
    python "%~dp0helix" %*
)
set "helix_exit=%errorlevel%"
popd
exit /b %helix_exit%
