@echo off
REM Runs the three probed simulations using the bundled ngspice_con.
REM Double-click this from project 2\SpiceFiles\breakdown\ to regenerate *_results.txt.
setlocal
set NGSPICE=..\..\ngspice-45.2_64\Spice64\bin\ngspice_con.exe
cd /d "%~dp0"
for %%F in (probed_2000ps.spi probed_320ps.spi probed_298ps.spi) do (
    echo === Running %%F ===
    "%NGSPICE%" -b %%F
    if errorlevel 1 (
        echo ngspice failed on %%F
        exit /b 1
    )
)
echo Done.
endlocal
