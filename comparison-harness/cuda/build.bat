@echo off
REM raw_launch.cu is PURE host code (CUDA driver API only -- no __global__, no <<<>>>),
REM so we compile it directly with MSVC cl.exe as C++ (/Tp) and link the driver-API
REM import lib cuda.lib. This sidesteps nvcc's Windows host-compiler integration
REM (which mis-forwards -o to cl and aborts with "single input for non-link phase").
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
set "CUDA_INC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\include"
set "CUDA_LIBDIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\lib\x64"
cl /nologo /O2 /EHsc /Tp raw_launch.cu /Fe:raw_launch.exe ^
   /I"%CUDA_INC%" ^
   /link /LIBPATH:"%CUDA_LIBDIR%" cuda.lib
echo BUILD_BAT_RC=%ERRORLEVEL%
