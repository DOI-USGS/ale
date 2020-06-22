mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=%LIBRARY_PREFIX% -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON -DALE_BUILD_TESTS=OFF -G "Visual Studio 15 2017 Win64" ..
cmake --build . --target ALL_BUILD --config Release

if errorlevel 1 exit 1
