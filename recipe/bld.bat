mkdir build
cd build

cmake   -GNinja ^
        -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON ^
        -DALE_BUILD_TESTS=OFF ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DCMAKE_INSTALL_PREFIX="%PREFIX%" ^
        -DCMAKE_INSTALL_LIBDIR="%LIBRARY_LIB%" ^
        -DCMAKE_INSTALL_INCLUDEDIR="%LIBRARY_INC%" ^
        -DCMAKE_INSTALL_BINDIR="%LIBRARY_BIN%" ^
        -DCMAKE_INSTALL_DATADIR="%LIBRARY_PREFIX%" ^
        ..

cmake --build . --target install --config Release

cd ..
pip install . --no-deps -vv

if errorlevel 1 exit 1
