mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$PREFIX -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release ..
make install
