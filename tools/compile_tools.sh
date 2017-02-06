#!/bin/bash

# 1. Get and compile SPTK

#echo "downloading SPTK-3.9..."
#wget http://downloads.sourceforge.net/sp-tk/SPTK-3.9.tar.gz
#tar xzf SPTK-3.9.tar.gz

#echo "compiling SPTK..."
#(
#    cd SPTK-3.9;
#    ./configure --prefix=$PWD/build;
#    make;
#    make install
#)

# 2. Getting WORLD

echo "compiling WORLD..."
(
    cd WORLD;
    make
    make analysis synth
    make clean
)

# 3. Getting GlottHMM

echo "Compiling GlottHMM..."
(
    git clone https://github.com/mjansche/GlottHMM.git
    cd GlottHMM/src
    make
)
# 3. Copy binaries

mkdir -p bin/{SPTK-3.9,WORLD,GlottHMM}

cp SPTK-3.9/build/bin/* bin/SPTK-3.9/
cp WORLD/build/analysis bin/WORLD/
cp WORLD/build/synth bin/WORLD/
cp GlottHMM/src/Analysis bin/GlottHMM
cp GlottHMM/src/Synthesis bin/GlottHMM
cp GlottHMM/cfg/config_default bin/GlottHMM

echo "All tools successfully compiled!!"
