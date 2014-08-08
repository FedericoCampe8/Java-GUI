#!/bin/bash
echo ------------------ COCOS ------------------
cd ./Cocos;
make clean;
./CompileAndRun.sh
cd ..
echo ------------------ FIASCO ------------------
cd ./Fiasco;
make clean;
make;
cd ..
