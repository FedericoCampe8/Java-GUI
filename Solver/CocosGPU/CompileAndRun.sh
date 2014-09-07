#!/bin/bash
cd src_jnet;
make clean;
make;
cp jnet ../bin_jnet;
cd ..;
make clean;
make;
