#!/bin/bash

for oldfile in *.cpp
do
    mv "$oldfile" "`basename $oldfile .cpp`.cu";
    echo "File $oldfile processed"
done
