#!/bin/bash

for oldfile in *.cu 
do
    mv "$oldfile" "`basename $oldfile .cu`.cpp";
    echo "File $oldfile processed"
done
