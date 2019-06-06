#!/bin/bash

for f in $1/*.JPG;
do
#    echo ${f////_}
    mv $f $1/${f////_}
done
