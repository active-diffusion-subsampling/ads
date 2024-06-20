#!/bin/bash

mkdir train
mkdir test

for i in {0..9}
do
    cp training/$i/*.png train/
    cp testing/$i/*.png test/
done