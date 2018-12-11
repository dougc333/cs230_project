#!/bin/bash

echo "anotherrunv1aonly2features runs for 40 epochs can see test error increase at epoch 40"
echo "10 epochs good enough for  anotherrunv1aonly2features"
echo "--------"

echo "hierattn Train:76.93  Dev:77.52 Test:77.61 "

echo "running 3 layer linear Train:62.75 Dev:62.53 Test:62.69" 
python testv1a-quora-3layerlinear.py > linearT7600
echo "running nonlinear Train:88.99 Dev:79.39 Test:78.73"
python testv1a-quora-3layernonlinear.py > nonlinearT7600
echo "3layernonlinear dropout 0.2"
python testv1a-quora-3layernonlinear-Dropout2.py > nonlinear_dp2_T7600
echo "3layernonlinear dropout 0.5"
python testv1a-quora-3layernonlinear-Dropout5.py > nonlinear_dp5_T7600
echo "tanh Train:85.54 Dev:78.7 Test:78.13"
python testv1a-quora-3layernonlineartanh.py > nonlineartanhT7600
echo "4096 Train:90.16 Dev:79.18 Test:78.92 "
python testv1a-quora-3layernonlinear4096.py > 4096T7600
echo "dropout122 Train:83.31 Dev:78.65 Test:78.77"
python testv1a-quora-3layernonlinearDropout122.py > Dropout122T7600
#echo "droupout125"
#python testv1a-quora-3layernonlinearDropout125.py > Dropout125
#echo "dropout 155"
#python testv1a-quora-3layernonlinearDropout155.py > Dropout155
echo "adam Train:97.0 Dev: 80.27 Test:80.19"
python testv1a-quora-3layernonlinearAdam.py > AdamT7600
