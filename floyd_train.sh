#!/bin/bash -
floyd run --data paulemmanuel/datasets/nyc_taxi_data_2016/1 --env tensorflow-1.2 --gpu "python train.py --floyd-job"
