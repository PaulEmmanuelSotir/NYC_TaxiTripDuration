# NYC Taxi Trip Duration - Kaggle competition

This is a Kaggle competition submission using a fully connected neural network implemented on tensorflow and whose hyperparameters are optimized using hyperopt.  
Fully connected neural network is trained using warm resart cyclic learning rate with snaphsot ensembling (see papers [SGDR: SGD with warm resart](https://arxiv.org/pdf/1608.03983.pdf) and [Snapshot ensembles: train 1, get M for free](https://openreview.net/pdf?id=BJYwwY9ll)).  
Neural network is composed of 10 dense layers with batch normalization and optimized using momentum SGD (ADAM performs better if learning rate is constant but doesn't plays well with learning rate warm resart cycles).  
Output layer is a softmax layer trained for classification on buckets using cross entropy loss (Discretizing target instead of doing a regression improves preformances).

## Run instructions

Assuming you installed Python 3 with *tensorflow*, *scikit-learn*, *numpy* and eventually *hyperopt*, you need to:  

- Clone the project

```bash
git clone https://github.com/PaulEmmanuelSotir/NYC_TaxiTripDuration.git
cd ./NYC_TaxiTripDuration
```

- Unzip dataset files located in 'NYC_taxi_data' directory

```bash
unzip ./NYC_taxi_data_2016/NYC_taxi_data_2016.zip -d ./NYC_taxi_data_2016/
# If you want to use OSRM features you should unzip OSRM data too:
unzip ./NYC_taxi_data_2016/osrm/fastest_routes_test.zip -d ./NYC_taxi_data_2016/osrm
unzip ./NYC_taxi_data_2016/osrm/fastest_routes_train_part_1.zip -d ./NYC_taxi_data_2016/osrm
unzip ./NYC_taxi_data_2016/osrm/fastest_routes_train_part_2.zip -d ./NYC_taxi_data_2016/osrm
```

- And train the model

```bash
python nyc_dnn.py
```

You can trigger hyperparameter optimization using the following command:

```bash
# This command will print the best hyperparameter set found. Then, you can edit nyc_dnn.py to use these hyperparameters.
python hyperparameter_opt.py
```

Also note that this project can run on [Floyd](https://www.floydhub.com/) (Heroku for deep learning):

```bash
# To run a Floyd training job, use the following command:
floyd run --data paulemmanuel/datasets/nyc_taxi_data_2016/6 --env tensorflow-1.3 --tensorboard --gpu "python nyc_dnn.py --floyd-job"
# Or run the following command for hyperparameter optimization:
floyd run --data paulemmanuel/datasets/nyc_taxi_data_2016/6 --env tensorflow-1.3 --tensorboard --gpu "python hyperparameter_opt.py --floyd-job"
```
