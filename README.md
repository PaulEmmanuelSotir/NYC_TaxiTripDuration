# NYC Taxi Trip Duration - Kaggle competition
Kaggle competition submission using a simple fully connected neural network implemented on tensorflow and whose hyperparamters are optimized using hyperopt.

## Run instructions
Assuming you installed Python 3 with *tensorflow*, *scikit-learn*, *numpy* and eventually *hyperopt*, you need to:  

- Clone the project
```bash
git clone https://github.com/PaulEmmanuelSotir/NYC_TaxiTripDuration.git
cd ./NYC_TaxiTripDuration
```

- Unzip dataset files located in 'NYC_taxi_data' directory
```bash
unzip ./NYC_taxi_data_2016/train.zip -d ./NYC_taxi_data_2016/
unzip ./NYC_taxi_data_2016/test.zip -d ./NYC_taxi_data_2016/
```

- And train the model
```bash
python nyc_dnn.py
```   

You can trigger hyperparameter optimization using the follwing command:
```bash
# This command will print the best hyperparameter set found. Then, you can edit nyc_dnn.py to use these hyperparameters.
python hyperparameter_opt.py
```

Also note that this project can be run on [Floyd](https://www.floydhub.com/) (Heroku for deep learning):
```bash
# To run a Floyd training job, use the follwing command:
floyd run --data paulemmanuel/datasets/nyc_taxi_data_2016/1 --env tensorflow-1.2 --gpu "python train.py --floyd-job"
# Or run the follwing command for hyperparameter optimization:
floyd run --data paulemmanuel/datasets/nyc_taxi_data_2016/1 --env tensorflow-1.2 --gpu "python hyperparameter_opt.py --floyd-job"
```
