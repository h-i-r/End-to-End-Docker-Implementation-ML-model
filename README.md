# Predict Customer Purchasing Power 

## Description
The project provides a ranking algorithm to predict the probability of purchase.
In other words, when a potential customer comes in, 
we predict the chance of selling the product to the new customer. 
We rank the new customers according to the predicted probability and 
first assign to consultants the customers with a higher probability 
of purchasing the product.

## Prerequisites
- `Python 3.11.7`

## Installation

1. Install required packages
    ```bash
    pip install -r requirements.txt
   
## Usage
1. Change the directory to /src
    ```bash
    cd src

2. Run the following command to start the program and rank the potential customers 
based on predictions:
    ```bash
    python main.py 

By default, the following configurations are set:

`model_type: logistic_regression` # Model to be used for predictions<br />
`lasso_penalty: 10` # Regularization strength. It can accept a list of numbers<br />
`file_name: Customers_Dataset` # Name of dataset file. 
The file should be placed in /src/datasets<br />
`n_estimators: 50` # Number of trees in random forest. It can accept a list of numbers. 
This value will be ignored when logistic regression is used. <br />
`show_plots: False` # If plots should be displayed while running the program. <br />
`number_of_potential_customers: 20` # Number of samples against which rankings 
will be generated<br />
`sparse_column_threshold: 60` # Removes features which are empty based on this threshold<br />

To run the program using random forest model, the following command can be
used:<br />

   `python main.py --model_type random_forest`

## Dataset
The program uses the dataset from the /src/datasets directory. The default dataset 
filename is "Customers_Dataset.csv" which
contains about 85k customers. 

If a new dataset needs to be used, the csv file 
should be placed in the mentioned directory and the filename should be passed as an 
argument while running the program e.g. If the filename is 
new_customers_dataset.csv then the following command should be used<br />

`python main.py --file_name new_customers_dataset`

## Plots
The plots are automatically saved in /src/plots directory when the 
program runs.

## Models
The models are saved in the /src/models directory with .pkl extension.

## Results
The customers are ranked and displayed as an output in the terminal 
when the program runs and also gets saved in a csv format file in 
/src/results directory.

## Docker
The docker image can be built using the Dockerfile to run the program. 
Run the following command to build the image:

`docker build -t purchase-predictor .`   

The image will be built with the name "purchase-predictor". Afterwards, 
the following command can be used to run the docker container.

`docker run purchase-predictor`

To use different configuration while running the docker container, the 
arguments can be passed as shown below:

`docker run purchase-predictor --sparse_column_threshold 70 --number_of_potential_customers 20`

## Exploratory Data Analysis
The Jupyter Notebook with the filename "exploratory_data_analysis.ipynb" contains the
initial data exploration. It is placed in the /src directory.

   





