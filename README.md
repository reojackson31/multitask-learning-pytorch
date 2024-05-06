# Multi-task Learning for Predicting House Prices and House Category using PyTorch Lightning

## 1. Introduction
In this project, the goal is to develop a multi-task learning model capable of predicting both house prices and house categories using a single neural network architecture. The dataset utilized for this project is the "House Prices - Advanced Regression Techniques Dataset" from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The model will be built using PyTorch Lightning to simplify the code for the neural network by removing much of the boilerplate PyTorch code that is typically needed. This allows us to focus more on the core model architecture and experiment with different configurations and optimizations more effectively.

## 2. Data Description
The dataset from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), includes a variety of details about houses that are on sale. Some of the key features in the dataset include 'MSSubClass' which categorizes the dwelling type involved in the sale, 'MSZoning' indicating zoning classification, and detailed characteristics of the house structure such as 'BldgType' and 'HouseStyle'. The dataset also captures information on the property’s size, shape, and topography with attributes like 'LotFrontage', 'LotArea', and 'LandContour'. Additionally, it includes various qualitative assessments of the property's features, condition, and utilities, which will be useful in the regression analysis of house prices. The target variable for regression is the Sale Price of the house, and for the classification model -  a new target variable is created by combining the 'House Style', 'Bldg Type', 'Year Built', and 'Year Remod/Add' variables.

Refer to the Python Notebook for insights from exploratory data analysis done on the dataset.

## 3. Data Preprocessing
For Classification Task, we need to create a new variable 'House Category' from 'House Style', 'Bldg Type', 'Year Built', and 'Year Remod/Add'. First, I have created a column to capture if the construction or latest remodeling of the property happened in the last 30 years or not, and they are classified into 'Old' and 'New' properties. Next, the houses are categorized by whether they are single story or multi-story houses by using the HouseStyle field. Then, by using the BuildingType field, houses are categorized into single-family homes or multi-unit properties, including duplexes and condos. Finally, all the 3 categorical columns created are combined to create a single House Category, which will be used as the target variable for the classification task, which would be a multi-class classification problem since we have 8 classes of house categories created in this approach.

## 4. Neural Network Model Architecture
The baseline architecture for the multi-task learning model is designed to simultaneously predict house prices (a regression task) and classify house categories (a classification task). The model is implemented using PyTorch Lightning. The baseline architecture for the model is explained in this section.

### 4.1 Shared Layers
First, I have created shared layers which are used to extract features that are common and beneficial for both tasks. This is implemented using a sequential module containing two fully connected (dense) layers each followed by a ReLU activation function. The first layer maps the input features to a hidden space, and the second layer enhances the feature extraction process before branching out to task-specific outputs.

### 4.2 Task-specific Layers
First, I have created a regression layer, which consists of a single linear layer that predicts the house price. The output dimension for this head is set to 1, corresponding to the price prediction. Next, the classification layer is created, which also uses a single linear layer but is designed to classify the house into one of several categories. The output dimension for this head is set as 8, which corresponds to the number of distinct house categories.

### 4.3 Experiments with Activation Functions & Optimizers
In order to explore the effects of different activation functions and optimizers on the performance of the multi-task learning model, a series of experiments was conducted. The activation functions and optimizers were selected based on their different characteristics and expected impacts on model convergence and accuracy. Loss function used for the regression task is Mean Squared Error (MSE) Loss, and Cross Entropy Loss for the classification task. The two loss functions are added to create the total loss which is minimized by the optimizer. Combinations of activation functions and optimizers experimented, and their results are given in the table below.

| **Activation Function** | **Optimizer** | **Validation Loss** | **Validation MSE** | **Validation Accuracy** |
|-------------------------|---------------|---------------------|--------------------|------------------------|
| Tanh                    | SGD           | 0.89                | 0.22               | 0.85                   |
| LeakyReLU               | Adam          | 1.05                | 0.27               | 0.83                   |
| LeakyReLU               | SGD           | 1.08                | 0.33               | 0.86                   |
| ReLU                    | SGD           | 1.11                | 0.22               | 0.84                   |
| Tanh                    | Adam          | 1.11                | 0.29               | 0.85                   |
| Tanh                    | RMSprop       | 1.36                | 0.31               | 0.82                   |
| ReLU                    | Adam          | 1.53                | 0.29               | 0.83                   |
| ReLU                    | RMSprop       | 1.68                | 0.24               | 0.78                   |
| LeakyReLU               | RMSprop       | 1.72                | 0.35               | 0.79                   |


From the table, we can see a significant variation in model performance across different combinations of activation functions and optimizers. The Tanh activation function with SGD optimizer gave the best validation loss, along with good validation mse and accuracy, suggesting that it might be particularly effective for this dataset. However, we will use a more systematic hyperparameter optimization approach using Optuna to find the best combination in the upcoming section.


## 5. Creating a Custom Loss Function
For the initial experiments, the loss function considered was the sum of loss functions for the regression (MSE) and classification (Cross Entropy Loss) tasks, with the assumption that both losses have to be minimized with equal importance. For this section, I have created a custom trainer module, which takes the weights for the regression and classification loss. So using this module, we can priortize the losses for the regression or classification task as per the requirement to create an optimized model. For example, I have demonstrated this by giving a higher weight for the regression loss, assuming that the requirement was that predicting house prices accurately is more important than predicting the house categories. (refer to the notebook for details on this implementation)

## 6. Using Advanced PyTorch Lightning Features

### 6.1 Checkpointing
The ModelCheckpoint callback is used to automatically save the top 3 performing models based on the validation loss, which is the criterion for monitoring performance improvements. The models are saved in the specified directory with a filename that includes the epoch number and the validation loss, making it easy to identify and retrieve specific model states. By setting the mode to 'min', the callback focuses on minimizing the validation loss, ensuring that only the best models in terms of lower loss are saved.

### 6.2 Early Stopping
The EarlyStopping callback is configured to monitor the validation loss and will stop the training process if there is no improvement in the validation loss for 20 consecutive epochs. This patience parameter is added to prevent premature stopping that might occur due to minor fluctuations in training dynamics. The strict mode ensures that the training stops only if the conditions specified are met. By focusing on minimizing the validation loss (mode='min'), this feature significantly helps to avoid overfitting and unnecessary computation by terminating the training at an optimal point. For example, in one model experiment tried in this section, the trainer which was configured to run for 100 epochs, stopped after 41 epochs with a validation loss of 0.681, which did not improve in the previous 20 epochs.

## 7. Hyperparameter Tuning
The final step in this project is hyperparameter tuning using PyTorch Lightning's integration with Optuna. This step includes all the considerations and features from the previous steps, combined together to find the best performing set of hyperparameters for the model. I am using Optuna to systematically search for the best model configuration. This section outlines how I have integrated Optuna into the training pipeline, experiment with various hyperparameters, and track the improvements in model performance.

### 8. Results and Best Hyperparameters
After conducting the specified number of trials, Optuna provides a summary of the best performing trial, including the lowest validation loss achieved and the corresponding set of hyperparameters. The best configuration found by Optuna is the following:
- Learning Rate (lr): 0.0656
- Optimizer Type: SGD
- Activation Function: Tanh
- Batch Size: 128

The evaluation metrics for the model with the best configuration of hyperparameters is given in the table below:

| **Metric**                | **Value** |
|---------------------------|-----------|
| Train Loss                | 0.368     |
| Validation Loss           | 0.703     |
| Validation MSE            | 0.256     |
| Validation R2 Score       | 0.775     |
| Validation MAE            | 0.276     |
| Validation Accuracy       | 0.84      |
| Validation ROC-AUC Score | 0.979     |
| Validation F1 Score       | 0.84      |

This systematic approach allows for a comprehensive exploration of the parameter space, which is better than the manual experimentation done earlier, which is also evidenced by the better model performance of the best model configuration found by Optuna.


