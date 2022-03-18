# meal-machine-learning
Calculation of meal quality predictor using labeled data and a series of supervised learning algorithms.

**Contents:** <br />
  LICENSE - MIT License <br />
  Requirements - Required modules <br />
  meal_ml_git/ml_nn - Neural network method <br />
  meal_ml_git/ml_decision_tree - Decision tree method <br />
  meal_ml_git/ml_knn - K-nearest neighbor method <br />
  meal_ml_git/input_file - Input data <br />
 
**How to use:**
  Neural network - In the meal_ml_git/ml_nn directory, run
  ```
  $ python meal_data.py
  ```
  to start training the neural network. It should take a few minutes.
  
  Decision tree - In the meal_ml_git/ml_decision_tree directory, run
  ```
  $ python decision_tree.py
  ```
  to make the decision tree and plot it.
  
  K-nearest neighbor - In the meal_ml_git/ml_knn directory, run
  ```
  $ python knn.py
  ```
  to start the k-nearest neighbor calculation. Since many plots will be generated,
  the program will then ask whether you wish to plot the data.
