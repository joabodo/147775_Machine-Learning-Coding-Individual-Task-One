# 147775_Machine-Learning-Coding-Individual-Task-One

### **Summary: Linear Regression using Gradient Descent in Python**

In this task, we implemented a **Linear Regression model** from scratch using **Python** to predict office prices based on their size. The objective was to find the best-fitting line between **office size (X)** and **office price (y)** by minimizing the **Mean Squared Error (MSE)** using **Gradient Descent**. 

The steps followed were:  
1. **Data Preparation**: 
   - We extracted two columns from the dataset: `SIZE` (feature) and `PRICE` (target).

2. **Model Initialization**: 
   - The slope \(m\) (weight) and intercept \(c\) (bias) were initialized randomly to start the learning process. 

3. **Gradient Descent**:
   - In each **epoch**, the model made predictions using the current values of \(m\) and \(c\).  
   - The gradients (partial derivatives) were computed to measure the direction and magnitude of change required to minimize the error.  
   - \(m\) and \(c\) were updated iteratively using these gradients scaled by a **learning rate** of 0.0001.

4. **Training the Model**:
   - The model trained for **10 epochs**, with the **MSE decreasing** over time, indicating improved accuracy.

5. **Line of Best Fit**:
   - After training, we plotted the **line of best fit** against the original data points to visualize the model’s performance.

6. **Prediction**:
   - Finally, the model predicted the price for a **100 sq. ft** office to be **135.37 units**.

This implementation showcases how a simple **linear regression model** can be built and trained using gradient descent. By continuously adjusting the slope and intercept to reduce the MSE, the model learns a linear relationship between the feature and target. This approach demonstrates the power of gradient descent for optimizing machine learning models.

