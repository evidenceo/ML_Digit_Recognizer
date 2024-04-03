# ML_Digit_Recognizer
**Group 6 Machine Learning Digit Recognizer**

**Brief Description:**
This machine learing digit recognizer loads and preprocesses data  from CSV files, splitting it into training and test sets, reshaping, and normalizing the images. It then trains and evaluates two models: a Support Vector Machine (SVM) classifier and a Convolutional Neural Network (CNN). The SVM model is trained on flattened image data and evaluated using a confusion matrix and accuracy score. The CNN model, defined with convolutional and dense layers, is trained and evaluated similarly, with additional visualization of loss and accuracy curves during training. Finally, both models are compared in terms of accuracy and performance on validation data, providing insights into their effectiveness in classifying handwritten digits.

**Project Functionalities**
Primary functionalities: digit recognition using Support Vector Machine (SVM) and Convolutional Neural Network (CNN) models. 
Here's a breakdown of its functionalities:

1. **Data Loading and Preprocessing**:
   - The code loads digit images from CSV files and preprocesses them by splitting them into training and test sets, reshaping them into suitable formats, and normalizing pixel values to a range between 0 and 1.

2. **Support Vector Machine (SVM) Model**:
   - It trains an SVM classifier using the flattened image data.
   - Evaluates the SVM model's performance using a confusion matrix and accuracy score.
   - Predictions are made on the test data using the trained SVM model, and some test images with their predicted labels are displayed.

3. **Convolutional Neural Network (CNN) Model**:
   - Defines and trains a CNN model using Keras with TensorFlow backend.
   - The CNN architecture consists of convolutional layers, max-pooling layers, dropout layers, and dense layers.
   - Evaluates the CNN model's performance using a confusion matrix and accuracy score.
   - Predictions are made on the validation data using the trained CNN model, and some validation images with their predicted labels are displayed.
   - Loss and accuracy curves are plotted for the CNN model during training.

4. **Comparison and Evaluation**:
   - Both SVM and CNN models are compared in terms of accuracy and performance on validation data.
   - Confusion matrices and accuracy scores are displayed for both models, offering insights into their effectiveness in classifying handwritten digits.

**Setup Instructions**:
   - Ensure Python and necessary libraries like TensorFlow, Keras, pandas, scikit-learn, NumPy, and Matplotlib are installed.
   - Download the provided CSV files from the MNIST dataset (train.csv and test.csv) for training and test data.
   - Load the notebook in a suitable environment such as Google Colab
   - Execute the code cells sequentially to load data, train models, and evaluate their performance.
   - Adjust hyperparameters, model architectures, or data preprocessing steps as needed for further experimentation.

**Conclusion**

In conclusion, this project demonstrates the implementation and comparison of Support Vector Machine (SVM) and Convolutional Neural Network (CNN) models for digit recognition. Through thorough preprocessing, training, and evaluation steps, it showcases the effectiveness of both approaches in accurately classifying handwritten digits. The SVM model, achieves decent performance, while the CNN model, outperforms SVM significantly. Moreover, the CNN model's ability to learn hierarchical features directly from pixel values makes it particularly suitable for image classification tasks like digit recognition.
