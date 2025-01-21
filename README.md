# Covid-19 dectection on Chest X-ray images



This project uses a Convolutional Neural Network (CNN) to classify COVID-19 images into different categories. The implementation is done using TensorFlow and Keras, with data augmentation and preprocessing to improve model performance. The project is divided into steps for data loading, visualization, preprocessing, model building, training, and prediction.

## Project Structure

- **train/**: Contains training images organized into subfolders for each class.
- **test/**: Contains testing images organized into subfolders for each class.

## Requirements

To run this project, ensure you have the following installed:

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- pandas


## Data Preprocessing

1. **Dataset Loading**:
   - Find dataset and model in given drive link : https://drive.google.com/drive/folders/14J6DrvU3Wz_EiTC3QJQX2Gb5qufAabLP?usp=sharing
   
2. **Data Augmentation**:
   - Random zoom, rotation, and horizontal flipping are applied to augment the dataset and improve model generalization.

3. **Normalization**:
   - All image pixel values are scaled to the range [0, 1] using a `Rescaling` layer.

## Model Architecture

The CNN model consists of the following layers:

1. **Preprocessing Layers**:
   - Normalization, random zoom, rotation, and flipping.

2. **Convolutional Layers**:
   - Three convolutional layers with ReLU activation and 5x5 filters.
   - Each convolutional layer is followed by max pooling and dropout for regularization.

3. **Fully Connected Layers**:
   - A dense layer with 256 units and ReLU activation.
   - Output layer with softmax activation for multi-class classification.

## Training

- The model is compiled using the Adam optimizer, Sparse Categorical Crossentropy loss, and accuracy as the evaluation metric.
- Training is conducted for 75 epochs with a batch size of 32.
- Validation is performed using a separate test dataset.

## Visualization

- Images and their corresponding labels are visualized from the test dataset using Matplotlib.

## Prediction

- After training, predictions are made on the test dataset using the `ImageDataGenerator` for preprocessing.
- Results include filenames and their predicted classes, saved as a CSV file.

## How to Run the Project

1. **Set up directories**:
   - Place training images in `train/` and testing images in `test/`.

2. **Train the model**:
   - Run the script to preprocess data, build, and train the CNN model.

3. **Save and Load Model**:
   - The trained model is saved as `covid-model.h5` for future inference.

4. **Make Predictions**:
   - Use the test dataset to generate predictions and save the results as a CSV file.

## Results

- The model outputs predictions for the test dataset, including class names and filenames.
- Accuracy and loss metrics are printed during training and validation.

## Future Improvements

- Add more robust data augmentation techniques.
- Experiment with different architectures and hyperparameters.
- Implement cross-validation for better model evaluation.
- Include Grad-CAM visualization to interpret model predictions.


