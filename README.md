🐶 Dog Breed Classification using Transfer Learning
This project classifies 70 different dog breeds from images using Transfer Learning with MobileNetV2. It applies data augmentation, builds a custom classification head, and achieves high prediction accuracy. Built using TensorFlow 2.x and Keras.

📂 Dataset
70 folders, one per dog breed.

Multiple images per breed.

Supervised image classification setup.

🛠️ Model Architecture
Base Model: MobileNetV2 (pre-trained on ImageNet, frozen layers).

Custom Layers: GlobalAveragePooling ➔ Dense(1024, ReLU) ➔ Dense(70, Softmax).

Data Augmentation: Rotation, shift, shear, zoom, and horizontal flip.

🏋️ Training Details
Batch size: 32

Epochs: 10

Optimizer: Adam

Loss function: Categorical Crossentropy

Validation split: 20%

🔮 Prediction Pipeline
Resize image to 224x224.

Normalize pixel values to [0,1].

Expand dimensions to batch size.

Predict dog breed and confidence score.

🚀 How to Run
1. Install Dependencies
pip install tensorflow numpy
2. Train the Model
python train_model.py
3. Make Predictions
4. python predict.py --image path_to_your_image.jpg
📈 Results
Achieved high accuracy on validation and test images.

Perfect prediction on unseen samples.

Deploy the model to a web or mobile application.

📜 License
This project is open-source and free to use for educational purposes.

✨ Thank you! ✨
