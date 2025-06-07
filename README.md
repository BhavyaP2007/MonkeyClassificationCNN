# MonkeyClassificationCNN
A deep learning-powered image classification web app that identifies monkey species from photos. Built with a custom CNN in PyTorch and a lightweight Streamlit interface, the model classifies 10 monkey species with high accuracy. Upload an image and get instant predictions!

To run the project :
1.Ensure that the 10_Monkey_Species.zip file is present in the root directory. Extract it in the main folder.
2.Install the following depencies
  pip install torch torchvision streamlit matplotlib pandas numpy pathlib PIL.
3. In the terminal - streamlit run gui.py

 🧠 Model Overview
Architecture: Custom CNN with multiple convolutional layers, batch normalization, ReLU activations, and max pooling.

Training: Utilizes CrossEntropyLoss and the Adam optimizer.

Performance: Achieves approximately 94% training accuracy and 80% testing accuracy.

Inference: The trained model weights are saved in model_weights.pth for quick inference without retraining.

Impact and Significance
This project showcases the practical application of deep learning in image classification tasks. By integrating PyTorch for model development and Streamlit for deployment, it demonstrates a full-stack approach to machine learning projects.

Key Highlights:
1. End-to-End Solution: From data preprocessing and model training to deployment.
2. User-Friendly Interface: Interactive web app for real-time predictions.
3. Reproducibility: Clear instructions and organized codebase for easy replication.
