# _Music_Generation_with_LSTMs
Developed a music generation model using LSTM


# Install preety_midi  
      pretty_midi is a Python library for handling and processing MIDI (Musical Instrument Digital Interface) files.
      It is widely used for music information retrieval, music analysis, and computational musicology.
      The library provides tools for loading, manipulating, and generating MIDI data, making it useful for music research and
      various music-related machine learning projects.


# 2. Importing the essential libraries:
from tensorflow.keras.losses import SparseCategoricalCrossentropy
Sparse Categorical Crossentropy is a loss function designed for multi-class classification tasks where the target labels are represented as integer values rather than one-hot encoded vectors, offering a more memory-efficient and computationally faster alternative.
•  Efficiency: It is more efficient in terms of both memory and speed because it does not require the conversion of the labels into one-hot encoded format.
•  Scalability: Better suited for tasks with a large number of classes (e.g., language modeling, image classification with a large number of categories).
Let's assume you have a multi-class classification problem where the target variable has 5 classes (0, 1, 2, 3, 4).
