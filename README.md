# _Music_Generation_with_LSTMs
Developed a music generation model using LSTM


  
# 1. Data Preparation:
Load MIDI Files: Load the MIDI files into the program using pretty_midi to extract the musical notes.
Parse Notes: Define a function midi_to_notes to extract musical notes (pitch, start time, end time) from each MIDI file and store them in a DataFrame.
Concatenate Notes: Combine all notes extracted from multiple files into a single dataset for training.


# 2. Dataset Preparation:
Convert Notes to Training Format: Arrange the notes into sequences with a specific length (seq_length) to create input sequences and labels.
Normalize Pitch Values: Scale the pitch values between 0 and 1 for easier model training.
Create TensorFlow Dataset: Convert the sequences into a TensorFlow dataset using tf.data.Dataset.


# 3. Define Batch and Buffer Sizes:
Set Batch Size and Buffer Size: Define batch_size (number of sequences per batch) and buffer_size for shuffling the dataset. This improves the randomness of the training process.
Batch and Prefetch Data: Use .batch() to batch the data and .prefetch() to allow the dataset to be fetched in the background while the model is training.


# 4. Define Custom Loss Function:
Mean Squared Error with Positive Pressure: Define a custom loss function mse_with_positive_pressure to calculate the loss while adding a penalty for negative values in predicted outputs (e.g., step and duration).


# 5. Build the Model:
Define Model Architecture: Build the LSTM model using Keras. The input layer accepts sequences of notes, and the LSTM layer captures patterns in sequences. The output layer consists of three branches predicting pitch, step, and duration.
Compile the Model: Use a dictionary of loss functions (custom loss for step and duration, sparse categorical cross-entropy for pitch) to compile the model. Set the optimizer (Adam) and learning rate.


# 6. Create Callbacks:
Model Checkpoints: Set up a callback to save the model's weights at checkpoints during training.
Early Stopping: Use early stopping to halt training if the loss does not improve after a certain number of epochs (patience).


# 7. Train the Model:
Fit the Model: Train the model using the prepared dataset, loss functions, and callbacks. Set the number of epochs for training.

# 8. Generate Music:
Define Prediction Function: Create a function predict_next_note that uses the trained model to predict the next note based on the current sequence of notes. The temperature parameter controls the randomness of the predictions.
Initialize Input Notes: Prepare a starting sequence of notes from the training data to begin music generation.
Generate Notes Iteratively: Use a loop to predict new notes, append them to the generated sequence, and update the input notes for the next prediction.


# 9. Save Generated Music:
Convert Notes to MIDI: Use the function notes_to_midi to convert the generated notes into a MIDI file format.
Save Output File: Save the generated MIDI file to a specified output path (out_file).


# 10. Repeat and Refine:
Adjust Parameters: Experiment with model parameters, such as batch_size, seq_length, and learning_rate, to optimize the model performance.
Test and Iterate: Continue testing, refining, and re-training the model to improve the quality of the generated music.


























****************************************************************************************************************************************************************





