# _Music_Generation_with_LSTMs
Developed a music generation model using LSTM


# Install preety_midi  
      pretty_midi is a Python library for handling and processing MIDI (Musical Instrument Digital Interface) files.
      It is widely used for music information retrieval, music analysis, and computational musicology.
      The library provides tools for loading, manipulating, and generating MIDI data, making it useful for music research and
      various music-related machine learning projects.


# 2. Importing the essential libraries:

1. from tensorflow.keras.losses import SparseCategoricalCrossentropy
Sparse Categorical Crossentropy is a loss function designed for multi-class classification tasks where the target labels are represented as integer values rather than one-hot encoded vectors, offering a more memory-efficient and computationally faster alternative.
•  Efficiency: It is more efficient in terms of both memory and speed because it does not require the conversion of the labels into one-hot encoded format.
•  Scalability: Better suited for tasks with a large number of classes (e.g., language modeling, image classification with a large number of categories).
Let's assume you have a multi-class classification problem where the target variable has 5 classes (0, 1, 2, 3, 4).


2.  Optional:
Represents a type that could either have a specific type or be None.
It is shorthand for Union[type, None], meaning it can be either a certain type or None.
Example: Optional[str] means a value that can be of type str or None.

3. import glob
The glob module in Python is used for retrieving files and directories that match a specified pattern. It allows you to search for files using Unix-style pathname pattern expansion, such as wildcards (*, ?) and character ranges ([abc]), making it a useful tool for handling file and directory operations.

Find all .py files in the current directory and subdirectories
py_files = glob.glob('**/*.py', recursive=True)

4. import pathlib
The pathlib module in Python provides an object-oriented approach to handle filesystem paths. It was introduced in Python 3.4 as part of the standard library and offers an easy and intuitive way to work with file and directory paths.

# Preparing the dataset:
The dataset we will utilize for this project will contain multiple MIDI files with numerous piano notes that our model can use for training. The MAESTRO dataset contains piano MIDI files that the viewers can download from the link (https://magenta.tensorflow.org/datasets/maestro?ref=blog.paperspace.com#v200). I would recommend downloading the maestro-v2.0.0-midi.zip file. The dataset is only 57 MB in the compressed format and about 85 MB when extracted. We can use the data, which contains over 1200 files, to train and develop our deep learning model to generate music.


# Accessing the file  
 we will define the path to the directory where we have downloaded and extracted the data folder. The zip file, when extracted from the provided download link, should be placed in the working environment so that you can easily access all the contents present in the folder.
 
======================================================================================================================================================
*****************sample_file = filenames[1]
                  print(sample_file) ********************************************
Purpose: This selects a specific MIDI file from a list of filenames and prints its name.
Explanation: filenames[1] retrieves the second file name from the filenames list. This file name is stored in the variable sample_file, which is then printed. This helps you confirm which file is being analyzed.

**************pm = pretty_midi.PrettyMIDI(sample_file)
load the midi file
Purpose: To load and parse the selected MIDI file.
Explanation: pretty_midi.PrettyMIDI creates an object pm that represents the MIDI file. This object contains all the data from the MIDI file, such as notes, instruments, and their timing.


***********print('Number of instruments:', len(pm.instruments))
Print Number of Instruments
Purpose: To display the number of different instruments in the MIDI file.
Explanation: pm.instruments returns a list of instruments used in the MIDI file. len(pm.instruments) gives the count of these instruments. This helps understand the complexity of the MIDI file.

*********************************************************************
Get and Print Instrument Name

instrument = pm.instruments[0]
instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
print('Instrument name:', instrument_name)

Purpose: To retrieve and print the name of the first instrument in the MIDI file.
Explanation:
instrument = pm.instruments[0] selects the first instrument from the list.
instrument.program gives a number that represents the instrument.
pretty_midi.program_to_instrument_name(instrument.program) converts this number to a human-readable instrument name.
The instrument name is then printed.

************************************************************************

Extract and Print Notes
for i, note in enumerate(instrument.notes[:10]):
    note_name = pretty_midi.note_number_to_name(note.pitch)
    duration = note.end - note.start
    print(f'{i}: pitch={note.pitch}, note_name={note_name}, duration={duration:.4f}')


Purpose: To extract and print details about the first 10 notes of the selected instrument.
Explanation:
instrument.notes is a list of all notes played by the instrument.
[:10] selects the first 10 notes.
enumerate provides an index (i) and the note data (note) for each note.
pretty_midi.note_number_to_name(note.pitch) converts the note’s pitch (number) to its name (e.g., "C4").
note.end - note.start calculates the duration of the note.
The details for each note (index, pitch, name, and duration) are printed.


********************************************************************************************

Define the Function midi_to_notes

def midi_to_notes(midi_file: str) -> pd.DataFrame:
Purpose: Defines a function midi_to_notes that takes the path to a MIDI file as input and returns a pandas DataFrame containing information about the notes in the file.
Explanation: midi_file: str indicates that the input should be a string representing the file path. The function returns a DataFrame.

***********************************************************************************************

 Load the MIDI File and Extract the Instrument
  pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]

Purpose: To load the MIDI file and select the first instrument in the file.
Explanation: pretty_midi.PrettyMIDI(midi_file) creates a PrettyMIDI object from the file, and pm.instruments[0] selects the first instrument from the list of instruments in the file.

***********************************************************************************************
Initialize a Dictionary to Store Notes

notes = collections.defaultdict(list)
Purpose: To create a dictionary to store the extracted note data.
Explanation: collections.defaultdict(list) initializes a dictionary where each key’s default value is an empty list. This will store lists of note properties like pitch, start time, end time, etc.

************************************************************************************************
Sort Notes by Start Time

 sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

Purpose: To sort the notes by their start time to maintain chronological order.
Explanation: sorted(instrument.notes, key=lambda note: note.start) sorts the list of notes based on their start time. prev_start keeps track of the start time of the previous note for calculating the time difference (step).

***************************************************************************************************

Extract and Store Note Information

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

Purpose: To extract information from each note and store it in the notes dictionary.
Explanation:
note.start and note.end provide the start and end times of the note.
note.pitch gives the pitch of the note (a number representing the note).
notes['pitch'].append(note.pitch) appends the pitch to the list in the dictionary.
notes['step'].append(start - prev_start) calculates the time since the previous note started and appends it to the list.
notes['duration'].append(end - start) calculates the duration of the note and appends it to the list.
prev_start is updated to the current note’s start time for the next iteration.

****************************************************************************************

Convert to a DataFrame
return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

Purpose: To convert the dictionary of notes into a pandas DataFrame.
Explanation:
{name: np.array(value) for name, value in notes.items()} creates a dictionary where each key (note property) is associated with a numpy array of values.
pd.DataFrame() converts this dictionary into a DataFrame, which is a tabular data structure suitable for analysis.

****************************************************************************************

Apply the Function and Display the Result

raw_notes = midi_to_notes(sample_file)
raw_notes.head()

Purpose: To apply the midi_to_notes function to the sample_file and display the first few rows of the resulting DataFrame.
Explanation:
midi_to_notes(sample_file) calls the function with the path to the MIDI file and stores the result in raw_notes.
raw_notes.head() displays the first few rows of the DataFrame, allowing you to quickly inspect the extracted note data.

********************************************************************************************

# Converting to note names by considering the respective pitch values

Convert Pitch Values to Note Names

get_note_names = np.vectorize(pretty_midi.note_number_to_name)
sample_note_names = get_note_names(raw_notes['pitch'])
print(sample_note_names[:10])


Purpose: To convert numerical pitch values into human-readable note names.
Explanation:
np.vectorize(pretty_midi.note_number_to_name) creates a vectorized version of pretty_midi.note_number_to_name, allowing it to apply this function to each element in an array.
get_note_names(raw_notes['pitch']) converts the array of pitch numbers into note names.
print(sample_note_names[:10]) prints the first 10 note names to check the conversion.
*******************************************************************************************************

Plot the Piano Roll

def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
    if count:
        title = f'First {count} notes'
    else:
        title = f'Whole track'
        count = len(notes['pitch'])
        
        plt.figure(figsize=(20, 4))
        plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
        plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
        
        plt.plot(plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")        
        plt.xlabel('Time [s]')
        plt.ylabel('Pitch')
        _ = plt.title(title)


Purpose: To visualize the musical notes of the piano over time in a piano roll format.

Explanation:

def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None): defines a function to create the plot.

Handling the count Parameter:

If count is provided, it plots only the first count notes and sets the title accordingly.

If count is not provided, it plots the entire track and adjusts the title to "Whole track".

Create the Plot:

plt.figure(figsize=(20, 4)) sets the size of the plot.
plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0) duplicates the pitch values for plotting.
plot_start_stop = np.stack([notes['start'], notes['end']], axis=0) creates an array with start and end times for plotting.
plt.plot(plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".") plots the start and end times against the pitch values. It uses blue color and dot markers.
plt.xlabel('Time [s]') and plt.ylabel('Pitch') label the axes.
_ = plt.title(title) sets the title of the plot based on the count parameter.



























