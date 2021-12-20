# CNN_sleep_stage_scoring
Classifying different sleep phases using the 6 EEG channels used in PSG studies, using amplitude envelope in 30s epochs. 

Funcs.py: The script containing the functions to filter, and detect outliers (i.e. motion artefacts).

Class.py: The python class where the user can define a class for the CNN model to train on, in a one-versus-all manner. 
