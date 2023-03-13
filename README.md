# LSTM networks for ECG-based biometric authentication systems

This is the repository for Master in Internet of Things (IoT): Applied Technologies Thesis: "LSTM networks for ECG-based biometric authentication systems" (2020).

This is a MATLAB-based project.
<br /><br />

## Project
----------------------------------------
The project aims to leverage ECG signals of 10 subjects to authenticate them in a ECG-based biometric authentication system scenario. This is considered as a multi-class classification problem where the goal is to identify which one of the known subjects is the analyzed one.

The developed algorithm includes a **LSTM (Long-Short Term Memory) neural network**, which extracts relationships and hidden dependencies in the ECG signals in order to identify a specific individual among a closed dataset. This is a type of recurrent network that is particularly suitable when dealing with time series, as is the case of the ECG signal. 

- **ECG_biometrics.m**: main code to test proposed algorithm with a ECG-signals dataset.

- **ReadECGData.m**: code that reads ECG signals from .mat files in "Sujetos" directory and creates ECGSignalsData.mat variable with Signals and associated Classes.
- **segmentSignals.m**: Matlab native script to segment signals in smaller windows with standarized 9000 samples each one.
- **dataset.csv**: CSV file that contains the names of the .mat files in "Sujetos" to be processed and their associated label/class.

<br />

## ECG Dataset 
----------------------------------------
A subset of 10 individuals from the public **MIT-BIH Normal Sinus Rhythm database**, composed of electrocardiographic signals from 18 non-pathological individuals, is used to validate the algorithm. 

Link: https://drive.google.com/drive/folders/10TjYyd9IUwSXYoG1xnIiCKuNKnzzS5mb?usp=sharing

<br />

## Contributors
----------------------------------------

Roberto Hernández Ruiz

July 2020