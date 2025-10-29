Web Version – Pre-Fire Detection System in file named app.py

The web version of the Pre-Fire Detection System offers a simple, interactive interface that predicts the likelihood of a fire based on environmental inputs such as temperature, humidity, gas concentrations, and air quality. It uses a trained machine learning model to analyze these values and display the fire-risk level in real time. Users can enter data manually and receive results both as numerical probabilities and visual outputs.

To make the experience engaging, the web app includes voice-based alerts that announce the risk level when the user clicks the “Voice Alert” button. It also features a pie chart visualization showing which atmospheric factors contribute most to the predicted fire risk, allowing better understanding and data-driven insights.

Required Libraries for Web Version:

pandas – for data handling

numpy – for numerical processing

matplotlib – for visual charts

scikit-learn – for machine learning predictions

gtts – for generating online text-to-speech alerts

flask / streamlit (depending on deployment) – for web interface



Local Desktop Version – Pre-Fire Detection System in file named terminal.py

The local desktop version runs entirely offline and offers a voice-interactive, GUI-based experience. Using Tkinter, it allows users to enter or speak atmospheric readings such as temperature, humidity, gas levels, and particulate matter. The trained machine learning model instantly predicts the fire risk level—Low, Medium, High, or Critical—and provides both visual pop-ups and spoken safety alerts using pyttsx3.

This version also logs all predictions to a CSV file for record-keeping and enables users to visualize factor contributions via a dynamic pie chart. It is lightweight, private, and ideal for use in environments where internet access may be limited.

Required Libraries for Local Desktop Version:

pandas – data handling

numpy – numerical operations

matplotlib – feature visualization

scikit-learn – model training and predictions

pyttsx3 – offline voice alerts

SpeechRecognition – voice input

tkinter – graphical user interface

pyaudio – microphone input for speech recognition
