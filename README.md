# Bangkit Academy 2024 - Product-Based Capstone Team - C242-PS027

## 📖 Peduli Bumil - Mechine Learning
Peduli Bumil is a health application designed to help pregnant women monitor and maintain their health during pregnancy. This application provides main features such as a pregnancy age tracker, pregnancy risk detection based on health parameters (height, weight, body temperature, blood pressure, blood sugar, age, and heart rate), as well as risk classification into High, Medium, or Low categories. In addition, this application is equipped with informative articles, an interactive chatbot to answer pregnancy-related questions,and personal reminders to maintain health.

This project uses Convolutional Neural Networks (CNN) models to analyze health data and detect pregnancy risk levels. This model is converted to TensorFlow Lite (TFLite) format to enable implementation on edge devices or mobile applications, thereby supporting broader accessibility, especially for pregnant women in remote areas. This application contributes to the Indonesian government's efforts to reduce maternal mortality rates by preventing complications and enhancing health understanding.

Pregnancy Risk Detection Model
This model uses relevant health features from the dataset to predict the level of pregnancy risk (High, Medium, Low) for users. This model is built using Convolutional Neural Networks (CNN), which enables high-accuracy health data analysis to support decision-making related to pregnancy care.

## Model
In this App use 2 Model
1. CLassification Model for Pregnancy Risk Classification
```
https://github.com/BangkitPeduliBumil/Machine-Learning/blob/efb3541930faaf407c36a9d1387c8b426b1f8581/ClassificationModel_PregenancyRisk_Notebook.ipynb
```
2. Chatbot using Vertex AI Agent Builder Conversation
```
https://github.com/BangkitPeduliBumil/Machine-Learning/tree/efb3541930faaf407c36a9d1387c8b426b1f8581/Data%20Vertex%20Used
```

## Dataset
```
https://drive.google.com/file/d/1ZpvL5YCr-1Rw_kAKpO38fs7oGMmakp8S/view
```

![](https://github.com/BangkitPeduliBumil/asset/blob/248cec9cfa48e637877265b34d08d29bf4bc50a5/data.png)

## Pair Plot
<img src="https://github.com/BangkitPeduliBumil/asset/blob/248cec9cfa48e637877265b34d08d29bf4bc50a5/pair-plot.png" alt="Pair Plot" width="50%">

#### Description
In this project, a pair plot is used to visualize the relationships between various health parameters of pregnant women, such as age, body temperature, Systolic Blood Pressure(mm Hg), Diastolic Blood Pressure(mm Hg), body mass index (BMI), blood sugar levels, and heart rate. This visualization helps us understand the patterns and correlations between variables that can contribute to the detection of pregnancy risks. With this analysis, we can identify the most relevant features for our prediction model.

## Bar Plot
![](https://github.com/BangkitPeduliBumil/asset/blob/248cec9cfa48e637877265b34d08d29bf4bc50a5/bar-plot.png)
#### Description
Bar plot is used to show the distribution of pregnancy risk categories (Low, Medium, High) based on health data. This visualization provides a clear picture of the prevalence of risk in the dataset.This bar plot also aids in the process of model evaluation and validation.

## Libraries Used

The following libraries were used in this project:

| **Library**      | **Purpose**                                                                                   |
|-------------------|-----------------------------------------------------------------------------------------------|
| **NumPy**        | For numerical operations and array manipulation.                                              |
| **Pandas**       | For manipulation and analysis of tabular data.                                                |
| **Matplotlib & Seaborn** | For data visualization, such as feature distribution and prediction results.                |
| **TensorFlow/Keras** | To build, train, and convert neural network models like CNN to TFLite format.              |
| **scikit-learn**  | For data preprocessing, model evaluation, and implementation of clustering algorithms (K-Means). |
| **NLTK/Spacy**    | For text processing in Natural Language Processing (NLP) on article recommendation models.    |
| **TFLite**        | For converting machine learning models to a format compatible with mobile or edge devices.    |
|


