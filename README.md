# ML-Cancer-Detection

---

### Table of Contents


- [Description](#description)
- [How To Use](#how-to-use)
- [References](#references)
- [Author Info](#author-info)

![Schematic-workflow-diagram-of-our-proposed-method-of-breast-cancer-prediction-with-data (1)](https://github.com/mSaadSalman/ML-Cancer-Detection/assets/105026161/eb69c8ab-49e1-4e04-9ee3-7eae16dc0ee0)


---
## Dataset

The Wisconsin Diagnostics dataset serves as the foundation for this project. It contains various features extracted from fine-needle aspirates (FNAs) of breast masses, and the target variable is the diagnosis, which can be either malignant (M) or benign (B).

---
## Description

In this project, a Machine Learning model was developed to detect breast cancer using the Wisconsin Diagnostics dataset. The dataset was split into training and testing data, with the training data used to train the model, and the testing data used to assess its accuracy.

To gain insights into the data and identify correlations between cancer diagnoses (Malignant or Benign) and different features, heat maps and the seaborn correlation library were employed. This visualization approach provided solid evidence of how various features relate to the cancer diagnosis, aiding in feature selection and engineering.

To implement the Machine Learning model, the scikit-learn API reference library was utilized. Specifically, the StandardScaler was applied to standardize the data, ensuring that all features have a mean of 0 and a standard deviation of 1. This standardization process helps prevent features with different scales from dominating the model's training. Additionally, the train_test_split function from scikit-learn was used to create the training and testing sets, allowing for an accurate evaluation of the model's performance on unseen data.

Furthermore, the knowledge acquired from the Stats 3Y03 course was leveraged to handle data standardization actions effectively. The course likely provided a solid foundation in statistical concepts, which proved valuable in the Machine Learning workflow

#### Libraries Used

This project leverages the following Python libraries:

- Pandas and NumPy for data manipulation and analysis
- Matplotlib and Seaborn for data visualization
- Scikit-learn for machine learning tasks, including data preprocessing, model training, and evaluation

---

## How To Use

#### Installation
```html
    pip3 install numpy
    pip3 install pandas
    pip3 install matplotlib
    pip3 install seaborn
    pip3 install sklearn
```

---

### Backlog 
| Id  | Feature  | Status  |  Date  |
|:-:  |---       | :-:     | :-:     |
| F01 | Proccessing Missing Values |  D | July-17-2023  |
| F02 | Count Plot |  D |  July-17-2023  |
| F03 | Correlation matrix and heatmap |  D | July-18-2023   |
| F04 | Splitting Data into train and test sets |  D | July-18-2023  |
| F05 | Feature Scaling |  D | July-18-2023 |
| F06 | Logistic Regression |  D |  July-20-2023 |
| F07 |Random forest Classifier  |  D |  July-27-2023 |


---

## Author Info

- Saad Salman, Student of Software Engineering at McMaster University
- Website - []()
