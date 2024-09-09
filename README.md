# Project-Titanic
Another perspective on the shipwreck

# Project Titanic: Predicting Passenger Survival

## Overview
The 'Titanic Project' is a data analysis initiative focused on predicting the survival of passengers aboard the Titanic based on various key factors. Using a well-known dataset from the Titanic disaster, this project aims to apply machine learning techniques and data analysis to provide insights into the dynamics of survival during this tragic event.

## Objectives
The main objectives of this project are:
1. Predict Survival Based on Nationality: Investigate the influence of passengers' nationality on their likelihood of survival.
2. Explore the Role of Class and Gender: Understand how class (1st, 2nd, 3rd) and gender (male, female) affected survival rates.
3. Develop a Global Predictive Model: Build a predictive model that integrates nationality, class, and gender to forecast the survival of passengers.
4. Additional Insights: Analyze additional variables such as age and fare to enhance the model's accuracy and provide further insights.

## Datasets
We used the famous 'Titanic dataset', which contains details about passengers, their survival status, and other relevant features. The key variables used in this project include:
- Passenger Information: Name, gender, and nationality.
- Class: Class of travel (1st, 2nd, 3rd).
- Survival Status: Whether the passenger survived (1 = yes, 0 = no).
- Age and Fare: Additional features such as the passenger's age and the price of their ticket.

## Data Analysis Approach
The project follows a structured approach:
1. Data Cleaning: Handling missing values, outliers, and standardizing variable formats.
2. Exploratory Data Analysis (EDA): Investigating survival rates based on nationality, class, and gender using visualizations and descriptive statistics.
3. Feature Engineering: Creating new features or refining existing ones to improve model performance.
4. Modeling: Using machine learning models such as **Logistic Regression**, **Random Forest**, and **XGBoost** to predict survival.
5. Evaluation: Evaluating model performance using accuracy, precision, recall, and other metrics. Model comparison to determine the best approach.

## Tools & Technologies
The following tools and technologies were used for this project:
- Python: The main programming language for data analysis and modeling.
- Pandas & NumPy: For data manipulation and cleaning.
- Matplotlib & Seaborn: For data visualization.
- Scikit-learn: For building and evaluating machine learning models.
- Jupyter Notebooks: For documenting and presenting the analysis process.
- GitHub: Version control and project management.

## Results & Insights
By analyzing the data, we found that:
- 'Gender' played a critical role in survival, with females having a significantly higher chance of survival than males.
- 'Class' also impacted survival, with first-class passengers having the highest survival rates, followed by second and third class.
- 'Nationality', although less frequently discussed in Titanic analyses, showed interesting patterns of survival likelihood based on embarkation points and countries of origin.

## Next Steps
Future improvements to this project may include:
1. Incorporating more advanced machine learning models like **XGBoost** and **Gradient Boosting** to improve prediction accuracy.
2. Investigating the impact of more nuanced features like family size, relationships between passengers, and fare per person.
3. Deploying the predictive model using a web interface (e.g., **Streamlit** or **Flask**) to make the model interactive for users.

## Conclusion
This project showcases the use of data analysis and machine learning to explore survival patterns in the Titanic disaster. The combination of nationality, class, and gender provides a unique perspective on survival rates, adding depth to the analysis of this historical event.

## How to Use
To explore this project:
1. Clone the repository:
   ```bash
   git clone https://github.com/ton-utilisateur/Project-Titanic.git
