# Jupyter-Notebook-Python-script-template
📊 Data Analysis & Visualization Assignment
🎯 Objective

The goal of this assignment is to:

Load and explore a dataset using Pandas.

Perform basic statistical analysis.

Visualize the dataset using Matplotlib and Seaborn.

📂 Dataset

For this project, I used the Iris dataset, a classic dataset in machine learning that contains measurements of sepal length, sepal width, petal length, and petal width for three species of iris flowers:

Iris-setosa

Iris-versicolor

Iris-virginica

🛠️ Steps Performed
🔹 Task 1: Load & Explore the Dataset

Loaded the Iris dataset using sklearn.datasets.

Inspected the first few rows with .head().

Checked data types and missing values with .info() and .isnull().

Cleaned the dataset (removed/fixed missing values if any).

🔹 Task 2: Basic Data Analysis

Generated descriptive statistics with .describe().

Grouped data by species and calculated mean values.

Observed key findings such as species differences in petal and sepal sizes.

🔹 Task 3: Data Visualization

Created 4 visualizations to understand trends, comparisons, distributions, and relationships:

Line Chart – Petal length trend across samples.

Bar Chart – Average petal length per species.

Histogram – Distribution of sepal width.

Scatter Plot – Sepal length vs petal length, colored by species.

📈 Tools & Libraries Used

Pandas → Data manipulation & exploration.

Matplotlib → Data visualization.

Seaborn → Enhanced visualization styling.

Scikit-learn → Loading the Iris dataset.

✅ Findings & Observations

Iris-setosa has the smallest petals overall.

Iris-virginica has the largest petal dimensions.

Iris-versicolor lies in between the two species.

Sepal and petal lengths show a clear positive relationship.
