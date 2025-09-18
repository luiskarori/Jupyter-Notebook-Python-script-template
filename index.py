# Assignment: Data Loading, Analysis, and Visualization with Pandas and Matplotlib
# Author: Luis Karori
# Objective: Explore dataset, perform basic analysis, and create visualizations.

# ===============================
# Task 1: Load and Explore Dataset
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load dataset
try:
    iris = load_iris(as_frame=True)
    df = iris.frame  # Convert sklearn dataset to pandas DataFrame
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print("❌ Error loading dataset:", e)

# Display first 5 rows
print("\n--- First 5 Rows ---")
print(df.head())

# Check structure
print("\n--- Info ---")
print(df.info())

# Check missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# (No missing values in Iris dataset, but if any:)
df = df.dropna()

# ===============================
# Task 2: Basic Data Analysis
# ===============================

# Basic statistics
print("\n--- Descriptive Statistics ---")
print(df.describe())

# Grouping by species and computing mean
print("\n--- Mean Values Grouped by Species ---")
grouped = df.groupby("target").mean()
print(grouped)

# Add species names for clarity
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

print("\n--- Interesting Findings ---")
print("1. Iris-setosa generally has the smallest petal length and width.")
print("2. Iris-virginica tends to have the largest sepal and petal measurements.")
print("3. Iris-versicolor lies in between the two.")

# ===============================
# Task 3: Data Visualization
# ===============================

plt.style.use("seaborn-v0_8")

# 1. Line chart (petal length trend by index)
plt.figure(figsize=(8,5))
plt.plot(df.index, df["petal length (cm)"], label="Petal Length")
plt.title("Line Chart: Petal Length Across Samples")
plt.xlabel("Sample Index")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()

# 2. Bar chart (average petal length per species)
plt.figure(figsize=(8,5))
sns.barplot(x="species", y="petal length (cm)", data=df, estimator="mean", palette="viridis")
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram (distribution of sepal width)
plt.figure(figsize=(8,5))
plt.hist(df["sepal width (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Histogram: Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot (sepal length vs petal length)
plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df, palette="deep")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
