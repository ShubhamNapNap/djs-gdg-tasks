# djs-gdg-tasks
For EDA task:
# Formula 1 Historical Data Analysis: DNF Classification EDA

This project performs Exploratory Data Analysis (EDA) on a historical Formula 1 dataset to clean and transform the data, visualize relationships between key features, and derive insights relevant to predicting race outcomes, particularly Did Not Finish (DNF) events.

## 🗄️ Dataset Description

* **Source:** [Kaggle: F1 DNF Classification](https://www.kaggle.com/datasets/pranay13257/f1-dnf-classification)
* **Content:** A comprehensive historical dataset spanning multiple decades of Formula 1, combining data on race results, drivers, constructors, and circuits.
* **Target Variable:** The feature **`target_finish`** is used as the target for classification: `1` (Finished) or `0` (DNF/Retired).

---

## 🛠️ Data Cleaning and Transformation

The initial data exploration revealed the need for several cleaning and feature engineering steps:

### 1. Feature Selection
The initial dataframe was filtered to include only the most relevant columns, consolidating information about the race, the driver, the constructor, and the circuit.

### 2. Data Type Conversion and Age Calculation
* The `dob` and `date` columns were converted to `datetime` objects.
* **Feature Engineering:** A new, crucial numerical feature, **`driver_age_at_race`**, was calculated by finding the difference between the race date and the driver's date of birth and converting the result into years.
* **Feature Engineering:** A consolidated **`driver_name`** column was created from `forename` and `surname`.

### 3. Duplicate Handling
Duplicate rows based on the combination of `year`, `round`, `driverRef`, and `constructorRef` were removed to ensure each result represents a unique driver/constructor entry for a specific race.

### 4. Target Variable Creation
A categorical label **`finish_status`** was created from `target_finish` for clear visualization:
* `target_finish = 1` $\rightarrow$ 'Finished'
* `target_finish = 0` $\rightarrow$ 'DNF/Retired'

---

## 📊 Exploratory Visualizations and Analysis

The following visualizations were used to explore the relationship between key numerical and categorical features and the race outcome (`finish_status`).

### 1. Driver Age vs. Laps with Finish Status (Scatter Plot)

**Chart Type:** Scatter Plot (`driver_age_at_race` vs. `laps`)

* **What it Shows:** This plot examines the total number of laps completed against the driver's age at the time of the race, segmented by the race outcome (Finished/DNF).
* **Why it Matters:** The chart clearly shows that the **'DNF/Retired' points cluster heavily at lower `laps` counts** (primarily 0 to $\sim30$), regardless of the driver's age. The 'Finished' points consistently span the higher `laps` range. This confirms the quality of the `laps` feature in determining the target outcome, and it indicates that there is **no immediate linear relationship between driver age and early retirement**.

### 2. Pairwise Relationships of Key Numerical Features (Pair Plot)

**Chart Type:** Pair Plot (Matrix of scatter plots and histograms)

* **Variables Analyzed:** `grid`, `laps`, `positionOrder`, `driver_age_at_race`
* **What it Shows:** The off-diagonal plots display the scatter plots between every pair of numerical features, and the diagonal plots show the distribution (histogram) of each feature, all colored by the `finish_status`.
* **Why it Matters:**
    * **`grid` vs. `positionOrder`:** This relationship is the most critical. There's a strong positive correlation, indicating that **starting position (`grid`) is highly predictive of the final finishing position (`positionOrder`)**. The **DNF points are scattered across all grid positions**, suggesting that starting position is a key factor for *finishing well*, but not the *only* predictor for retirement.
    * **`positionOrder` Distribution:** The distribution is heavily skewed towards smaller numbers (better finishing positions) for the 'Finished' group, while the 'DNF/Retired' group is concentrated at higher `positionOrder` values, representing lower final classifications (often 15-22, depending on the historical field size).

### 3. Total Finishes vs. DNFs for Top 10 Constructors (Count Plot)

**Chart Type:** Count Plot

* **What it Shows:** This plot compares the absolute counts of 'Finished' vs. 'DNF/Retired' outcomes for the top 10 constructors historically (based on total race entries).
* **Why it Matters:** This reveals the historical reliability (or lack thereof) of the most dominant teams.
    * **Ferrari** and **McLaren** have the highest total entries (and total finishes), as expected for long-standing constructors.
    * The plot reveals **relative reliability**. While all top constructors have many more finishes than DNFs, comparing the height of the green bar to the red bar gives a quick visual indication of the DNF rate. For instance, **Renault** appears to have a slightly higher proportion of DNFs compared to teams like **Williams** or **McLaren** relative to their total entry count. This suggests **`constructorRef` is a powerful categorical feature** for predicting race outcome.

---

## 💡 Key Insights for Machine Learning Modeling

Based on the exploration, here are three strong insights that can guide the choice and construction of a classification model for predicting DNFs:

### Insight 1: Grid Position is a Primary Feature, but Not a DNF Determinant.

* **Evidence:** The Pair Plot shows a strong correlation between **`grid`** and **`positionOrder`**, confirming its value in predicting a *successful finish* (P1, P2, etc.). However, DNF points are distributed across all starting grids.
* **ML Implication:** The model must include `grid` as a key feature, but a purely linear model won't suffice. The relationship between `grid` and DNF is non-linear and likely mediated by other factors (e.g., *low grid position increases risk of first-lap accidents*). **Feature interaction** between `grid` and circuit properties might be necessary.

### Insight 2: Constructor Identity is a Strong, Necessary Categorical Feature.

* **Evidence:** The Constructor Count Plot shows distinct, measurable differences in the ratio of Finishes to DNFs across the top teams. The reliability history of a constructor is not uniform.
* **ML Implication:** **`constructorRef`** should be included in the model, likely using a form of **Target Encoding** (e.g., using the historical DNF rate of the constructor as a numerical feature) rather than simple One-Hot Encoding due to the high cardinality of the feature across the entire dataset.

### Insight 3: The Problem is Heavily Imbalanced and Requires Specialized Model Types.

* **Evidence:** Visually, the 'Finished' points (green) vastly outnumber the 'DNF/Retired' points (red) across all plots (especially clear in the constructor chart and the pair plot distributions). The overall DNF rate is significantly lower than the finish rate.
* **ML Implication:** This is a classic **class imbalance problem**.
    * **Metric Choice:** Accuracy will be misleading. Metrics like **Recall** (to minimize false negatives—predicting a finish when it was a DNF) and the **F1 Score** will be more appropriate for evaluation.
    * **Model/Strategy Choice:** The model will benefit from techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) during training or the use of models inherently robust to imbalance, such as **Tree-Based Classifiers** (e.g., XGBoost, Random Forest), which can capture complex, localized patterns that lead to DNFs.
