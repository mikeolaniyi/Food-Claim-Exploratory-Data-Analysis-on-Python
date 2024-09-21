# Food Claim Exploratory Data Analysis on Python
By Michael Michael Jeremiah
![Food_claim](https://github.com/mikeolaniyi/Food-Claim-Exploratory-Data-Analysis-on-Python/assets/120651356/141bc6d3-6510-4303-a229-f0eb5a15d2c0)

### Table of Contents:

- [Business Background](#business-background)
- [Business Problem](#business-problem)
- [Business Questions](#business-questions)
- [Data cleaning and validation](#data-cleaning-and-validation)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Findings](#Findings)
- [Recommendation](#Recommendation)


### Business Background
Vivendo is a fast food chain in Brazil with over 200 outlets. Customers often claim compensation from the company for food poisoning. The legal team processes these claims. The legal team has offices in four locations. The legal team wants to improve how long it takes to reply to customers and close claims. The head of the legal department wants a report on how each location differs in the time it takes to close claims.


### Business Problem:
The legal department is dealing with:
- Inefficient customer response times.
- Unanalyzed geographical differences in claim processing time.
- Inadequate insights into claim closure duration


### Business Questions:
- How many customers are there in each location?
- Is there any relationship between Locations and the time it takes to close claims?
- How does the time to close claims looks like?



![Food Claim EDA On Python Project Executive Summary (1)](https://github.com/mikeolaniyi/Food-Claim-Exploratory-Data-Analysis-on-Python/assets/120651356/70a1948a-057d-4895-b47c-f440c1418e81)




```python
# import all the the required library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.style as style
style.use('ggplot')
plt.figure(figsize=(10,6))
sns.set_palette(['gray'])

# Load the dataset
df = pd.read_csv('food_claims.csv')


# Check all info and description
df.head()
df.info()
df.describe()
df.boxplot()

# Check for null values
null_counts = df.isnull().sum()
print(null_counts)

# - Two fields found with null values.
```
**Output: Data head**

![image](https://github.com/user-attachments/assets/7645a83f-b888-4969-b319-23476ff02275)

**Output: Dataset info**

![image](https://github.com/user-attachments/assets/2145ace8-43e3-4f96-b444-3be70e9fd84f)

**Output: Dataset statistic description**

![image](https://github.com/user-attachments/assets/49000013-2411-425e-8f66-8fff2c33e73c)

**Output: Dataset numeric statistic data points**

![image](https://github.com/user-attachments/assets/a5b7902f-b496-4059-80f7-b5a11935bd58)

```python
# Check for null values
null_counts = df.isnull().sum()
print(null_counts)
```
- Two fields found with null values.

**Output: null count**

![image](https://github.com/user-attachments/assets/f1eaabe0-fba6-4128-9db4-cd0ee40dbf46)



**Data Validation:** Data cleaning and validating for all the columns.

**1. Claim_ID**

```python


# checking if there's any duplicates
has_duplicates = df['claim_id'].duplicated().any()
print(has_duplicates)

# # checking for numebr of duplicates
df['claim_id'].duplicated().sum() 

# checking for number of unique ID
len(df['claim_id'].unique()) 

```
 - Unique 2000 records.
Output: No duplicates and 2000 unique claim ids

![image](https://github.com/user-attachments/assets/c5ef1388-4923-4c97-9a61-a0bbb71ef13b)


**2. Time_to_close**

```python
df['time_to_close'].info()

# checking Missing values
df['time_to_close'].isnull().sum()
df['time_to_close'].describe()

check = df['time_to_close'] >= 400
filtered_data = df[check]

# Print the filtered data
print(filtered_data)
```
No missing values

Output: time_to_close description

![image](https://github.com/user-attachments/assets/e1f0cdd7-fb59-4844-9e34-fb9f70502e35)

Output: time_to_close greater than 400

![image](https://github.com/user-attachments/assets/f7473fd7-d6ba-4ece-8fb1-26339bac6d38)


**3. Claim_amount**

![image](https://github.com/user-attachments/assets/cea1f60d-b08b-4ee0-a9fe-07ec94be19fd)

The claim amount column has the Brazilian dollar sign 'R$' before the amount, making the column a string data type. I have to remove the R$ preset and set the column data type to float.
```python

# Removing R$ from 'claim_amount'
df['claim_amount'] = df['claim_amount'].str.replace('R\$ ', '')

# Convert the claim_amount to a float data type
df['claim_amount'] = df['claim_amount'].astype(float)

# check the data info
df['claim_amount'].info
```

 - The claim_amount Claim cleaned and converted to float datatype

Output:

![image](https://github.com/user-attachments/assets/cbf65fd7-12e8-4d23-a5cb-28e43252c70f)



**4. Amount_paid:**
This column contains the claims amount paid out to customers, I want to check for any inconsistencies, null values or errors.
```python
# 4. Amount_paid

# view claim_amount description
df['amount_paid'].describe()

# check the data info
df['amount_paid'].info

# check amount equal to 0
count_am = (df['amount_paid'] == 0).sum()
print(count_am)

# chech for nulls
null_counts_am = df['amount_paid'].isnull().sum()
print(null_counts_am)
```
Output: 36 null values

![image](https://github.com/user-attachments/assets/1bae3c4d-d1cb-470b-b697-353dc22908ee)

There were 36 null values, after carefully observing the null values, I decided that it was best to replace the null values with the amount_paid median value.

**Replace null values with amount median value:**

```python
# Replace null values with amount median value
median_amount_paid = df['amount_paid'].median()
df['amount_paid'].fillna(median_amount_paid, inplace=True)

Check for nulls
# chech for nulls
null_counts_am = df['amount_paid'].isnull().sum()
print(null_counts_am)
df.info()

```
Output: all null values replaced

![image](https://github.com/user-attachments/assets/99189c50-8c15-4a25-8153-b99e5b8ecf81)


**5. Location:**

```python
# check for unique values
unique_location = df['location'].unique()
print(unique_location)

# check for null values
null_counts_location = df['location'].isnull().sum()
print(null_counts_location)


# Grouping by location to observe
location = df.groupby('location')['location'].count()
```
Output: Location names are capitalized and i need to normalize it.

![image](https://github.com/user-attachments/assets/6ce26ad1-bf06-4f8c-bdc1-f75d41fadf4f)


```python
# Capitalized and titled
df['location'] = df['location'].str.capitalize()
df['location'] = df['location'].str.title()
```
Output: Location capitalized 

![image](https://github.com/user-attachments/assets/6c04a919-3b17-4f1a-942b-cf484ff55fad)


**6. Individuals on claim:**
Checking for null values and column description.
```python
# check for null values
individuals_on_claim = df['individuals_on_claim'].isnull().sum()
print(individuals_on_claim)

#view claim_amount description
df['individuals_on_claim'].describe()

```
Output: No null values, 15 unique values and no cleaning needed.

![image](https://github.com/user-attachments/assets/bfffe055-d6ed-4dfc-88e6-54d429796ddd)



**7. Linked_cases**
The linked cases column is supposed to be 2 unique values True and False.
```python
# check the unique linked_case 
unique_linked_cases = df['linked_cases'].unique()
print(unique_linked_cases)

# count the null values in linked_cases
null_counts_linked_cases = df['linked_cases'].isnull().sum()
print(null_counts_linked_cases)
```

Output: Linked cases has 26 missing values.

![image](https://github.com/user-attachments/assets/5bde0dae-53c8-47a7-ab04-bf3e8b304f10)

Since there are 26 missing values, I replaced them with 'False'
```python
# Fill NA/Null values with False
df['linked_cases'].fillna(False, inplace=True)

# Desclribe 
df['linked_cases'].describe()
```
Output: Missing values replaced 

![image](https://github.com/user-attachments/assets/4716ce03-66e9-4b50-8961-662a43eeb2fa)


**8. cause**

```python
# 8 cause

# check unique cause
unique_cause = df['cause'].unique()
print(unique_cause)

# check null values
null_counts_cause = df['cause'].isnull().sum()
print(null_counts_cause)
```

Output:

![image](https://github.com/user-attachments/assets/13edfbb1-bf55-4792-8b66-c417484147e0)

The cause column is meant to be 3 unique values, next step is to clean up by  renaming meat and vegetable to merge them, and also capitalize it.

```python
# Group cause categories
group = df.groupby('cause')['cause'].count()
print(group)
```
Output:

![image](https://github.com/user-attachments/assets/b2491844-8d98-42fc-b0e4-a48ceb28bd10)


Replacing Meat with meat and VEGETABLES

```python
# Replacing Meat with meat and VEGETABLES
df['cause'] = df['cause'].replace({' Meat': 'meat', 'VEGETABLES': 'vegetable'})

# Group cause categories to check value replacement
group = df.groupby('cause')['cause'].count()
print(group)

# Capitalize cause
df['cause'] = df['cause'].str.capitalize()
```
Output: Cause cleaned and capitalised.

![image](https://github.com/user-attachments/assets/20081c63-9418-48e6-82d6-db8ee938a5e6)


**View cleaned data head**
```python

# View cleaned data head
df.head()
```

![image](https://github.com/user-attachments/assets/e6775165-9706-49a1-86a6-6ddd3a0a4673)





## Data cleaning and validation summary report

**Claim ID:** 2000 unique values match the description given. There are no missing values. No changes were made to this column.

**Time to close:** The values of this column range from 76 to 518. There were no missing values. No changes were made to this column.

**Claim amount:** The values of this column were all rounded to 2 decimal places, and the values range from 1637.94 to 76106.80. There were no missing values, rather 'R$' Brazilian currency logo was removed, and the data type was converted to float.

**Amount paid:** The values of this column were all rounded to 2 decimal places ranging from 1516.72 to 52498.75, which is consistent with the description given. 36 values were missing. The missing values were replaced with the median value of the remaining data, which was 20105.69.

**Location:** This column had four unique categories,'RECIFE', 'FORTALEZA', 'SAO LUIS', 'NATAL', that match those in the description. There were no missing values, and were also capitalized and used the string title method for the column.

**Individual on claim:** The values of this column range from 1 to 15, and match those in the description. There were no missing values and no changes were made to this column.

**Linked cases:** The values in this column were either True, False or missing. There were 26 missing values. All missing values were replaced with FALSE.

**Cause:** This column had three unique categories, 'meat','unknown', and 'vegetable', that match those in the description. But had twe additional categories that doesn't match the description, ' Meat' and 'VEGETABLES', both were renamed to match the description, and also capitalized the column.

**The dataset had 2000 rows and 8 columns before cleaning, and after cleaning and validation, the dataset remains 2000 rows and 8 columns.**



## Exploratory Data Analysis to Business Questions
I will be creating visualizations to answer **the Business Questions**



### 1. How many customers are there in each location?

```python
# Number of Claims in each location
value_counts = df['location'].value_counts()

 # Adjust the figure size 
plt.figure(figsize=(10, 6)) 

# Create the bar chart
plt.bar(value_counts.index, value_counts.values)

# Set the color palette and linestyle for the grid lines
sns.set_palette(['dimgray'])
sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '-', 'grid.color': '#F0F0F0'})

# Label each bar with their respective numbers
for i, value in enumerate(value_counts.values):
    plt.text(i, value, str(value), ha='center', va='bottom')

# Customize the chart
plt.xlabel('Location')
plt.ylabel('Count / Number of Claims')
plt.title('Number of Claims in each location')

# Display the chart
plt.show()
```
**Output:**

![image](https://github.com/user-attachments/assets/256af9fe-aea4-4f22-9a0f-4f581206a3c7)

### Findings:

There are four categories of location included in this data. The location with the most number of claims is Recife, with Sao Luis being second although with 25% of the number of the overall claim. 

The categories are unbalanced, with most observations being either Recife or Sao Luis. The legal team should focus on Recife and Sao Luis as they are the location with most claims. Recife: 44.25%, Sao Luis: 25.85%, Fortaleza: 15.55%, Natal: 14.35%.



### 2. Is there any relationship between Locations and the time it takes to close claims?

To find the relationship between Locations and the time I have to create a visualization that describes the distribution of time to close for all claims.


```python
# graph size
plt.figure(figsize=(10, 6))

# Create a histogram with time_to_close
sns.histplot(df['time_to_close'], bins=30)

# Set the color palette and linestyle for the grid lines
sns.set_palette(['dimgray'])
sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '-', 'grid.color': '#F0F0F0'})

# Customize the chart
plt.ylabel('Count')
plt.xlabel('Time to close')
plt.title('Distribution of Time to Close')

# Display chart
plt.show()
```
**Output: Distribution of Time to Close**

![image](https://github.com/user-attachments/assets/c1a5cafe-bf92-4fe0-9c09-293fbee91bb3)



### Findings:

The graph above exhibits a predominantly symmetrical, with more data clusters around the peak. distribution, indicating a normal distribution pattern. However, when specifically examining the time it takes to close claims, the distribution is right-skews. While outliers exist, indicating rare instances of claims taking over 300 days to close, the majority of claims fall within the range of 173 to 188 days.

Based on this insightful analysis, the legal team is now equipped to establish targeted objectives and performance metrics aimed at enhancing the closure time of claims. By considering the typical timeframe within which the majority of claims are resolved, the team can set realistic goals and develop strategies to streamline and expedite the claims closure process.



### 3. How does the time to close claims look like?

Here I created a visualization that shows the relationship between time to close and location


```python
# Adjust the figure size as needed
plt.figure(figsize=(9, 6))  

# Create a box plot to visualize the distribution of 'time_to_close' over 'location'
sns.boxplot(data=df, x='location', y='time_to_close', color='gray')
plt.xlabel('Location')
plt.ylabel('Time to close')
plt.title('Range in number of Time to close by Location')

# Display the plot
plt.show();
```
**Distribution of time to close over individual location**

![image](https://github.com/user-attachments/assets/ed7ded04-9fdf-4da6-b379-74df13bc31e1)


### Findings:

We can examine two variables to explore the impact of office location on claim closure time. This analysis provides valuable insights for the legal team regarding the distribution of observations across different office locations. 

From the graph, it is evident that all office locations exhibit similar interquartile ranges. The mean slightly surpasses the median for each location, indicating the presence of outliers. However, both the mean and median values demonstrate consistency, ranging from 178 to 180. 

Notably, the Recife office location stands out with the highest number of outliers, representing cases that took more than 300 to 400 days to close. Similarly, Sao Luis exhibits a few cases exceeding 400 days for closure, alongside outlier cases taking over 300 days. Fortaleza and Natal, on the other hand, have a lower number of outliers.


### Recommendation

**Given these observations, the legal team can prioritize Recife and Sao Luis office locations when formulating solutions to enhance the claims closure process.**

**Additionally, further investigations can be conducted to determine any potential relationships between variables such as claim amount, individuals involved in the claim, and the duration of the claims closure process.**



Thank you for taking out time to read through this project, kindly drop comments on your thoughts, suggestions etc.

**By Michael Olaniyi Jeremiah**
