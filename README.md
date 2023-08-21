# Food Claim Exploratory Analysis in Python

![Food_claim](https://github.com/mikeolaniyi/Food-Claim-Exploratory-Analysis-on-Python/assets/120651356/ec38440d-1e7c-4cf9-a61d-487542ddfb43)

# Background
Vivendo is a fast food chain in Brazil with over 200 outlets. Customers often claim compensation from the company for food poisoning. The legal team processes these claims. The legal team has offices in four locations. The legal team wants to improve how long it takes to reply to customers and close claims. The head of the legal department wants a report on how each location differs in the time it takes to close claims.


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

# Check all info and descrption
df.head()
df.info()
df.describe()
df.boxplot()

# Check for null values
null_counts = df.isnull().sum()
print(null_counts)
```

# - Two fields found with null values.


# Data Validation: Let's start cleaning and validating all the fields one by one.

```python

# 1. Claim_ID

# checking if there's any duplicates
has_duplicates = df['claim_id'].duplicated().any()
print(has_duplicates)

# # checking for numebr of duplicates
df['claim_id'].duplicated().sum() 

# checking for number of unique ID
len(df['claim_id'].unique()) 

# - Unique 2000 records.


# 2. Time_to_close
df['time_to_close'].info()

# checking Missing values
df['time_to_close'].isnull().sum()
df['time_to_close'].describe()

check = df['time_to_close'] >= 400
filtered_data = df[check]

# Print the filtered data
print(filtered_data)

# - No missing values


# 3. Claim_amount

# Removing R$ from 'claim_amount'
df['claim_amount'] = df['claim_amount'].str.replace('R\$ ', '')

# Convert the claim_amount to a float data type
df['claim_amount'] = df['claim_amount'].astype(float)

# check the data info
df['claim_amount'].info

# - Claim cleaned and converted to float datatype



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

# Replace null values with amount median value
median_amount_paid = df['amount_paid'].median()
df['amount_paid'].fillna(median_amount_paid, inplace=True)

# chech for nulls
null_counts_am = df['amount_paid'].isnull().sum()
print(null_counts_am)
df.info()



# 5 Location

# check for unique values
unique_location = df['location'].unique()
print(unique_location)

# check for null values
null_counts_location = df['location'].isnull().sum()
print(null_counts_location)

# Capitalized and titled
df['location'] = df['location'].str.capitalize()
df['location'] = df['location'].str.title()



# 6 Individuals on claim

# check for null values
individuals_on_claim = df['individuals_on_claim'].isnull().sum()
print(individuals_on_claim)

#view claim_amount description
df['individuals_on_claim'].describe()



# 7 Location

# check the unique linked_case 
unique_linked_cases = df['linked_cases'].unique()
print(unique_linked_cases)

# count the null values in linked_cases
null_counts_linked_cases = df['linked_cases'].isnull().sum()
print(null_counts_linked_cases)

# Fill NA/Null values with False
df['linked_cases'].fillna(False, inplace=True)

# Desclribe 
df['linked_cases'].describe()
```


```python

# 8 cause


# check unique cause
unique_cause = df['cause'].unique()
print(unique_cause)

# check null values
null_counts_cause = df['cause'].isnull().sum()
print(null_counts_cause)

# Group cause gategories
group = df.groupby('cause')['cause'].count()
print(group)

# Replacing Meat with meat and VEGETABLES
df['cause'] = df['cause'].replace({' Meat': 'meat', 'VEGETABLES': 'vegetable'})

# Check cause again
cause_count = df.groupby('cause')['cause'].count()
print(cause_count)

# capitalize
df['cause'] = df['cause'].str.capitalize()

# Let's check the head of the cleaned dataset
df.head()
```

Cleaned data head:
![Food_claim_data_head](https://github.com/mikeolaniyi/Food-Claim-Exploratory-Analysis-on-Python/assets/120651356/42ed35bc-0667-47a4-912d-cefc1360454b)



# Task 2
# Create a visualization that shows the number of claims in each location. Use the visualization to:

# State which category of the variable location has the most observations. Explain whether the observations are balanced across categories of the variable location


```python
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
![1  Number of Claims in each location](https://github.com/mikeolaniyi/Food-Claim-Exploratory-Analysis-on-Python/assets/120651356/352920f5-1b5d-40e2-8829-8d22c22f3ed0)


# Findings:
**There are four categories of location included in this data. The location with the most number of claims is Recife, with Sao Luis being second although with 25% of the number of the overall calim. The categories are unbalanced, with most observations being either Recife or Sao Luis. The legal team should focus on Recife and Sao Luis as they are the location with most claim. Recife: 42.92%, Sao Luis: 25.07%, Fortaleza: 15.04%, Natal: 13.97%.**



# Task 3
# Describe the distribution of time to close for all claims. Your answer must include a visualization that shows the distribution.


```python
# give the graph a size
plt.figure(figsize=(10, 6))

#Create an histogram with time_to_close
sns.histplot(df['time_to_close'], bins=30)


# Set the color palette and linestyle for the grid lines
sns.set_palette(['dimgray'])
sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '-', 'grid.color': '#F0F0F0'})

#Customize the chart
plt.ylabel('Count')
plt.xlabel('Time to close')
plt.title('Distribution of Time to Close')

#Display chart
plt.show()
```
![2  Distribution of Time to Close](https://github.com/mikeolaniyi/Food-Claim-Exploratory-Analysis-on-Python/assets/120651356/b8604bf6-1b49-49d3-a333-91c1cff09e41)


# Findings:
**The graph below exhibits a predominantly symmetrical, with more data clusters around the peak. distribution, indicating a normal distribution pattern. However, when specifically examining the time it takes to close claims, the distribution is right-skews. While outliers exist, indicating rare instances of claims taking over 300 days to close, the majority of claims fall within the range of 173 to 188 days.**

**Based on this insightful analysis, the legal team is now equipped to establish targeted objectives and performance metrics aimed at enhancing the closure time of claims. By considering the typical timeframe within which the majority of claims are resolved, the team can set realistic goals and develop strategies to streamline and expedite the claims closure process.**




# Task 4
# Describe the relationship between time to close and location. Your answer must include a visualization to demonstrate the relationship.



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



![3 Range in number of Time to close by Location](https://github.com/mikeolaniyi/Food-Claim-Exploratory-Analysis-on-Python/assets/120651356/ece8aeca-5a76-4f6a-812b-ca8553434783)




# Heatmap 
sns.heatmap(df.corr(),annot=True);
plt.title('The relationship between the values');
```
![Hitmap- Relationship between values](https://github.com/mikeolaniyi/Food-Claim-Exploratory-Analysis-on-Python/assets/120651356/33135678-92a9-4112-a070-e8240c819b91)


# Findings:
**We can examine two variables to explore the impact of office location on claim closure time. This analysis provides valuable insights for the legal team regarding the distribution of observations across different office locations.**
**From the graph, it is evident that all office locations exhibit similar interquartile ranges. The mean slightly surpasses the median for each location, indicating the presence of outliers. However, both the mean and median values demonstrate consistency, ranging from 178 to 180.**
**Notably, the Recife office location stands out with the highest number of outliers, representing cases that took more than 300 to 400 days to close. Similarly, Sao Luis exhibits a few cases exceeding 400 days for closure, alongside outlier cases taking over 300 days. Fortaleza and Natal, on the other hand, have a lower number of outliers**.



# Recommendation
# Given these observations, the legal team can prioritize Recife and Sao Luis office locations when formulating solutions to enhance the claims closure process. Additionally, further investigations can be conducted to determine any potential relationships between variables such as claim amount, individuals involved in the claim, and the duration of the claims closure process.

