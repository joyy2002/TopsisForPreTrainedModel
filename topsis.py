import pandas as pd
import numpy as np

def normalize_data(df):
    squared_sums = np.zeros(df.shape[1])

    for column in df.columns:
        squared_sums[df.columns.get_loc(column)] = np.sum(df[column] ** 2)

    normalized_df = df.copy()

    for column in df.columns:
        col_loc = df.columns.get_loc(column)
        normalized_df[column] = df[column] / np.sqrt(squared_sums[col_loc] if squared_sums[col_loc] != 0 else 1e-10)

    return normalized_df

def multiply_weights(df, weights):
    df_array = df.values
    weights_array = np.array(weights, dtype=np.float64)
    weights_array = weights_array / sum(weights_array)

    result = df_array * weights_array
    result_df = pd.DataFrame(result, columns=df.columns, index=df.index)

    return result_df

def SplusSminus(df, impacts):
    num_columns = df.shape[1]
    Splus = np.zeros(num_columns)
    Sminus = np.zeros(num_columns)

    for i in range(num_columns):
        column_values = df.iloc[:, i]
        if impacts[i] == '-':
            Splus[i] = max(column_values)
            Sminus[i] = min(column_values)
        elif impacts[i] == '+':
            Splus[i] = min(column_values)
            Sminus[i] = max(column_values)

    return Splus, Sminus

def PerformanceScore(df, Splus, Sminus):
    EDPlus = ((df - Splus) ** 2).sum(axis=1).apply(np.sqrt)
    EDMinus = ((df - Sminus) ** 2).sum(axis=1).apply(np.sqrt)

    df['Topsis Score'] = EDPlus / (EDPlus + EDMinus)
    df['Rank'] = df['Topsis Score'].rank(ascending=False)

    return df

# Given data
df = pd.read_csv('data.csv')



# Drop the 'Model' column
df_models=df.iloc[:,0]
df = df.drop(columns='Model')

# Step 1: Normalize the data
df_normalized = normalize_data(df)
print("Step 1: Normalized Data")
print(df_normalized)

# Step 2: Multiply by weights
weights = [1, 1]
df_weighted = multiply_weights(df_normalized, weights)
print("\nStep 2: Weighted Data")
print(df_weighted)

# Step 3: Determine Splus and Sminus
impacts = ['-', '-']
Splus, Sminus = SplusSminus(df_weighted, impacts)
print("\nStep 3: Splus and Sminus")
print("Splus:", Splus)
print("Sminus:", Sminus)

# Step 4: Calculate the Topsis Score and Rank
df_result = PerformanceScore(df_weighted, Splus, Sminus)
print("\nStep 4: Topsis Score and Rank")
print(df_result[['Topsis Score', 'Rank']])

# Step 5: Store result in csv file
result=pd.concat([df_models,df_result[['Topsis Score', 'Rank']]],axis=1)
result.to_csv('result.csv',index=False)
