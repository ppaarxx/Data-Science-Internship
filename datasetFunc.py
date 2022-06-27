import pandas as pd
df=pd.read_csv("C:/Users/parth puri/PycharmProjects/datascience/dataset1/IRIS.csv")
# print(df)
print(df.head(10)) # prints the values from 1-10
print(df.tail(20))  # prints the values from 20-end
print(df.head(1))

print(df.columns.values)
# sepal_length sepal_width petal_length petal_width species

print(df.describe())
'''OUTPUT of describe
        sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
'''

print(df[df['petal_width']>1.0])
print(df.loc[:,["petal_length","species"]])

print (df[df['species']=='Iris-setosa'])
print(df[45:74])