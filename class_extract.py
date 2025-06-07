import pandas as pd
df = pd.read_csv("C:/Users/bpmch/OneDrive/Desktop/python/pytorch/monkey_classification/10_Monkey_Species/monkey_labels.txt")
df = df[["Label"," Common Name                   "]]
df.rename(columns={" Common Name                   ":"Column Name"},inplace=True)
df["Label"] = df["Label"].str.replace(" ","")
df["Column Name"] = df["Column Name"].str.replace(" ","").str.replace("_"," ").str.title()
print(len(df))
dictionary = {i: (df.loc[i, "Label"],df.loc[i, "Column Name"]) for i in range(len(df))}
print(dictionary[0][1])