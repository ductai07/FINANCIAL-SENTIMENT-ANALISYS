import pandas as pd
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("C:/Users/ASUS/Desktop/Financial Sentiment/all-data.csv",encoding="ISO-8859-1")
header = ['label','content']
df.columns = header
df.head(10)

classes = df['label'].unique()
classes = {class_name : idx for idx,class_name in enumerate(classes.tolist()) }
df['label']=df['label'].map(classes)
df.head(10)


df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=42)



 

