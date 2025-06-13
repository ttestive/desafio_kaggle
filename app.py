import numpy as numpy
import pandas as pd 
import os 

import tensorflow as tf
import tensorflow_decision_forests as tfdf


print(f"Found TF-DF {tfdf.__version__}")

train_data = pd.read_csv("assets/train (1).csv")
test_data = pd.read_csv("assets/test (1).csv")


def process_data(df):
    df = df.copy()

    def normalize(x):
        return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])

    def ticket_number(x):
        return x.split(" ")[-1]

    def ticket_item(x):
        items = x.split(" ")
        if len(items) == 1:
            return "NONE"
        return "_".join(items[0:-1])
    

    df["Name"] = df["Name"].apply(normalize)
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)


preprocessed_train = process_data(test_data)
preprocessed_serving = process_data(train_data)

print(preprocessed_serving)

