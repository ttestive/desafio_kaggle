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
    return df


preprocessed_train = process_data(test_data)
preprocessed_serving = process_data(train_data)

print(preprocessed_serving.columns)

input_features = list(preprocessed_train.columns)
input_features

input_features.remove("Ticket")
input_features.remove("PassengerId")

print(f"Input features: {input_features}")

print(preprocessed_train.columns)

def convert_pandas_to_tensor(features, labels = None):
    features_name = tf.strings.split(features["Name"])
    return features, labels

train = tfdf.keras.pd_dataframe_to_tf_dataset(preprocessed_serving, label="Survived").map(convert_pandas_to_tensor)
serving = tfdf.keras.pd_dataframe_to_tf_dataset(preprocessed_train).map(convert_pandas_to_tensor)



model = tfdf.keras.GradientBoostedTreesModel(
    verbose=0, # Very few logs
    features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
    exclude_non_specified_features=True, # Only use the features in "features"
    random_seed=1234,
)
model.fit(train)

self_evaluation = model.make_inspector().evaluation()
print(f"Accuracy: {self_evaluation.accuracy} Loss:{self_evaluation.loss}")


model = tfdf.keras.GradientBoostedTreesModel(
    verbose=0, # Very few logs
    features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
    exclude_non_specified_features=True,

    min_examples=1,
    categorical_algorithm="RANDOM",
    #max_depth=4,
    shrinkage=0.05,
    #num_candidate_attributes_ratio=0.2,
    split_axis="SPARSE_OBLIQUE",
    sparse_oblique_normalization="MIN_MAX",
    sparse_oblique_num_projections_exponent=2.0,
    num_trees=2000,
    #validation_ratio=0.0,
    random_seed=1234,
    
)

model.fit(train)


self_evaluation = model.make_inspector().evaluation()
print(f"Accuracy: {self_evaluation.accuracy} Loss:{self_evaluation.loss}")


model.summary()


def prediction_to_kaggle_format(model, threshold=0.5):
    proba_survive = model.predict(serving, verbose=0)[:,0]
    return pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": (proba_survive >= threshold).astype(int)
    })

def make_submission(kaggle_predictions):
    path="Gold/submission.csv"
    kaggle_predictions.to_csv(path, index=False)
    print(f"Submission exported to {path}")
    
kaggle_predictions = prediction_to_kaggle_format(model)
make_submission(kaggle_predictions)