from airflow.decorators import dag, task
from airflow.sdk.definitions.asset import Asset
from pendulum import datetime
import pandas as pd

@dag(
    start_date=datetime(2025, 6, 23),
    schedule="@daily",
    catchup=False,
    default_args={"owner": "Astro", "retries": 3},
    tags={"kaggle challenge"}
)
def execute_kaggle_challenge():
    
    @task(outlets=[Asset("current_challenge_dag")])
    def ler_train():
        df_train = pd.read_csv("assets/train (1).csv")
        return df_train.to_json()

    @task()
    def ler_test():
        df_test = pd.read_csv("assets/test (1).csv")
        return df_test.to_json()

    @task()
    def process_data(json_train, json_test):
        df_train = pd.read_json(json_train)
        df_test = pd.read_json(json_test)

        def normalize(x):
            return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])

        def ticket_number(x):
            return x.split(" ")[-1]

        def ticket_item(x):
            items = x.split(" ")
            return "NONE" if len(items) == 1 else "_".join(items[:-1])
        
        for df in [df_train, df_test]:
            df["Name"] = df["Name"].apply(normalize)
            df["Ticket_number"] = df["Ticket"].apply(ticket_number)
            df["Ticket_item"] = df["Ticket"].apply(ticket_item)
        
        return {
            "train": df_train.to_json(),
            "test": df_test.to_json()
        }

    @task()
    def treinar_modelo(processed_data):
        import tensorflow_decision_forests as tfdf
        import tensorflow as tf

        df_train = pd.read_json(processed_data["train"])
        dataset = tfdf.keras.pd_dataframe_to_tf_dataset(df_train, label="Survived")

        model = tfdf.keras.GradientBoostedTreesModel()
        model.fit(dataset)

        acc = model.make_inspector().evaluation().accuracy
        print(f"Accuracy: {acc}")
        return acc

    # Orquestração
    json_train = ler_train()
    json_test = ler_test()
    processed = process_data(json_train, json_test)
    treinar_modelo(processed)

dag_instance = execute_kaggle_challenge()
