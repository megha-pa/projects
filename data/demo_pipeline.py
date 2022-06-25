import matplotlib.pyplot as plt
import pandas as pd

from kfp.v2 import compiler, dsl
from kfp.v2.dsl import (
    pipeline,
    component,
    Artifact,
    Dataset,
    Input,
    Metrics,
    Model,
    Output,
    InputPath,
    OutputPath,
)

from google.cloud import aiplatform

# We'll use this namespace for metadata querying
from google.cloud import aiplatform_v1

PROJECT_ID = "pod-devops"
BUCKET_NAME = "gs://welspun-data"
REGION = "us-central1"
PIPELINE_ROOT = f"{BUCKET_NAME}/vertex_ai_pipeline_artifacts/pipeline_root"


@component(
    packages_to_install=["google-cloud-bigquery", "pandas", "db-dtypes"],
    base_image="python:3.9",
)
def read_data_from_bq(
    req_query: str, PROJECT_ID: str, output_bq_df_file_path: Output[Dataset]
):
    try:
        import pandas as pd
        from google.cloud import bigquery
        import logging

        bq_client = bigquery.Client(project=PROJECT_ID)
        df = bq_client.query(req_query).to_dataframe()
        df.to_csv(output_bq_df_file_path.path, index=False)
    except Exception as e:
        logging.error(f"error read_data_component:::: error message {e}")


@component(
    packages_to_install=["numpy", "pandas", "google-cloud-bigquery==2.26.0"],
    base_image="python:3.9",
)
def preprocess_data(
    input_file_path_from_bq: InputPath("Dataset"),
    output_bq_df_file_path: Output[Dataset],
):
    try:
        import pandas as pd
        from datetime import date, timedelta
        import logging

        DATE_COL = "Order_Date"
        ID_COL = "Product_ID"
        TARGET_COL = "Sales"
        FORECAST_HORIZON = 18
        df = pd.read_csv(input_file_path_from_bq)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        monthly_aggrigated_ecommerce_data = pd.DataFrame()
        for article in df[ID_COL].unique():
            sub_df = df[(df[ID_COL] == article)]
            end_date_last_day = sub_df[DATE_COL].max().replace(day=1) - timedelta(
                days=1
            )
            end_date_last_day = pd.to_datetime(end_date_last_day)
            sub_df = sub_df[(sub_df[DATE_COL] <= end_date_last_day)]
            if not (sub_df.empty):
                sub_df_copy = sub_df.copy()
                # add a record to fill 0s for discontinued product untill last month
                if sub_df[DATE_COL].max() < end_date_last_day:
                    record_to_insert = {
                        ID_COL: article,
                        DATE_COL: end_date_last_day,
                        TARGET_COL: 0,
                    }  # populate a dummy record of last month so that resampling function handles filling 0s for discontinued products
                    sub_df = sub_df.append(record_to_insert, ignore_index=True)
                sub_df[DATE_COL] = pd.to_datetime(sub_df[DATE_COL])
                sub_df.index = sub_df[DATE_COL]
                sub_df = sub_df.resample(rule="MS").agg({"Sales": "sum"})
                sub_df = sub_df.reset_index()
                sub_df[ID_COL] = article
                monthly_aggrigated_ecommerce_data = (
                    monthly_aggrigated_ecommerce_data.append(sub_df, ignore_index=True)
                )

                monthly_aggrigated_ecommerce_data.to_csv(
                    output_bq_df_file_path.path, index=False
                )
    except Exception as e:
        logging.error(f"error preprocess data:::: error message {e}")


@component(
    packages_to_install=["numpy", "pandas", "google-cloud-bigquery==2.26.0"],
    base_image="python:3.9",
)
def split_timeseries(
    input_file_path_from_bq: InputPath("Dataset"),
    train_df_path: Output[Dataset],
    validation_df_path: Output[Dataset],
):
    try:
        import logging
        import pandas as pd
        from datetime import date, timedelta

        DATE_COL = "Order_Date"
        ID_COL = "Product_ID"
        TARGET_COL = "Sales"
        monthly_aggrigated_ecommerce_data = pd.read_csv(input_file_path_from_bq)
        FORECAST_HORIZON = 18

        def split_time_series_for_modeling(df, evaluation_size):
            import pandas as pd
            import numpy as np

            train_size = len(df) - evaluation_size
            train, eval = df.iloc[:train_size], df.iloc[train_size:]
            return train, eval

        train_df1 = pd.DataFrame()
        eval_df1 = pd.DataFrame()
        for article in monthly_aggrigated_ecommerce_data[ID_COL].unique():
            single_article_df = monthly_aggrigated_ecommerce_data[
                monthly_aggrigated_ecommerce_data[ID_COL] == article
            ]
            if not (single_article_df.empty):
                train_df, evaluation_df = split_time_series_for_modeling(
                    single_article_df, evaluation_size=6
                )  # Need to applied in component 2 and only train, test dfs need to be there
                train_df1 = train_df1.append(train_df)
                eval_df1 = eval_df1.append(evaluation_df)
        train_df1.to_csv(train_df_path.path, index=False)
        eval_df1.to_csv(validation_df_path.path, index=False)
    except Exception as e:
        logging.error(f"ERROR : split_training :::: error message {e}")


@component(
    packages_to_install=[
        "numpy",
        "pandas",
        "google-cloud-bigquery==2.26.0",
        "statsmodels==0.12.2",
        "scikit-learn==0.24.2",
        "pmdarima==1.8.2",
        "pyarrow",
    ],
    base_image="python:3.9",
)
def Model_training(
    input_file_path_from_bq: InputPath("Dataset"),
    train_df_path: InputPath("Dataset"),
    validation_df_path: InputPath("Dataset"),
    result_df_path: Output[Dataset],
):
    try:
        import logging
        import pandas as pd
        from datetime import date, timedelta
        import numpy as np
        from statsmodels.tsa.ar_model import AutoReg
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_percentage_error,
            mean_absolute_error,
        )
        from google.cloud import bigquery

        DATE_COL = "Order_Date"
        ID_COL = "Product_ID"
        TARGET_COL = "Sales"
        # df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        FORECAST_HORIZON = 18
        PROJECT_ID = "pod-devops"

        # df[DATE_COL] = pd.to_datetime(df[DATE_COL])

        monthly_aggrigated_ecommerce_data = pd.read_csv(input_file_path_from_bq)
        train_df1 = pd.read_csv(train_df_path)
        eval_df1 = pd.read_csv(validation_df_path)

        bigquery_result_table_name = "pod-devops.mlops_data.forecast_result"

        def ceil_list_of_values(values):
            return np.ceil(values)

        def generate_dates(last_date, horizon):
            print("generate dates started!!!")
            dates_list = list(
                pd.date_range(start=last_date, periods=horizon + 1, freq="M")
            )
            print("dates SUCESSFUL!!!")
            return dates_list[1:]

        def generate_algorithm_results_df(
            eval_df, model_predictions, model_forecasts, algorithm_name
        ):
            FORECAST_HORIZON = 18
            print("algorithem started!!!")
            eval_df[TARGET_COL + "_PREDICTED"] = model_predictions
            last_date = eval_df[DATE_COL].values.tolist()[-1]  # last date
            projected_dates = generate_dates(last_date, FORECAST_HORIZON)
            forecast_df = pd.DataFrame()
            forecast_df[DATE_COL] = projected_dates
            forecast_df[ID_COL] = eval_df[ID_COL].values[0]
            # forecast_df["CUSTOMER_NAME"] = eval_df["CUSTOMER_NAME_"].values[0] #comment for primary and secondary retail
            forecast_df[TARGET_COL + "_Forecasted"] = np.ceil(model_forecasts)
            final_outcome_df = pd.concat([eval_df, forecast_df], axis=0)
            final_outcome_df["MODEL"] = algorithm_name
            print("algorithem SUCESSFUL!!!")
            return final_outcome_df

        def ma_train_with_optimal_parameters(best_window, article_df):
            series = article_df[TARGET_COL].values.tolist()
            preds = []  # to append predictions
            for t in range(FORECAST_HORIZON):
                length = len(series)
                forecast = np.mean(
                    [series[i] for i in range(length - best_window, length)]
                )

                # append the forecast to the history and predictions
                series.append(forecast)
                preds.append(forecast)
            return preds

        def MAMODEL(train, evaluation, article_df):
            best_rmse = float("inf")
            best_window = None
            evaluation_predictions = None
            test_data = None
            data_length = len(article_df[TARGET_COL].values.tolist())
            windows = [x for x in range(2, data_length // 2)]

            # iterate through windows
            for window in windows:
                # history and test series
                series = article_df[TARGET_COL].values.tolist()
                history = [series[i] for i in range(window)]
                test = [series[i] for i in range(window, len(series))]

                predictions = []  # predictions

                # walk forward over time steps in test data
                for t in range(len(test)):
                    length = len(history)
                    yhat = np.mean([history[i] for i in range(length - window, length)])
                    obs = test[t]

                    predictions.append(yhat)
                    history.append(obs)

                # error rate
                error_rate = np.sqrt(mean_squared_error(test, predictions))

                # best parameters
                if best_rmse >= error_rate:
                    best_rmse = error_rate
                    evaluation_predictions = predictions
                    best_window = window
                    test_data = test

            forecast_predictions = ma_train_with_optimal_parameters(
                best_window, article_df
            )
            evaluation_predictions = ceil_list_of_values(
                evaluation_predictions[-len(evaluation) :]
            )
            forecast_predictions = ceil_list_of_values(forecast_predictions)
            return evaluation_predictions, forecast_predictions

        def write_dataframe_to_bigquery_table(df, bigquery_full_table_name):
            client = bigquery.Client(project=PROJECT_ID)
            job = client.load_table_from_dataframe(df, bigquery_full_table_name)
            job.result()

        algorithem_df = pd.DataFrame()
        for article in monthly_aggrigated_ecommerce_data[ID_COL].unique():
            single_article_training_df = train_df1[train_df1[ID_COL] == article]
            single_article_evaluation_df = eval_df1[eval_df1[ID_COL] == article]
            try:

                article_df = single_article_training_df.append(
                    single_article_evaluation_df, ignore_index=True
                )
                model_evaluation_predictions, model_forecast_predictions = MAMODEL(
                    single_article_training_df, single_article_evaluation_df, article_df
                )
                # print("fg")
                algorithm_results = generate_algorithm_results_df(
                    single_article_evaluation_df,
                    model_evaluation_predictions,
                    model_forecast_predictions,
                    "AR",
                )
                algorithem_df = algorithem_df.append(
                    algorithm_results, ignore_index=True
                )
                # write_dataframe_to_bigquery_table(algorithm_results, bigquery_result_table_name)

                # algorithem_df.to_csv("algo_df.csv",index = False)

            except Exception as e:
                print(e)
                continue

        algorithem_df.to_csv(result_df_path.path, index=False)
    except Exception as e:
        logging.error(f"ERROR : model_training :::: error message {e}")


@component(
    packages_to_install=[
        "google-cloud-bigquery==2.26.0",
        "pandas==1.3.0",
        "gcsfs==2021.6.1",
        "numpy==1.19.5",
        "pyarrow==4.0.1",
        "statsmodels==0.12.2",
        "scikit-learn==0.24.2",
        "pmdarima==1.8.2",
        "google-cloud-aiplatform==1.3.0",
    ],
    base_image="python:3.9",
)
def evaluation(
    final_dataset: InputPath("Dataset"), evaluation_df_path: Output[Dataset]
):
    try:
        import logging
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_percentage_error,
            mean_absolute_error,
        )
        import numpy as np
        import pandas as pd
        from google.cloud import storage
        from google.cloud import bigquery

        bigquery_final_table_name = "pod-devops.mlops_data.evaluation_result"
        bigquery_result_table_name = "pod-devops.mlops_data.forecast_result"
        PROJECT_ID = "pod-devops"
        DATE_COL = "Order_Date"
        ID_COL = "Product_ID"
        TARGET_COL = "Sales"
        # df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        FORECAST_HORIZON = 18
        predicted_col = "Sales_PREDICTED"

        modeling_df = pd.read_csv(final_dataset)
        modeling_df = modeling_df.fillna(0)

        def evaluation_metrics(single_article_results, article):
            evaluation_df = pd.DataFrame()
            rmse = np.sqrt(
                mean_squared_error(
                    single_article_results[TARGET_COL].to_list(),
                    single_article_results[predicted_col].to_list(),
                )
            )
            mae = mean_absolute_error(
                single_article_results[TARGET_COL].to_list(),
                single_article_results[predicted_col].to_list(),
            )
            mape = mean_absolute_percentage_error(
                single_article_results[TARGET_COL].to_list(),
                single_article_results[predicted_col].to_list(),
            )
            mse = mean_squared_error(
                single_article_results[TARGET_COL].to_list(),
                single_article_results[predicted_col].to_list(),
            )
            evaluation_df = evaluation_df.append(
                {"ID_COL": article, "rmse": rmse, "mae": mae, "mape": mape, "mse": mse},
                ignore_index=True,
            )
            print("evaluation_completed")
            write_dataframe_to_bigquery_table(evaluation_df, bigquery_final_table_name)

            evaluation_df.to_csv(evaluation_df_path.path, index=False)

        def write_dataframe_to_bigquery_table(df, bigquery_full_table_name):
            print("write started")
            client = bigquery.Client(project=PROJECT_ID)
            job = client.load_table_from_dataframe(df, bigquery_full_table_name)
            job.result()

        write_dataframe_to_bigquery_table(modeling_df, bigquery_result_table_name)

        for article in modeling_df[ID_COL].unique():
            single_article_df = modeling_df[modeling_df[ID_COL] == article]
            article = article
            evaluation_metrics(single_article_df, article)

    except Exception as e:
        logging.error(f"ERROR evaluation :::: error message {e}")


PROJECT_ID = "pod-devops"
query_string = f"""SELECT Order_Date,Product_ID,Sales FROM `{PROJECT_ID}.mlops_data.data_final`
where Product_ID IN("OFF-AR-10001166","OFF-PA-10004022","OFF-BI-10001525","FUR-CH-10000988","TEC-PH-10003885") """


@pipeline(
    pipeline_root=PIPELINE_ROOT,
    name="pipeline-trial",
)
def my_pipeline(req_query: str = "", PROJECT_ID: str = ""):

    read_data_from_bq_task = read_data_from_bq(req_query, PROJECT_ID)

    preprocess_data_task = preprocess_data(
        read_data_from_bq_task.outputs["output_bq_df_file_path"]
    )

    split_timeseries_task = split_timeseries(
        preprocess_data_task.outputs["output_bq_df_file_path"]
    )

    Model_training_task = Model_training(
        preprocess_data_task.outputs["output_bq_df_file_path"],
        split_timeseries_task.outputs["train_df_path"],
        split_timeseries_task.outputs["validation_df_path"],
    )

    evaluation_task = evaluation(Model_training_task.outputs["result_df_path"])


def run_pipeline():
    print("pipeline")
    compiler.Compiler().compile(
        pipeline_func=my_pipeline, package_path="azdemo-pipeline.json"
    )


if __name__ == "__main__":
    run_pipeline()

