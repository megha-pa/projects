PROJECT_ID = "pod-devops"
req_query=f"""SELECT Order_Date,Product_ID,Sales FROM `{PROJECT_ID}.mlops_data.data_final`
where Product_ID IN("OFF-AR-10001166","OFF-PA-10004022","OFF-BI-10001525","FUR-CH-10000988","TEC-PH-10003885") """


import datatest as dt
import pytest

FLOAT_TITLE_COLUMNS = ["Sales"]
#Test if target data type is float


EXPECTED_COLUMNS = ["Order_Date","Product_ID","Sales"]

@pytest.mark.mandatory

def test_float_title(df):
    fllen = len(FLOAT_TITLE_COLUMNS)
    for i in range(fllen):
        dt.validate(df[FLOAT_TITLE_COLUMNS[i]], float)

def test_columns(df):
    dt.validate(
       df.columns, EXPECTED_COLUMNS,)

def read_data_from_bq(req_query,PROJECT_ID):
        from google.cloud import bigquery
        bq_client = bigquery.Client(project = PROJECT_ID)
        df = bq_client.query(req_query).to_dataframe()
        print(len(df))
        return df

if __name__ == "__main__":
    df=read_data_from_bq(req_query,PROJECT_ID)
    test_columns(df)
    test_float_title(df)


