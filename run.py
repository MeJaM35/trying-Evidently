import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently import ColumnMapping
from evidently.metrics import DatasetDriftMetric

# Define features
features = [
    "dest_npa_data_fv_stats_view__dti","dest_npa_data_fv_stats_view__term","dest_npa_data_fv_stats_view__grade",
    "dest_npa_data_fv_stats_view__purpose","dest_npa_data_fv_stats_view__int_rate","dest_npa_data_fv_stats_view__zip_code",
    "dest_npa_data_fv_stats_view__loan_amnt","dest_npa_data_fv_stats_view__sub_grade","dest_npa_data_fv_stats_view__total_acc",
    "dest_npa_data_fv_stats_view__addr_state","dest_npa_data_fv_stats_view__annual_inc","dest_npa_data_fv_stats_view__recoveries",
    "dest_npa_data_fv_stats_view__revol_util","dest_npa_data_fv_stats_view__tot_cur_bal","dest_npa_data_fv_stats_view__tot_coll_amt",
    "dest_npa_data_fv_stats_view__last_week_pay","dest_npa_data_fv_stats_view__total_rec_int","dest_npa_data_fv_stats_view__batch_enrolled",
    "dest_npa_data_fv_stats_view__home_ownership","dest_npa_data_fv_stats_view__total_rev_hi_lim","dest_npa_data_fv_stats_view__initial_list_status"
]

categorical_features = [
    "dest_npa_data_fv_stats_view__term","dest_npa_data_fv_stats_view__grade","dest_npa_data_fv_stats_view__purpose",
    "dest_npa_data_fv_stats_view__zip_code","dest_npa_data_fv_stats_view__sub_grade","dest_npa_data_fv_stats_view__addr_state",
    "dest_npa_data_fv_stats_view__last_week_pay","dest_npa_data_fv_stats_view__batch_enrolled",
    "dest_npa_data_fv_stats_view__home_ownership","dest_npa_data_fv_stats_view__initial_list_status"
]

numerical_features = [
    "dest_npa_data_fv_stats_view__dti","dest_npa_data_fv_stats_view__int_rate","dest_npa_data_fv_stats_view__loan_amnt",
    "dest_npa_data_fv_stats_view__total_acc","dest_npa_data_fv_stats_view__annual_inc","dest_npa_data_fv_stats_view__recoveries",
    "dest_npa_data_fv_stats_view__revol_util","dest_npa_data_fv_stats_view__tot_cur_bal","dest_npa_data_fv_stats_view__tot_coll_amt",
    "dest_npa_data_fv_stats_view__total_rec_int","dest_npa_data_fv_stats_view__total_rev_hi_lim"
]

# Load data
reference = pd.read_csv("reference_data.csv", usecols=features)
inference = pd.read_csv("inference_data.csv", usecols=features)

# categorical_features = list(set(categorical_features) & set(reference))
# numerical_features = list(set(numerical_features) & set(reference))

# Define column mapping for Evidently
column_mapping = ColumnMapping(
    categorical_features=categorical_features,
    numerical_features=numerical_features,
    task="classification",
    datetime=None,
    prediction=None,
    target=None
)

# Create Evidently report with DatasetDriftMetric
report = Report(metrics=[DatasetDriftMetric()])

try:
    report.run(
        reference_data=reference,
        current_data=inference,
        column_mapping=column_mapping
    )
    # Getting the calculated metrics output in dict
    metrics_dict = report.as_dict()
    print(metrics_dict)
except Exception as e:
    print(f"Evidently report failed: {e}")