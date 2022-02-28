from utilities import ChainedAssignment, correct_doy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_GDD_on_doy(df, temp_df, skip_years=None):
    if not skip_years:
        skip_years = []

    if 2022 not in df["year"].to_list():
        skip_years.append(2022)

    results_dict = {
        "year": [],
        "GDD_on_bloom_doy": [],
    }

    days_before_doy = [10, 20, 30]
    for d_minus_N in days_before_doy:
        results_dict[f"GDD_on_bloom_doy_minus_{d_minus_N}"] = []

    for year, group in temp_df.groupby("year"):
        if year in skip_years:
            continue
        results_dict["year"].append(year)
        dataset_row = df.loc[df["year"] == year]

        # print(dataset_row)
        bloom_doy = dataset_row["bloom_doy"].item()

        # Compute accumulated GDD on bloom DOY
        gdd = temp_df[(temp_df["year"] == year) & (temp_df["DOY"] == bloom_doy)][
            "GDD"
        ].item()
        results_dict["GDD_on_bloom_doy"].append(gdd)
        # print(bloom_doy, gdd)

        # Compute accumulated GDD on bloom DOY - N
        for d_minus_N in days_before_doy:
            doy_to_query = correct_doy(bloom_doy - d_minus_N)
            # print(d_minus_N, bloom_doy - d_minus_N, doy_to_query)
            # print(temp_df[(temp_df["year"] == year) & (temp_df["DOY"] == doy_to_query)])
            gdd = temp_df[(temp_df["year"] == year) & (temp_df["DOY"] == doy_to_query)][
                "GDD"
            ].item()
            results_dict[f"GDD_on_bloom_doy_minus_{d_minus_N}"].append(gdd)
    return pd.DataFrame(results_dict)
