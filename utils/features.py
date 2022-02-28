import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utilities import ChainedAssignment, correct_doy


def compute_model_param_features(
    temp_df,
    r_chill,
    r_warm,
    months_to_consider=[9, 10, 11, 12, 1, 2, 3],
    last_day_of_last_month=None,
    verbose=False,
):
    features_dict = {
        "r_chill_complete_doy": [],
        "year": [],
        "r_warm_completion": [],
        "r_warm_half_completion_doy": [],
    }
    r_warm_completion_features = [0.01, 0.1, 0.25, 0.5, 0.75]
    for fraction_complete in r_warm_completion_features:
        features_dict[f"r_warm_{fraction_complete}_completion"] = []

    r_warm_completion_doy_features = [1, 15, 29, 43, 57]
    for doy in r_warm_completion_doy_features:
        features_dict[f"r_warm_{doy}_completion"] = []

    for year, group in temp_df.groupby("year"):

        if last_day_of_last_month:
            last_month_mask = group["month"].isin([months_to_consider[-1]]) & (
                group["day"] <= last_day_of_last_month
            )
            month_mask = group["month"].isin(months_to_consider[:-1])
            month_mask = month_mask | last_month_mask
        else:
            month_mask = group["month"].isin(months_to_consider)

        temp_df_within_months = group.loc[month_mask]

        # Find the doy when r_chill is complete
        with ChainedAssignment():
            temp_df_within_months.loc[:, ["cum_chill"]] = temp_df_within_months.loc[
                :, "chill_day"
            ].cumsum()

        index_when_r_chill_is_complete = int(
            np.argmin(np.abs(temp_df_within_months["cum_chill"] - r_chill))
        )
        doy_r_chill_complete = temp_df_within_months.iloc[
            index_when_r_chill_is_complete
        ]["DOY"]
        date_r_chill_complete = temp_df_within_months.iloc[
            index_when_r_chill_is_complete
        ]["date"]

        if verbose and False:
            print(f"Index when r_chill is complete: {index_when_r_chill_is_complete}")
            print(f"DOY when r_chill is complete: {doy_r_chill_complete}")
            print(f"Date when r_chill is complete: {date_r_chill_complete}")

            fig, ax = plt.subplots(1, 1, figsize=(30, 10))
            sns.lineplot(data=temp_df_within_months, y="date", x="cum_chill", ax=ax)
            ax.axvline(
                r_chill, color="green", linestyle="--", label="Chill requirement"
            )
            plt.show()

        # Find the doy when r_warm is complete; this day should fall after the day when r_chill is complete!
        constrain_mask = temp_df_within_months["date"] >= date_r_chill_complete
        temp_df_within_months_after_r_chill = temp_df_within_months.loc[constrain_mask]

        with ChainedAssignment():
            temp_df_within_months_after_r_chill.loc[
                :, "cum_GDD"
            ] = temp_df_within_months_after_r_chill.loc[:, "GDD_day"].cumsum()

        index_when_r_warm_is_complete = int(
            np.argmin(np.abs(temp_df_within_months_after_r_chill["cum_GDD"] - r_warm))
        )
        amount_of_r_warm_completed = temp_df_within_months_after_r_chill.iloc[
            index_when_r_warm_is_complete
        ]["cum_GDD"]
        doy_r_warm_complete = temp_df_within_months_after_r_chill.iloc[
            index_when_r_warm_is_complete
        ]["DOY"]
        date_r_warm_complete = temp_df_within_months_after_r_chill.iloc[
            index_when_r_warm_is_complete
        ]["date"]

        # Compute r_warm half complete DOY
        index_when_r_warm_half_is_complete = int(
            np.argmin(
                np.abs(temp_df_within_months_after_r_chill["cum_GDD"] - (r_warm / 2))
            )
        )
        doy_r_warm_half_complete = temp_df_within_months_after_r_chill.iloc[
            index_when_r_warm_half_is_complete
        ]["DOY"]

        # Compute if r_warm has completed to a certain amount
        for fraction_complete in r_warm_completion_features:
            is_complete = (
                1 if amount_of_r_warm_completed >= (r_warm * fraction_complete) else 0
            )

            features_dict[f"r_warm_{fraction_complete}_completion"].append(is_complete)

        # Compute what fraction of r_warm has been completed on a given DOY
        for doy in r_warm_completion_doy_features:
            doy_index = int(
                np.argmin(np.abs(temp_df_within_months_after_r_chill["DOY"] - doy))
            )
            amount_of_fractional_r_warm_completed = (
                temp_df_within_months_after_r_chill.iloc[doy_index]["cum_GDD"]
            )

            fraction_completed_on_doy = amount_of_fractional_r_warm_completed / r_warm
            features_dict[f"r_warm_{doy}_completion"].append(fraction_completed_on_doy)

        features_dict["r_chill_complete_doy"].append(correct_doy(doy_r_chill_complete))
        features_dict["r_warm_completion"].append(amount_of_r_warm_completed / r_warm)
        features_dict["r_warm_half_completion_doy"].append(doy_r_warm_half_complete)
        features_dict["year"].append(year)

    return pd.DataFrame(features_dict)


def compute_date_features(df):
    # to_month = dict([(1, "January"), (2, "Feburary"), (3, "March"), (4, "April"), (5, "May")])
    months = []
    for date in df["bloom_date"]:
        months.append(int(date.split("-")[1]))
    df["month"] = months
