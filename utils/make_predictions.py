import pandas as pd
import numpy as np

from dataset import get_dataset
from models import create_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


def generate_prediction_file(
    use_all_data_to_train=False, verbose=False, read_meteo_from_disk=False
):
    locations_to_compute_for = ["kyoto", "liestal", "washingtondc", "vancouver"]
    predictions_dict = {
        column_names: [] for column_names in ["year"] + locations_to_compute_for
    }
    for y in range(2022, 2032):
        predictions_dict["year"].append(y)

    for location in locations_to_compute_for:
        print("-" * 10, f" Making predictions for {location}... ", "-" * 10)
        print()

        if location in ["kyoto", "liestal", "washingtondc"]:
            train_df, test_df, temp_df = get_dataset(
                earliest_year=1954,  # 1954
                test_set_from=2021 if use_all_data_to_train else 2005,  # 2005, 2015
                add_2022=True,
                months_to_consider=[9, 10, 11, 12, 1, 2, 3],  # [9, 10, 11, 12, 1, 2]
                last_day_of_last_month=7,  # None
                month_to_offset_from=4,  # 3
                location_list=[location],
                run_optimizer=False,
                read_meteo_from_disk=read_meteo_from_disk,
            )  # 1890
        else:
            train_df, test_df, temp_df = get_dataset(
                earliest_year=2016,
                test_set_from=2021,
                add_2022=True,
                months_to_consider=[9, 10, 11, 12, 1, 2, 3],  # [9, 10, 11, 12, 1, 2]
                last_day_of_last_month=7,  # None
                month_to_offset_from=4,  # 3
                location_list=["vancouver"],
                run_optimizer=False,
                read_meteo_from_disk=read_meteo_from_disk,
            )  # 1890

        if location == "liestal":
            create_model(
                Ridge(),
                train_df,
                test_df,
                features_in=["r_chill_complete_doy"]
                + [f"r_warm_{doy}_completion" for doy in [29]]
                + ["winter_chill", "winter_GDD", "spring_GDD"],
                verbose=verbose,
            )  # 1st for Liestal; Test MAE: 4.9864

        elif location == "washingtondc":
            create_model(
                Ridge(),
                train_df,
                test_df,
                features_in=["winter_GDD", "summer_avg", "fall_avg", "winter_avg"],
                verbose=verbose,
            )  # 1st for Washington; Test MAE: 4.5340

        elif location == "kyoto":
            create_model(
                Ridge(),
                train_df,
                test_df,
                features_in=["r_chill_complete_doy", "r_warm_completion", "year"],
                verbose=verbose,
            )  # 1st for Kyoto; Test MAE: 3.3768

        elif location == "vancouver":

            create_model(
                Ridge(
                    alpha=10.0
                ),  # extra regularization due to limited and un-reliable training data
                train_df,
                test_df,
                features_in=["temperature_avg"],  # simple feature to avoid over-fitting
                n_splits=2,
                verbose=verbose,
            )  # 1st for Vancouver; Test MAE: 0.4291
        else:
            raise ValueError(f"{location} is not a valid location.")

        all_predictions = []

        # Get prediction for 2022
        all_predictions.append(
            test_df.loc[test_df["year"] == 2022, "bloom_doy_pred"].to_list()[0]
        )

        # Get predictions for 2023-2030 by fitting a linear model
        full_df = pd.concat([train_df, test_df])
        years_mask = (full_df["year"] >= 1990) & (full_df["year"] <= 2021)
        trend_regression_model = LinearRegression().fit(
            full_df.loc[years_mask, ["year"]],
            full_df.loc[years_mask, ["bloom_doy"]],
        )
        doy_predictions_2023_2031 = trend_regression_model.predict(
            np.array([y for y in range(2023, 2032)]).reshape(-1, 1)
        )
        doy_predictions_2023_2031 = np.squeeze(doy_predictions_2023_2031)
        all_predictions.extend(doy_predictions_2023_2031.tolist())

        predictions_dict[location] = [
            round(predicted_doy) for predicted_doy in all_predictions
        ]

        print("-" * 10, f" Finished making predictions for {location} ", "-" * 10)
        print()

    predictions = pd.DataFrame(predictions_dict)
    predictions.to_csv("../predictions/predicted_bloom_doy.csv", index=False)

    print(
        f"Predictions for all sites have been written to: ../predictions/predicted_bloom_doy.csv"
    )

    return predictions


if __name__ == "__main__":
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_df = generate_prediction_file(use_all_data_to_train=True, verbose=False)
