import pathlib
import pandas as pd
from features import compute_model_param_features
from chill_and_warm_model import ModelOptimizer


chill_and_warm_requirements = {
    "kyoto": (28.58007812, 1806.39648438),
    "liestal": (24.74389648, 1258.0871582),
    "washingtondc": (50.13498235, 1644.31919747),
    "vancouver": (24.828125, 1327.734375),
}


def aggregate_meteo_data(df, temp_df):
    #     print(df.head())
    #     print(temp_df.head())

    temperature_key = "tmean"
    temperature_min_key = "tmin"
    temperature_max_key = "tmax"
    gdd_day_key = "GDD_day"
    chill_day_key = "chill_day"

    temp_df[chill_day_key] = temp_df["tmean"] <= 50.0

    for months, season_name in zip(
        [[4, 5], [6, 7, 8], [9, 10, 11], [12, 1, 2, 3]],
        ["spring", "summer", "fall", "winter"],
    ):
        df = pd.merge(
            df,
            temp_df.groupby(["year"]).apply(
                lambda x: pd.Series(
                    [
                        x[temperature_key][x["month"].isin(months)].mean(),
                        x[temperature_key][x["month"].isin(months)].std(),
                        x[temperature_min_key][x["month"].isin(months)].min(),
                        x[temperature_max_key][x["month"].isin(months)].max(),
                        x[gdd_day_key][x["month"].isin(months)].sum() / 100.0,
                        x[chill_day_key][x["month"].isin(months)].sum(),
                    ],
                    index=[
                        f"{season_name}_avg",
                        f"{season_name}_std",
                        f"{season_name}_min",
                        f"{season_name}_max",
                        f"{season_name}_GDD",
                        f"{season_name}_chill",
                    ],
                ),
            ),
            how="left",
            on="year",
        )

    #     print(df.isnull().sum())
    #     print(df[df.isna().any(axis=1)].year)

    df = pd.merge(
        df,
        temp_df.groupby(["year"]).apply(
            lambda x: pd.Series(
                [
                    x["month"].iloc[x[temperature_max_key].argmax()],
                    x["month"].iloc[x[temperature_min_key].argmin()],
                ],
                index=[
                    "month_warmest",
                    "month_coldest",
                ],
            )
        ),
        how="left",
        on="year",
    )

    df = pd.merge(
        df, temp_df.groupby(["year"])[temperature_key].mean(), how="left", on="year"
    )
    df.rename(columns={temperature_key: "temperature_avg"}, inplace=True)

    df = pd.merge(
        df, temp_df.groupby(["year"])[temperature_key].std(), how="left", on="year"
    )
    df.rename(columns={temperature_key: "temperature_std"}, inplace=True)

    df = pd.merge(
        df, temp_df.groupby(["year"])[temperature_min_key].min(), how="left", on="year"
    )
    df.rename(columns={temperature_min_key: "temperature_min"}, inplace=True)

    df = pd.merge(
        df, temp_df.groupby(["year"])[temperature_max_key].max(), how="left", on="year"
    )
    df.rename(columns={temperature_max_key: "temperature_max"}, inplace=True)

    # Annual GDD and chill days
    df = pd.merge(
        df, temp_df.groupby(["year"])[gdd_day_key].sum() / 100.0, how="left", on="year"
    )
    df.rename(columns={gdd_day_key: "annual_GDD"}, inplace=True)

    df = pd.merge(
        df, temp_df.groupby(["year"])[chill_day_key].sum(), how="left", on="year"
    )
    df.rename(columns={chill_day_key: "annual_chill_day"}, inplace=True)

    return df


def add_2022_row(df):
    last_row_dict = df.iloc[-1].to_dict()
    last_row_dict["year"] = 2022
    last_row_dict["bloom_date"] = "2022-01-01"
    last_row_dict["bloom_doy"] = 1
    df = df.append(last_row_dict, ignore_index=True)
    return df


def get_dataset(
    earliest_year=1850,
    test_set_from=2010,
    add_2022=True,
    months_to_consider=[9, 10, 11, 12, 1, 2],
    last_day_of_last_month=None,
    month_to_offset_from=3,
    location_list=["Liestal"],
    run_optimizer=False,
    read_meteo_from_disk=False,
):

    dfs, temp_dfs = [], []
    for location in location_list:
        df = read_and_preprocess_dataset(add_2022, earliest_year, location)

        temp_df = read_and_preprocess_temperature_dataset(
            earliest_year,
            location,
            month_to_offset_from,
            read_from_disk=read_meteo_from_disk,
        )

        df, temp_df = extract_features(
            df,
            temp_df,
            location=location,
            months_to_consider=months_to_consider,
            last_day_of_last_month=last_day_of_last_month,
        )

        if run_optimizer:
            temp_df_for_model = read_and_preprocess_temperature_dataset(
                earliest_year,
                location,
                month_to_offset_from=6,
                read_from_disk=read_meteo_from_disk,
            )

            df_for_model, temp_df_for_model = extract_features(
                df,
                temp_df_for_model,
                location=location,
                months_to_consider=months_to_consider,
                last_day_of_last_month=last_day_of_last_month,
            )
            m_optimizer = ModelOptimizer(
                df_for_model,
                temp_df_for_model,
                requirement_params=chill_and_warm_requirements[location],
            )
            m_optimizer.find_parameters()

        dfs.append(df)
        temp_dfs.append(temp_df)

    final_temp_df = pd.concat(temp_dfs, ignore_index=True)

    final_df = pd.concat(dfs, ignore_index=True)
    test_mask = final_df["year"] >= test_set_from
    train_df, test_df = final_df.loc[~test_mask], final_df.loc[test_mask]

    rows_before_train_df = len(train_df)
    train_df = train_df.dropna()
    rows_after_train_df = len(train_df)

    print(
        f"Removing rows with NaNs from train_df. Rows before {rows_before_train_df} and after removal {rows_after_train_df}"
    )

    rows_before_test_df = len(test_df)
    test_df = test_df.dropna()
    rows_after_test_df = len(test_df)

    print(
        f"Removing rows with NaNs from test_df. Rows before {rows_before_test_df} and after removal {rows_after_test_df}"
    )

    return train_df, test_df, final_temp_df


def extract_features(
    df,
    temp_df,
    location,
    months_to_consider=[9, 10, 11, 12, 1, 2],
    last_day_of_last_month=None,
):
    # Feature extraction

    r_chill, r_warm = chill_and_warm_requirements[location]

    df = aggregate_meteo_data(df, temp_df)
    df_with_param_features = compute_model_param_features(
        temp_df,
        r_chill=r_chill,
        r_warm=r_warm,
        months_to_consider=months_to_consider,
        last_day_of_last_month=last_day_of_last_month,
    )
    df = pd.merge(df, df_with_param_features, how="left", on="year")
    return df, temp_df


def read_and_preprocess_temperature_dataset(
    earliest_year,
    location,
    month_to_offset_from=3,
    read_from_disk=False,
):
    file_on_disk = pathlib.Path(f"../temperature_data/{location}.csv")

    # Import Meteostat library and dependencies
    from datetime import datetime, date, timedelta
    from meteostat import Point, Daily

    # Set time period
    start = datetime(earliest_year - 1, 1, 1)
    end = datetime(year=2022, month=3, day=8)

    point_location = {
        "kyoto": Point(35.0120, 135.6761, 44),
        "liestal": Point(47.4814, 7.730519, 350),
        "washingtondc": Point(38.8853, -77.0386, 0),
        "vancouver": Point(49.2237, -123.1636, 24),
    }
    temp_df = Daily(point_location[location], start, end).fetch()
    temp_df = temp_df.drop(
        columns=["prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun"]
    )

    if file_on_disk.exists() and read_from_disk:
        temp_df = pd.read_csv(file_on_disk)
        return temp_df

    temp_df["date"] = pd.to_datetime(temp_df.index)
    temp_df = temp_df.reset_index()

    temp_df["day"] = temp_df["date"].dt.day
    temp_df["month"] = temp_df["date"].dt.month
    temp_df["year"] = temp_df["date"].dt.year
    temp_df["DOY"] = temp_df["date"].dt.dayofyear

    temp_df["tavg"] = temp_df["tavg"] * 10.0
    temp_df["tmin"] = temp_df["tmin"] * 10.0
    temp_df["tmax"] = temp_df["tmax"] * 10.0

    temp_df.rename(
        columns={
            "Year": "year",
            "Month": "month",
            "Temperature": "temperature",
            "tavg": "tmean",
        },
        inplace=True,
    )

    temp_df["GDD_day"] = temp_df.apply(lambda row: max(0, row["tmean"] - 50.0), axis=1)
    temp_df["chill_day"] = temp_df["tmean"] <= 50.0

    offset_months(temp_df, month_to_offset_from)
    temp_df = temp_df[temp_df["year"] > earliest_year]

    if not file_on_disk.exists():
        temp_df.to_csv(file_on_disk, index=False)

    return temp_df


def read_and_preprocess_dataset(add_2022, earliest_year, location):
    csv_files = {
        "kyoto": pathlib.Path("../data/kyoto.csv"),
        "liestal": pathlib.Path("../data/liestal.csv"),
        "washingtondc": pathlib.Path("../data/washingtondc.csv"),
        "vancouver": pathlib.Path("../data/vancouver.csv"),
    }
    file_path = csv_files[location]

    df = pd.read_csv(file_path)
    df = df[df["year"] > earliest_year]
    if add_2022:
        df = add_2022_row(df)
    compute_trend_and_lag_features(df)
    return df


def offset_months(temp_df, month_to_offset_from=3):
    # To predict the cheery blosson for a year we need the data from the previous year from March onwards (after the previous year's blossom)
    temp_df.loc[temp_df["month"] >= month_to_offset_from, "year"] = (
        temp_df[temp_df["month"] >= month_to_offset_from]["year"] + 1
    )  # should nbe 3 instead of 4


def compute_trend_and_lag_features(df):
    # Create lag feature from previous year's bloom_doy
    df["bloom_doy_previous_year"] = df["bloom_doy"].shift(
        periods=1, fill_value=df["bloom_doy"].median()
    )
    for trend_window in [5, 10, 20, 30, 40, 50]:
        # Create trend feature using moving averages
        df[f"bloom_doy_trend_{trend_window}_year_window"] = (
            df["bloom_doy"]
            .rolling(
                window=trend_window,
                min_periods=1,  # trend_window//2,
                center=False,  # if set to True we will leak the "future" trend to the model
            )
            .mean()
        )
