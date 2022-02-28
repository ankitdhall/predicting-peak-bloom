import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping, Bounds

from utilities import ChainedAssignment, correct_doy


def f(
    df,
    temp_df,
    r_chill,
    r_warm,
    months_to_consider=[9, 10, 11, 12, 1, 2, 3, 4, 5],
    verbose=False,
):
    results_dict = {
        "r_chill_complete_doy": [],
        "predicted_bloom_doy": [],
        "true_bloom_doy": [],
        "year": [],
    }

    for year, group in temp_df.groupby("year"):
        if year not in df["year"].to_list():
            continue
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
        doy_r_warm_complete = temp_df_within_months_after_r_chill.iloc[
            index_when_r_warm_is_complete
        ]["DOY"]
        date_r_warm_complete = temp_df_within_months_after_r_chill.iloc[
            index_when_r_warm_is_complete
        ]["date"]

        if verbose and False:
            print(f"Index when r_warm is complete: {index_when_r_warm_is_complete}")
            print(f"DOY when r_warm is complete: {doy_r_warm_complete}")
            print(f"Date when r_warm is complete: {date_r_warm_complete}")

            fig, ax = plt.subplots(1, 1, figsize=(30, 10))
            sns.lineplot(
                data=temp_df_within_months_after_r_chill, y="date", x="cum_GDD", ax=ax
            )
            ax.axvline(0, color="green", linestyle="--", label="Chill requirement")
            ax.axvline(r_warm, color="red", linestyle="--", label="Warm requirement")
            ax.axhline(
                date_r_warm_complete,
                color="orange",
                linestyle="--",
                label="Date of Warm Requirement completion",
            )
            ax.axhline(
                df.loc[df["year"] == year]["bloom_date"],
                color="tab:olive",
                linestyle="--",
                label="True Bloom date",
            )
            plt.legend()
            plt.show()
        results_dict["r_chill_complete_doy"].append(correct_doy(doy_r_chill_complete))
        results_dict["predicted_bloom_doy"].append(doy_r_warm_complete)
        results_dict["true_bloom_doy"].append(
            df.loc[df["year"] == year]["bloom_doy"].tolist()[0]
        )
        results_dict["year"].append(year)

    results = pd.DataFrame(results_dict)
    results = pd.merge(results, df, how="left", on="year")
    preds = np.array(results["predicted_bloom_doy"])
    gt = np.array(results["true_bloom_doy"])
    results["error"] = results["predicted_bloom_doy"] - results["true_bloom_doy"]
    results["abs_error"] = results["error"].abs()
    if verbose:
        mask = results["predicted_bloom_doy"] > 150
        print(f"Filtering {mask.sum()} values larger than 150 DOY!")
        filtered_results = results.loc[~mask]

        preds = np.array(filtered_results["predicted_bloom_doy"])
        gt = np.array(filtered_results["true_bloom_doy"])
        error = preds - gt
        print(f"MAE: {float(np.mean(np.abs(error))): .4f}")
        print(f"std: {float(np.std(error)): .4f}")

        fig, ax = plt.subplots(1, 1, figsize=(30, 10))
        sns.lineplot(
            data=filtered_results,
            x="year",
            y="abs_error",
            label="Absolute error",
            ax=ax,
        )
        sns.lineplot(
            data=filtered_results,
            x="year",
            y="true_bloom_doy",
            label="true_bloom_doy error",
            ax=ax,
        )
        sns.lineplot(
            data=filtered_results,
            x="year",
            y="predicted_bloom_doy",
            label="predicted_bloom_doy",
            ax=ax,
        )
        # sns.lineplot(data=results, x="year", y="r_chill_complete_doy", label="r_chill_complete_doy", ax=ax)

    return results


class ModelOptimizer:
    def __init__(self, df, temp_df, requirement_params, years_to_test=None):
        self.df = df
        self.temp_df = temp_df
        self.months_to_consider = [9, 10, 11, 12, 1, 2, 3, 4, 5]
        self.verbose = False
        self.requirement_params = requirement_params

    def evaluate_f(self, params):
        r_chill, r_warm = params

        results = f(
            self.df,
            self.temp_df,
            r_chill,
            r_warm,
            self.months_to_consider,
            self.verbose,
        )

        return float((results["abs_error"]).mean())

    def find_parameters(self, verbose=False):
        print("----------Finding model parameters----------")

        years_to_skip = []  # [2015, 2016]
        self.df = self.df[
            (self.df["year"] < 2021) & (~self.df["year"].isin(years_to_skip))
        ]
        self.temp_df = self.temp_df[
            (self.temp_df["year"] < 2021) & (~self.temp_df["year"].isin(years_to_skip))
        ]

        print("Calling f()")
        # f(60, 100)
        # print(basinhopping(f, (30, 1000), niter=100))
        # print(minimize(f, (30, 1000), method='L-BFGS-B'))

        # Optimal until now:
        self.verbose = True
        self.evaluate_f(self.requirement_params)
        self.verbose = False

        # Tokyo: 37.453125  , 1299.609375
        # Liesthal: 29.05515575, 1544.47233677
        # WDC: 32.44367149, 1734.98590408

        #         print(minimize(f, (2, 10), method='Powell', bounds=Bounds([0, 0], [365, 30000])))
        #         print(minimize(f, (30, 1000), method='Powell'))
        #         print(minimize(f, (50, 10000), method='Powell'))

        #         print(minimize(f, (2, 10), method='nelder-mead', bounds=Bounds(lb=[0, 0], ub=[365, 30000])))
        #         print(minimize(f, (1, 100), method='nelder-mead'))
        #         print(minimize(f, (10, 100), method='nelder-mead'))
        #         print(minimize(f, (10, 1000), method='nelder-mead'))
        optimal_values = minimize(self.evaluate_f, (20, 500), method="nelder-mead")
        print(
            f"Optimal values to use for chill and warm requirement are: {optimal_values.x}"
        )
        #         print(minimize(f, (30, 1000), method='nelder-mead'))
        #         print(minimize(f, (50, 10000), method='nelder-mead'))

        print("----------Exiting Finding model parameters----------")
