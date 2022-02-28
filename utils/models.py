import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_validate
from sklearn.model_selection import TimeSeriesSplit


def create_model(
    model,
    train_df,
    test_df,
    features_in=["year", "temperature_avg"],
    features_out=["bloom_doy"],
    verbose=True,
    n_splits=5,
):
    input_features_train = train_df.loc[:, features_in]
    output_features_train = train_df.loc[:, features_out]

    # Check if dataset has 2022 data
    dataset_has_2022 = 2022 == test_df["year"].to_list()[-1]

    # Take all features except last year i.e 2022
    if dataset_has_2022:
        print("Dataset has 2022!")
        input_features_test = test_df.iloc[:-1].loc[:, features_in]
        output_features_test = test_df.iloc[:-1].loc[:, features_out]

        input_features_test_2022 = test_df.iloc[-1].loc[features_in]
    else:
        print("Dataset does NOT have 2022!")
        input_features_test = test_df.loc[:, features_in]
        output_features_test = test_df.loc[:, features_out]

    # Setup normalizer

    std_scaler = StandardScaler()
    std_scaler.fit(input_features_train)

    # Fit model
    cv_metrics = cross_validate(
        estimator=model,
        X=std_scaler.transform(input_features_train),
        y=output_features_train,
        cv=TimeSeriesSplit(n_splits=n_splits),
        return_train_score=True,
        return_estimator=True,
        scoring="neg_mean_absolute_error",
    )

    if verbose:
        print(f"--------- Results of k-fold cross validation ---------")
        print("MAE = mean absolute error")
        print(
            f'In-fold MAE (mean, std): {-np.mean(cv_metrics["train_score"]):.4f}, {np.std(cv_metrics["train_score"]):.4f}'
        )
        print(
            f'Out-of-fold MAE (mean, std): {-np.mean(cv_metrics["test_score"]):.4f}, {np.std(cv_metrics["test_score"]):.4f}'
        )

    out_of_fold_score = float(np.mean(cv_metrics["test_score"]))

    if verbose:
        cv_perf_df = pd.DataFrame(
            {
                "In-fold error": -cv_metrics["train_score"],
                "Out-of-fold error": -cv_metrics["test_score"],
            }
        )
        cv_perf_df.plot()
        plt.show()

    model.fit(X=std_scaler.transform(input_features_train), y=output_features_train)

    predicted_values_train = model.predict(std_scaler.transform(input_features_train))
    predicted_values_test = model.predict(std_scaler.transform(input_features_test))

    if dataset_has_2022:
        predicted_values_test_2022 = model.predict(
            std_scaler.transform(
                np.expand_dims(input_features_test_2022.to_numpy(), axis=0)
            )
        )

    # Compute performance on test set
    test_mae = mean_absolute_error(
        y_true=output_features_test, y_pred=predicted_values_test
    )

    # Display feature importances with mean and std across folds
    if verbose and hasattr(model, "coef_"):
        coeffs = pd.DataFrame(
            [np.squeeze(model.coef_).tolist() for model in cv_metrics["estimator"]],
            columns=features_in,
        )
        print(
            f"--------- Model features and their importance during prediction ---------"
        )

        print("Features used by the model:")
        print(features_in)
        print()

        print("Feature values across the k-folds:")
        print(coeffs)
        print()

        print(
            f"Feature coefficients for the final model "
            f"(complete training data {train_df['year'].min()}-{train_df['year'].max()}):"
        )
        print(model.coef_)
        print()

        print(
            f"Feature importance in percentage for the final model "
            f"(complete training data {train_df['year'].min()}-{train_df['year'].max()}):"
        )
        print(np.around(100 * np.abs(model.coef_) / np.sum(np.abs(model.coef_)), 3))
        print()

        print(
            f"MAE on test data (years {test_df['year'].min()}-{test_df['year'].max()}): {test_mae:.4f}"
        )
        print()

        print("Feature coefficients across the k-fold (mean and std):")
        sns.barplot(orient="h", data=coeffs, order=features_in, ci="sd", capsize=0.2)
        plt.show()

    predicted_names = [f"{f}_pred" for f in features_out]
    train_df[predicted_names] = (
        predicted_values_train
        if predicted_values_train.ndim > 1
        else np.expand_dims(predicted_values_train, axis=1)
    )
    if dataset_has_2022:
        predicted_values_test = np.concatenate(
            [predicted_values_test, predicted_values_test_2022], axis=0
        )
    test_df[predicted_names] = (
        predicted_values_test
        if predicted_values_test.ndim > 1
        else np.expand_dims(predicted_values_test, axis=1)
    )

    # Plot predictions on train and test data
    if verbose:
        print(
            "Plot showing model predictions on training (orange) and testing data (red)"
        )
        print("`True bloom DOY` is in blue.")
        print(
            "Note: for plotting purposes we set the `True bloom DOY` to 1 for 2022 as its true value is unknown."
        )
        for predicted_name, expected_name in zip(predicted_names, features_out):
            g = sns.lineplot(
                x="year", y=expected_name, hue="location", data=train_df
            )  # ground truth train
            sns.lineplot(
                x=train_df["year"].tolist(),
                y=train_df[predicted_name].tolist(),
                label=predicted_name,
                ax=g,
            )  # predicted train

            sns.lineplot(
                x="year", y=expected_name, hue="location", data=test_df, ax=g
            )  # ground truth test
            sns.lineplot(
                x=test_df["year"].tolist(),
                y=test_df[predicted_name].tolist(),
                label=f"forecasted_{predicted_name}",
                ax=g,
            )  # predicted test

            g.figure.set_size_inches(15, 4)
            plt.show()

    return -out_of_fold_score, model


def get_best_model(models, train_df, test_df, features_in, verbose=False, n_splits=5):
    if len(models) != len(features_in):
        raise ValueError(
            f"Number of models (={len(models)} does not match features in (={len(features_in)}) "
        )
    best_out_of_fold_score = np.inf
    best_model_index = -1
    for index, (model, features_in_) in enumerate(zip(models, features_in)):
        out_of_fold_score, _ = create_model(
            model,
            train_df,
            test_df,
            features_in=features_in_,
            verbose=verbose,
            n_splits=n_splits,
        )
        if out_of_fold_score < best_out_of_fold_score:
            best_model_index = index
            best_out_of_fold_score = out_of_fold_score

    print(f"Tested {len(models)} models and found the best")
    print(
        f"Found the best model with {best_out_of_fold_score} score at {best_model_index}"
    )
    create_model(
        models[best_model_index],
        train_df,
        test_df,
        features_in=features_in[best_model_index],
        verbose=True,
        n_splits=n_splits,
    )
