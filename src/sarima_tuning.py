import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def sarima_grid_search(
    train_series: pd.Series,
    test_series: pd.Series,
    seasonal_period: int = 7,
    max_results: int = 5,
):
    """
    Small, controlled SARIMA search.
    Ranks models by MAE (lower is better), then RMSE.

    train_series/test_series must be a daily-frequency indexed Series (DatetimeIndex).
    """

    # Small grid (kept intentionally small to avoid long runtimes)
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]

    P_values = [0, 1]
    D_values = [0, 1]
    Q_values = [0, 1]

    results = []

    total = 0
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            total += 1

    tried = 0
    print(f"Total combinations to try: {total}\n")

    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            tried += 1
                            order = (p, d, q)
                            seasonal_order = (P, D, Q, seasonal_period)

                            # Skip obviously weak / redundant configs
                            if order == (0, 0, 0) and seasonal_order[0:3] == (0, 0, 0):
                                continue

                            try:
                                model = SARIMAX(
                                    train_series,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                )

                                # Keep it fast; SARIMA can get slow otherwise
                                fitted = model.fit(disp=False, maxiter=200)

                                forecast = fitted.forecast(steps=len(test_series))
                                forecast = np.array(forecast)

                                y_true = test_series.values
                                y_pred = forecast[: len(y_true)]

                                mae = mean_absolute_error(y_true, y_pred)
                                rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                                results.append({
                                    "order": order,
                                    "seasonal_order": seasonal_order,
                                    "mae": mae,
                                    "rmse": rmse,
                                    "aic": fitted.aic,
                                })

                                print(
                                    f"[{tried}/{total}] SARIMA{order}x{seasonal_order} "
                                    f"MAE={mae:.3f} RMSE={rmse:.3f} AIC={fitted.aic:.1f}"
                                )

                            except Exception as e:
                                # Many configs fail to converge or are unstable; that's normal
                                print(
                                    f"[{tried}/{total}] SARIMA{order}x{seasonal_order} FAILED: {type(e).__name__}"
                                )
                                continue

    if not results:
        print("\nNo valid SARIMA models found in this grid.")
        return None, []

    # Sort by MAE then RMSE (AIC only as a tie-breaker-ish)
    results_sorted = sorted(results, key=lambda r: (r["mae"], r["rmse"], r["aic"]))

    best = results_sorted[0]
    topk = results_sorted[:max_results]

    print("\n===== TOP CONFIGS =====")
    for i, r in enumerate(topk, 1):
        print(
            f"{i}. SARIMA{r['order']}x{r['seasonal_order']} "
            f"MAE={r['mae']:.3f} RMSE={r['rmse']:.3f} AIC={r['aic']:.1f}"
        )

    print("\n===== BEST =====")
    print(
        f"SARIMA{best['order']}x{best['seasonal_order']} "
        f"MAE={best['mae']:.3f} RMSE={best['rmse']:.3f} AIC={best['aic']:.1f}"
    )

    return best, results_sorted


if __name__ == "__main__":
    # Minimal runner so you can execute this file directly.
    from data_loader import load_data
    from preprocessing import preprocess_data

    df_raw = load_data("../data/sales_data.csv")
    df_reg = preprocess_data(df_raw.copy())

    # Use regression test start date to align windows (same as your pipeline)
    split_index = int(len(df_reg) * 0.8)
    split_date = df_reg["date"].iloc[split_index]

    series = df_raw.set_index("date")["sales"].sort_index().asfreq("D")

    train_series = series[series.index < split_date]
    test_series = series[series.index >= split_date]

    sarima_grid_search(train_series, test_series, seasonal_period=7, max_results=5)