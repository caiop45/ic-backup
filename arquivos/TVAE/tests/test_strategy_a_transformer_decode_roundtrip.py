import pandas as pd
import pandas.testing as pdt

from data_processing.strategy_a_transformer import StrategyATransformer


def test_strategy_a_transformer_decode_roundtrip():
    df = pd.DataFrame(
        {
            "hora_do_dia": [5, 10, 15],
            "r": [0.25, 0.5, 0.75],
            "pickup_id": [1, 2, 3],
            "dropoff_id": [3, 2, 1],
            "dia_da_semana": [0, 3, 6],
        }
    )

    transformer = StrategyATransformer().fit(df)
    encoded = transformer.transform(df)
    decoded = transformer.decode(encoded)

    pdt.assert_series_equal(
        decoded["hora_do_dia"], df["hora_do_dia"].reset_index(drop=True), check_dtype=False
    )
    pdt.assert_series_equal(
        decoded["pickup_id"], df["pickup_id"].reset_index(drop=True), check_dtype=False
    )
    pdt.assert_series_equal(
        decoded["dropoff_id"], df["dropoff_id"].reset_index(drop=True), check_dtype=False
    )
    pdt.assert_series_equal(
        decoded["dia_da_semana"],
        df["dia_da_semana"].reset_index(drop=True),
        check_dtype=False,
    )
    pdt.assert_series_equal(
        decoded["r"], df["r"].reset_index(drop=True).astype("float32"), check_dtype=False
    )
