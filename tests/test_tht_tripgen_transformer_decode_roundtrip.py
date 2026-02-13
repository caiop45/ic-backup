import pandas as pd
import pandas.testing as pdt

from tvae.data_processing.tht_tripgen_transformer import THTTripGenTransformer


def test_tht_tripgen_transformer_decode_roundtrip():
    df = pd.DataFrame(
        {
            "hour_of_day": [5, 10, 15],
            "r": [0.25, 0.5, 0.75],
            "pickup_id": [1, 2, 3],
            "dropoff_id": [3, 2, 1],
            "day_of_week": [0, 3, 6],
            "passenger_count": [1, 2, 1],
            "total_amount": [10.0, 20.0, 30.0],
        }
    )

    transformer = THTTripGenTransformer().fit(df)
    encoded = transformer.transform(df)
    decoded = transformer.decode(encoded)

    pdt.assert_series_equal(
        decoded["hour_of_day"], df["hour_of_day"].reset_index(drop=True), check_dtype=False
    )
    pdt.assert_series_equal(
        decoded["pickup_id"], df["pickup_id"].reset_index(drop=True), check_dtype=False
    )
    pdt.assert_series_equal(
        decoded["dropoff_id"], df["dropoff_id"].reset_index(drop=True), check_dtype=False
    )
    pdt.assert_series_equal(
        decoded["day_of_week"],
        df["day_of_week"].reset_index(drop=True),
        check_dtype=False,
    )
    pdt.assert_series_equal(
        decoded["r"], df["r"].reset_index(drop=True).astype("float32"), check_dtype=False
    )
    pdt.assert_series_equal(
        decoded["passenger_count"],
        df["passenger_count"].reset_index(drop=True),
        check_dtype=False,
    )
    clip_lo = transformer.total_amount_clip_lo
    clip_hi = transformer.total_amount_clip_hi
    expected_amount = df["total_amount"].copy()
    if clip_lo is not None:
        expected_amount = expected_amount.clip(lower=clip_lo)
    if clip_hi is not None:
        expected_amount = expected_amount.clip(upper=clip_hi)
    pdt.assert_series_equal(
        decoded["total_amount"].round(4),
        expected_amount.reset_index(drop=True).round(4),
        check_dtype=False,
    )
