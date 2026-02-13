import pandas as pd

from data_processing.tht_tripgen_transformer import THTTripGenTransformer


def test_tht_tripgen_transformer_drop_unknown():
    train_df = pd.DataFrame(
        {
            "hour_of_day": [0, 1],
            "r": [0.1, 0.2],
            "pickup_id": [1, 2],
            "dropoff_id": [2, 1],
            "day_of_week": [0, 1],
            "passenger_count": [1, 2],
            "total_amount": [12.0, 24.0],
        }
    )
    val_df = pd.DataFrame(
        {
            "hour_of_day": [0, 1],
            "r": [0.3, 0.4],
            "pickup_id": [1, 99],
            "dropoff_id": [2, 1],
            "day_of_week": [0, 1],
            "passenger_count": [1, 2],
            "total_amount": [13.0, 25.0],
        }
    )

    transformer = THTTripGenTransformer().fit(train_df)
    encoded = transformer.transform(val_df, drop_unknown=True)

    assert len(encoded) == 1
    assert encoded["o_idx"].iloc[0] == transformer.zone_to_idx[1]
