import pandas as pd

from data_processing.tht_tripgen_transformer import THTTripGenTransformer


def test_tht_tripgen_conditional_cardinalities_keys_match_encoded_columns():
    df = pd.DataFrame(
        {
            "hora_do_dia": [0, 1, 2],
            "r": [0.1, 0.2, 0.3],
            "pickup_id": [1, 2, 3],
            "dropoff_id": [3, 2, 1],
            "dia_da_semana": [0, 3, 5],
            "is_weekend": [0, 0, 1],
            "month": [4, 4, 5],
        }
    )

    transformer = THTTripGenTransformer(use_weekend=True, use_month=True).fit(df)
    encoded = transformer.transform(df)

    conditional_keys = set(transformer.conditional_cardinalities.keys())
    assert "dow_idx" in conditional_keys
    assert "is_weekend_idx" in conditional_keys
    assert "month_idx" in conditional_keys

    for key in conditional_keys:
        assert key in encoded.columns
