import pandas as pd

from data_processing.tht_tripgen_transformer import THTTripGenTransformer


def test_tht_tripgen_shared_zone_vocab_union():
    train_df = pd.DataFrame(
        {
            "hora_do_dia": [0, 1, 2],
            "r": [0.1, 0.2, 0.3],
            "pickup_id": [1, 2, 2],
            "dropoff_id": [2, 3, 3],
            "dia_da_semana": [0, 1, 2],
        }
    )

    transformer = THTTripGenTransformer()
    transformer.fit(train_df)

    assert transformer.zone_categories == [1, 2, 3]
    assert transformer.num_zones == 3

    encoded = transformer.transform(train_df)
    assert encoded.loc[0, "o_idx"] == transformer.zone_to_idx[1]
    assert encoded.loc[1, "d_idx"] == transformer.zone_to_idx[3]
