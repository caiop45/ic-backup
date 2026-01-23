import pandas as pd

from data_processing.strategy_a_transformer import StrategyATransformer


def test_strategy_a_transformer_drop_unknown():
    train_df = pd.DataFrame(
        {
            "hora_do_dia": [0, 1],
            "r": [0.1, 0.2],
            "pickup_id": [1, 2],
            "dropoff_id": [2, 1],
            "dia_da_semana": [0, 1],
        }
    )
    val_df = pd.DataFrame(
        {
            "hora_do_dia": [0, 1],
            "r": [0.3, 0.4],
            "pickup_id": [1, 99],
            "dropoff_id": [2, 1],
            "dia_da_semana": [0, 1],
        }
    )

    transformer = StrategyATransformer().fit(train_df)
    encoded = transformer.transform(val_df, drop_unknown=True)

    assert len(encoded) == 1
    assert encoded["o_idx"].iloc[0] == transformer.zone_to_idx[1]
