"""Auxiliares para discretização de pares origem-destino."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from utils.zone_id import assign_zone_names

import unicodedata

__all__ = [
    "ODDiscretizerArtifacts",
    "discretize_od_pairs",
    "prepare_discretized_data",
    "apply_artifacts",
]


def _normalize_location(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(text.split())


@dataclass
class ODDiscretizerArtifacts:
    pickup_categories: List[str]
    dropoff_categories: List[str]
    od_categories: List[str]
    pickup_to_id: Dict[str, int]
    dropoff_to_id: Dict[str, int]
    od_to_id: Dict[str, int]
    od_to_pickup_id: Sequence[int]
    od_to_dropoff_id: Sequence[int]

    def id_to_pair(self, od_id: int) -> str:
        return self.od_categories[od_id]

    def id_to_pickup(self, pickup_id: int) -> str:
        return self.pickup_categories[pickup_id]

    def id_to_dropoff(self, dropoff_id: int) -> str:
        return self.dropoff_categories[dropoff_id]

    def pickup_id_for_od(self, od_id: int) -> int:
        return int(self.od_to_pickup_id[od_id])

    def dropoff_id_for_od(self, od_id: int) -> int:
        return int(self.od_to_dropoff_id[od_id])


def discretize_od_pairs(df: pd.DataFrame) -> Tuple[pd.DataFrame, ODDiscretizerArtifacts]:
    df = df.copy()
    df["pickup_location"] = df["PU_zone_name"].map(_normalize_location)
    df["dropoff_location"] = df["DO_zone_name"].map(_normalize_location)

    mask_valid = (df["pickup_location"] != "") & (df["dropoff_location"] != "")
    df = df.loc[mask_valid].reset_index(drop=True)

    df["od_pair"] = df["pickup_location"] + "-" + df["dropoff_location"]

    pickup_cat = pd.Categorical(df["pickup_location"])
    dropoff_cat = pd.Categorical(df["dropoff_location"])
    od_cat = pd.Categorical(df["od_pair"])

    df["pickup_id"] = pickup_cat.codes.astype("int64")
    df["dropoff_id"] = dropoff_cat.codes.astype("int64")
    df["od_id"] = od_cat.codes.astype("int64")

    od_to_pickup = (
        df.groupby("od_id")["pickup_id"].first().reindex(range(len(od_cat.categories))).to_numpy()
    )
    od_to_dropoff = (
        df.groupby("od_id")["dropoff_id"].first().reindex(range(len(od_cat.categories))).to_numpy()
    )

    artifacts = ODDiscretizerArtifacts(
        pickup_categories=pickup_cat.categories.tolist(),
        dropoff_categories=dropoff_cat.categories.tolist(),
        od_categories=od_cat.categories.tolist(),
        pickup_to_id={name: idx for idx, name in enumerate(pickup_cat.categories)},
        dropoff_to_id={name: idx for idx, name in enumerate(dropoff_cat.categories)},
        od_to_id={name: idx for idx, name in enumerate(od_cat.categories)},
        od_to_pickup_id=od_to_pickup,
        od_to_dropoff_id=od_to_dropoff,
    )
    return df, artifacts


def prepare_discretized_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, ODDiscretizerArtifacts]:
    enriched = assign_zone_names(df.copy())
    return discretize_od_pairs(enriched)


def apply_artifacts(df: pd.DataFrame, artifacts: ODDiscretizerArtifacts) -> pd.DataFrame:
    """
    Aplica mapeamentos de categorias (treinados) a um novo DataFrame já
    enriquecido com nomes de zonas, garantindo consistência de IDs entre
    treino/val/hold.

    Espera colunas: 'PU_zone_name', 'DO_zone_name'.
    Retorna DataFrame com colunas: pickup_id, dropoff_id, od_id, od_pair
    e cópias normalizadas de 'pickup_location'/'dropoff_location'. Linhas com
    pares OD desconhecidos (fora do vocabulário de treino) são removidas.
    """
    df = df.copy()
    # Normaliza textos como em discretize_od_pairs
    df["pickup_location"] = df["PU_zone_name"].map(_normalize_location)
    df["dropoff_location"] = df["DO_zone_name"].map(_normalize_location)
    mask_valid = (df["pickup_location"] != "") & (df["dropoff_location"] != "")
    df = df.loc[mask_valid].reset_index(drop=True)

    # IDs a partir dos dicionários do treino; valores ausentes viram NaN
    df["pickup_id"] = df["pickup_location"].map(artifacts.pickup_to_id).astype("float")
    df["dropoff_id"] = df["dropoff_location"].map(artifacts.dropoff_to_id).astype("float")
    df["od_pair"] = df["pickup_location"] + "-" + df["dropoff_location"]
    df["od_id"] = df["od_pair"].map(artifacts.od_to_id).astype("float")

    before = len(df)
    df = df.dropna(subset=["pickup_id", "dropoff_id", "od_id"]).reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"[Discretização] apply_artifacts: linhas removidas por categorias desconhecidas = {removed}/{before}")

    # Converte para int64
    df["pickup_id"] = df["pickup_id"].astype("int64")
    df["dropoff_id"] = df["dropoff_id"].astype("int64")
    df["od_id"] = df["od_id"].astype("int64")

    return df
