from sklearn.preprocessing import StandardScaler

def scale_features(df):
    """
    Ajusta StandardScaler no DataFrame e devolve (scaler, X_scaled).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return scaler, X_scaled
