# Importierung der benötigten Bibliotheken und Klassen
import numpy as np
import matplotlib.pyplot as plt
import quandl
from sklearn.linear_model import LinearRegression
from yahoo_fin.stock_info import get_data
from keras.layers import LSTM
from keras.models import Sequential

nasdaq_unternehmen = "fb"


class Unternehmen:
    # Daten vorbereiten
    df = get_data(nasdaq_unternehmen, start_date="01/02/2009", end_date="12/29/2019", index_as_date=False,
                  interval="1d")  # Wir holen die Daten - 1.Quelle
    df["Close before"] = df["close"].shift(-1)
    df["Changes"] = (df["close"] / df[
        "Close before"]) - 1  # Berechnung der prozentuellen Veränderung zwischen Close-Werten und Close-Werten davor
    df = df.dropna()
    changes = df["Changes"]
    X_train = []
    y_train = []

    # Unsere X Trainingsdaten müssen folgende Shape haben....
    # X_train = (samples, sequence_length, features)
    for i in range(0, len(df["Changes"]) - 20):  # Bildung von X und y Trainingsdaten
        y_train.append(changes[i])
        X_train.append(np.array(changes[i + 1:i + 21][::-1]))
    X_train = np.array(X_train).reshape(-1, 20, 1)
    y_train = np.array(y_train)

    # Wir bauen unser Neuronales Netzwerk
    model = Sequential()
    model.add(LSTM(1, input_shape=(20, 1)))  # LSTMs werden vor allem bei einer längeren Sequenz von Daten verwendet
    model.compile(optimizer="rmsprop", loss="binary_crossentropy")
    model.fit(X_train, y_train, batch_size=64, epochs=10)  # Das Netzwerk trainiert
    preds = model.predict(X_train)  # Predictions werden gemacht

    # Ergebnis
    preds = np.append(preds, np.zeros(20))
    df["predicted"] = preds
    df["Close predicted"] = df["Close before"] * (
            1 + df["predicted"])  # Hier werden die echten predictedeten Close Werte berechnet

    # Future prediction - Hier machen wir Predictions für die zukünfigen Tage
    # Daten vorbereiten
    df2 = quandl.get("WIKI/" + nasdaq_unternehmen, start_date="01/01/2009", end_date="01/01/2019",
                     api_key="uwhDmpJXeXBPmmzRHaYD")  # Wir holen unsere Daten - 2.Quelle
    forecast_out = 1000  # Tage, für die gepredicted werden sollen
    df2["Prediction"] = df2["Adj. Close"].shift(-forecast_out)

    # Bildung unserer X und y Trainingsdaten
    X = np.array(df2.drop(["Prediction"], 1))
    X = X[:-forecast_out]
    y = np.array(df2["Prediction"])
    y = y[:-forecast_out]

    # Ergebnis predicten
    lr = LinearRegression()  # Verwendung einer "mathematischen Methode", um zukünftige Werte berechnen zu können
    lr.fit(X, y)  # Training
    X_forecast = np.array(df2.drop(["Prediction"], 1))[-forecast_out:]
    pred = lr.predict(X_forecast)  # Predictions werden gemacht
    print("Werte für die nächsten " + str(forecast_out) + " Tage: ", pred)
    df2["Prediction"][-forecast_out:] = pred

    dates = np.array(df["date"]).astype(np.datetime64)
    one_day = np.array('2019-12-30', dtype=np.datetime64)
    date = one_day + np.arange(forecast_out)
    date = date.astype("datetime64[ns]")

    # Hier werden die Graphen geplottet
    plt.plot(dates, np.array(df["close"], dtype=np.float64), label="Close")
    plt.plot(dates, np.array(df["Close predicted"], dtype=np.float64), label="Close predicted")
    plt.plot(date, np.array(df2["Prediction"][-forecast_out:], dtype=np.float64), label="Future predictions")
    plt.title(nasdaq_unternehmen.upper() + "-Kursverlauf")
    plt.legend()
    plt.show()

    # Methoden, um verschiedene Werte auszugeben
    def returnRealVal(self):
        return self.df["close"]

    def returnPredictedVal(self):
        return self.preds

    def returnFutureVal(self):
        return self.pred

    def returnDate1(self):
        return self.dates

    def returnDate2(self):
        return self.date


ai_nasdaq_unternehmen = Unternehmen()  # Objektinstanziierung
print("Reale Werte: ", ai_nasdaq_unternehmen.returnRealVal())
print("Predictedete Werte: ", ai_nasdaq_unternehmen.returnPredictedVal())
print("Future Werte: ", ai_nasdaq_unternehmen.returnFutureVal())
print("Date1: ", ai_nasdaq_unternehmen.returnDate1())
print("Date2: ", ai_nasdaq_unternehmen.returnDate2())
