import numpy as np
import matplotlib.pyplot as plt
import quandl
from sklearn.linear_model import LinearRegression
from yahoo_fin.stock_info import get_data
from keras.layers import LSTM
from keras.models import Sequential

class Apple:
    #Daten vorbereiten
    df = get_data("amzn", start_date="01/02/2009", end_date="12/29/2017", index_as_date=False, interval="1d")
    df["Close before"] = df["close"].shift(-1)
    df["Changes"] = (df["close"] / df["Close before"]) - 1
    df = df.dropna()
    print(df)
    changes = df["Changes"]
    X_train = []
    y_train = []

    #X_train = (samples, sequence_length, features)
    for i in range(0, len(df["Changes"]) - 20):
        y_train.append(changes[i])
        X_train.append(np.array(changes[i+1:i+21][::-1]))
    X_train = np.array(X_train).reshape(-1, 20, 1)
    y_train = np.array(y_train)


    #Neuronales Netzwerk
    model = Sequential()
    model.add(LSTM(1, input_shape=(20, 1)))
    model.compile(optimizer="rmsprop", loss="binary_crossentropy")
    model.fit(X_train, y_train, batch_size=64, epochs=10)
    preds = model.predict(X_train)

    #Ergebnis
    preds = np.append(preds, np.zeros(20))
    df["predicted"] = preds
    df["Close predicted"] = df["Close before"] * (1 + df["predicted"])

    # Future Predictions
    #Daten vorbereiten
    df2 = quandl.get("WIKI/AMZN", start_date="01/01/2009", end_date="01/01/2018", api_key="uwhDmpJXeXBPmmzRHaYD")
    forecast_out = 30
    df2["Prediction"] = df2["Adj. Close"].shift(-forecast_out)

    X = np.array(df2.drop(["Prediction"], axis=1))
    X = X[:-forecast_out]
    y = np.array(df2["Prediction"])
    y = y[:-forecast_out]

    #Modell bauen
    lr = LinearRegression()
    lr.fit(X, y)

    #Ergebnis predicten
    X_forecast = np.array(df2.drop(["Prediction"], axis=1))[-forecast_out:]
    pred = lr.predict(X_forecast)
    print("Werte für die nächsten " + str(forecast_out) + " Tage: ", pred)
    df2["Prediction"][-forecast_out:] = pred

    dates = np.array(df["date"]).astype(np.datetime64)
    one_day = np.array('2017-12-30', dtype=np.datetime64)
    date = one_day + np.arange(forecast_out)
    date = date.astype("datetime64[ns]")

    print("Date shape: ", date.shape)
    print(dates.shape)

    #Plotten
    plt.plot(dates, np.array(df["close"], dtype=np.float64), label="Close")
    plt.plot(dates, np.array(df["Close predicted"], dtype=np.float64), label="Close predicted")
    plt.plot(date, np.array(df2["Prediction"][-forecast_out:], dtype=np.float64), label="Future predictions")
    plt.title("Amazon-Kursverlauf")
    plt.legend()
    plt.show()

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


ai_apple = Apple()
print("Reale Werte: ", ai_apple.returnRealVal())
print("Predictedete Werte: ", ai_apple.returnPredictedVal())
print("Future Werte: ", ai_apple.returnFutureVal())
print("Date1: ", ai_apple.returnDate1())
print("Date2: ", ai_apple.returnDate2())