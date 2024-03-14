def get_supertrend_description():
    return (
        "The SuperTrend strategy is a trend-following indicator",
        "that uses",
        ("Average True Range (ATR)", "", "#fea"),
        "to determine the direction of the trend. The ATR is a measure of market volatility, \
                and the SuperTrend indicator uses it to calculate the trend direction. \
                A buy signal is generated when the price crosses above the SuperTrend line, \
                indicating a bullish trend, while a sell signal occurs when the price crosses below the SuperTrend line, \
                signaling a bearish trend. This method aims to filter market noise and improve trade reliability by using \
                a volatility-based indicator to determine trend direction.",
    )


def get_sma_description():
    return (
        "The Triple SMA Crossover strategy is a trend-following indicator",
        "that uses",
        ("Simple Moving Averages (SMA)", "", "#fea"),
        "to determine the direction of the trend. This strategy uses three different \
                timeframes to calculate the moving averages, generating buy and sell signals \
                based on the crossovers of these moving averages. A buy signal is generated when \
                the short-term moving average crosses above the medium-term moving average, \
                indicating a bullish trend, while a sell signal occurs when the short-term moving \
                average crosses below the medium-term moving average, signaling a bearish trend. \
                This method aims to filter market noise and improve trade reliability by using \
                a trend-following indicator to determine trend direction.",
    )


def get_donchian_description():
    return (
        "The Donchian Channel strategy, developed by Richard Donchian",
        "is a technical analysis tool used to identify market trends ",
        (
            "through the highest and lowest prices over a specific period.",
            "",
            "#fea",
        ),
        "This period is defined by",
        ("upper_length and lower_length parameters", "", "#fea"),
        "which dictate the timeframe for the",
        ("highest high (upper channel) and lowest low (lower channel)", "", "#fea"),
        "respectively. Typically, traders use these channels to spot breakout points,"
        "going long when prices push above the upper channel and short when they drop "
        "below the lower channel, leveraging these parameters to fine-tune the strategy's sensitivity to market movements.",
    )


def get_ml_strategy_description():
    return (
        "The ML-based strategy uses",
        ("machine learning models", "", "#fea"),
        "to predict future price movements. The strategy uses historical price data to train the model, \
                which then generates buy and sell signals based on the predicted price movements. \
                This method aims to filter market noise and improve trade reliability by using \
                a machine learning model to predict future price movements.",
    )


def get_lstm_strategy_description():
    return (
        "The LSTM-based strategy uses",
        ("Long Short-Term Memory (LSTM) networks", "", "#fea"),
        "to predict future price movements. The strategy uses historical price data to train the model, \
                which then generates buy and sell signals based on the predicted price movements. \
                This method aims to filter market noise and improve trade reliability by using \
                a machine learning model to predict future price movements.",
    )
