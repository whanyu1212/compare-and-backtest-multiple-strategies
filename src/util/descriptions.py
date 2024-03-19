def get_supertrend_description():
    return (
        "The Supertrend indicator, combines the",
        ("Average True Range (ATR) with a multiplier", "", "#fea"),
        "to calculate the trend direction. This calculation incorporates both market volatility, captured by the ATR, and a predefined multiplier which adjusts the sensitivity of the trend indicator. \
        Buy signals are generated when the price moves",
        ("above the indicator line, suggesting an uptrend", "", "#fea"),
        "indicating potential buying opportunities. Conversely, sell signals are issued when the price moves",
        ("below the indicator line, indicating a downtrend", "", "#fea"),
        "pointing towards selling or short-selling opportunities. Traders leverage these signals to capitalize on the momentum of the market's direction, aiming to enter or exit trades based on the trend's strength and potential reversal points.",
    )


def get_sma_description():
    return (
        "The Triple SMA Crossover strategy is a trend-following indicator",
        "that uses",
        ("Simple Moving Averages (SMA)", "", "#fea"),
        "to determine the direction of the trend. This strategy uses",
        ("three different timeframes", "", "#fea"),
        "to calculate the moving averages, generating buy and sell signals based on the crossovers of these moving averages. A buy signal is generated when the",
        (
            "short-term moving average crosses above the medium-term moving average",
            "",
            "#fea",
        ),
        "indicating a bullish trend, while a sell signal occurs when the",
        (
            "short-term moving average crosses below the medium-term moving average",
            "",
            "#fea",
        ),
        "signaling a bearish trend. This method aims to filter market noise and improve trade reliability by using a trend-following indicator to determine trend direction.",
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
        "The ML-based trading strategy employs a",
        ("LightGBM (LGB) Classifier", "", "#fea"),
        "to forecast if the price of a financial asset n days in the future will exceed the current price. \
        This forecast leverages historical price data and potentially other financial indicators to train the LightGBM model, \
        a gradient boosting framework that uses tree based learning algorithms. The prediction of future price movements is achieved by analyzing patterns in the data, \
        where the model outputs a probability score for the likelihood of the price increase.",
        "Based on the model's",
        ("predicted price movements", "", "#fea"),
        "and a defined",
        ("predict_proba threshold", "", "#fea"),
        ", buy signals are generated when the probability of a price increase exceeds the threshold, suggesting a bullish outlook. \
        Conversely, sell signals are issued when the probability falls below the threshold, indicating a bearish market sentiment. \
        This strategy enables traders to make informed decisions on entering or exiting positions by quantitatively assessing the potential for future price changes.",
    )


def get_lstm_strategy_description():
    return (
        "The LSTM-based trading strategy employs",
        ("Long Short-Term Memory (LSTM) networks", "", "#fea"),
        "to forecast future price movements of financial assets. This strategy leverages a deep learning model, specifically LSTM networks, \
                known for their ability to capture temporal dependencies and patterns in time-series data, such as historical price series. \
                By training on this data, the LSTM model learns to predict future prices.",
        "The strategy advances by deriving",
        ("predicted returns", "", "#fea"),
        "from these predicted prices, calculating the potential percentage change in price from the current price to the predicted future price. \
                Signals to buy or sell are then generated based on whether these predicted returns exceed a certain threshold.",
        "Specifically, a",
        ("buy signal", "", "#fea"),
        "is issued if the predicted return is above a positive threshold, indicating an anticipated price increase significant enough to consider entering a long position. \
                Conversely, a",
        ("sell signal", "", "#fea"),
        "is generated if the predicted return falls below a negative threshold, suggesting a sufficient anticipated price decrease to consider taking a short position. \
                This approach aims to mitigate market noise and enhance trading decision reliability by utilizing a predictive machine learning model to guide entry and exit strategies based on anticipated price movements and derived returns.",
    )
