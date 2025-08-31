from backtesting import Strategy

class MyStrategy(Strategy):

    def init(self):
        # 从数据中读取提前准备好的布尔信号
        self.buy_signal = self.data.df['buy_signal']
        self.sell_signal = self.data.df['sell_signal']
        self.index = 0

    def next(self):
        if self.buy_signal.iloc[self.index]:
            if not self.position.is_long:
                self.position.close()
                self.buy()
        elif self.sell_signal.iloc[self.index]:
            if not self.position.is_short:
                self.position.close()
                self.sell()
        self.index += 1
