import numpy as np


class Calc:

    @staticmethod
    def sma(values, window=50):
        weights = np.repeat(1.0, window) / window
        smas = np.convolve(values, weights, 'valid')
        smas = Calc.__fill_ma_beg(smas, len(values), window)
        return smas

    @staticmethod
    def ema(s, n=200):
        """
        returns an n period exponential moving average for
        the time series s

        s is a list ordered from oldest (index 0) to most
        recent (index -1)
        n is an integer

        returns a numeric array of the exponential
        moving average
        """
        ema = []
        j = 1

        # get n sma first and calculate the next n period ema
        sma = sum(s[:n]) / n
        multiplier = 2 / float(1 + n)
        ema.append(sma)

        # EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
        ema.append(((s[n] - sma) * multiplier) + sma)

        # now calculate the rest of the values
        for i in s[n + 1:]:
            tmp = ((i - ema[j]) * multiplier) + ema[j]
            j = j + 1
            ema.append(tmp)

        ema = Calc.__fill_ma_beg(ema, len(s), n)
        return ema

    @staticmethod
    def smma(closes, n=50):
        smma = []
        j = 0
        sma = sum(closes[:n]) / n
        smma.append(sma)

        for i in range(n, len(closes)):
            tmp = (smma[j] * n - smma[j] + closes[i]) / n
            j += 1
            smma.append(tmp)

        smma = Calc.__fill_ma_beg(smma, len(closes), n)
        return smma

    @staticmethod
    def __fill_ma_beg(mas, len, window):
        mas1 = list()
        for i in range(len):
            if i < window - 1:
                mas1.append(mas[0])
            else:
                mas1.append(mas[i - window + 1])
        return mas1

    @staticmethod
    def get_dif(mas1, mas2):
        s1 = np.array(mas1)
        s2 = np.array(mas2)
        d = s1 - s2
        return list(d)

    @staticmethod
    def is_dif_changed_to_neg(dif):
        last_dif = dif[-1]
        ret = False
        if last_dif < 0:
            for d in reversed(dif[:len(dif) - 1]):
                if d == 0:
                    continue
                elif d > 0:
                    return True
                else:
                    return False
        return ret

    @staticmethod
    def get_closes(chart_data):
        closes = [li['close'] for li in chart_data]
        return closes

    @staticmethod
    def get_high_low(chart_data):
        highs = [li['high'] for li in chart_data]
        lows = [li['low'] for li in chart_data]
        return highs, lows

    @staticmethod
    def macd(values):
        macd = np.subtract(Calc.ema(values, 12), Calc.ema(values, 26))
        sig = Calc.ema(macd, 9)
        hist = np.subtract(macd, sig)
        return macd, sig, hist

    @staticmethod
    def stoch(chart_data, N=14, m=3):
        closes = Calc.get_closes(chart_data)
        highs, lows = Calc.get_high_low(chart_data)
        percent_k = Calc.percent_k(closes, highs, lows)
        percent_d = Calc.percent_d(closes, highs, lows)
        return percent_k, percent_d

    @staticmethod
    def percent_k(closes, highs, lows, n=14, t=0):
        highs_t = highs if t == 0 else highs[:-t]
        lows_t = lows if t == 0 else lows[:-t]
        highest = max(highs_t[len(highs_t) - n:])  # максимум из последних n
        lowest = min(lows_t[len(lows_t) - n:])
        percent_k = (closes[::-1][t] - lowest) / (highest - lowest) * 100
        return percent_k

    @staticmethod
    def percent_d(closes, highs, lows, n=14, m=3):
        percent_d = 0
        for t in range(m):
            percent_d += Calc.percent_k(closes, highs, lows, n=n, t=t)
        percent_d = percent_d / m
        return percent_d

    @staticmethod
    def bb_last(closes, n=20, d=2):
        center = sum(closes[len(closes) - n:]) / n
        std = np.std(closes[len(closes) - n:])
        upper = center + d * std
        lower = center - d * std
        return lower, center, upper

    @staticmethod
    def stoch_all(chart_data, N=14, m=3):
        k_all = list()
        d_all = list()
        for i in range(len(chart_data)):
            j = len(chart_data) - i - 1
            if i < N:
                k_all.append(0)
                d_all.append(0)
            else:
                if j > 0:
                    cd = chart_data[:-j]
                else:
                    cd = chart_data
                percent_k, percent_d = Calc.stoch(cd, N, m)
                k_all.append(percent_k)
                d_all.append(percent_d)
        return k_all, d_all

    @staticmethod
    def vsa(candles, prev=0, lenght=200, divisor=3.6):
        # arr = np.array(candles[0][0])
        try:
            arr = np.array(candles)
            if prev > 0:
                arr = np.delete(arr, range(prev), axis=0)
            arr = arr[::-1]
            high = arr[:, 3]
            low = arr[:, 4]
            volume = arr[:, 5]
            close = arr[:, 2]
            open = arr[:, 1]
        except IndexError:
            return None

        range_ = high - low
        rangeAvg = Calc.sma(range_, lenght)
        volumeA = Calc.sma(volume, lenght)

        high1 = high[-2]
        low1 = low[-2]
        mid1 = (high1 + low1) / 2

        u1 = mid1 + (high1 - low1) / divisor
        d1 = mid1 - (high1 - low1) / divisor

        r_enabled1 = (range_[-1] > rangeAvg[-1]) and (close[-1] < d1) and volume[-1] > volumeA[-1]
        r_enabled2 = close[-1] < mid1
        r_enabled3 = close[-1] <= open[-1]
        r_enabled = (r_enabled1 or r_enabled2) and r_enabled3

        g_enabled1 = close[-1] > mid1
        g_enabled2 = (range_[-1] > rangeAvg[-1]) and (close[-1] > u1) and (volume[-1] > volumeA[-1])
        g_enabled3 = (high[-1] > high1) and (range_[-1] < rangeAvg[-1] / 1.5) and (volume[-1] < volumeA[-1])
        g_enabled4 = (low[-1] < low1) and (range_[-1] < rangeAvg[-1] / 1.5) and (volume[-1] > volumeA[-1])
        g_enabled5 = close[-1] >= open[-1]
        g_enabled = (g_enabled1 or g_enabled2 or g_enabled3 or g_enabled4) and g_enabled5

        gr_enabled1 = (range_[-1] > rangeAvg[-1]) and (close[-1] > d1) and (close[-1] < u1) and \
                      (volume[-1] > volumeA[-1]) and (volume[-1] < volumeA[-1] * 1.5) and (volume[-1] > volume[-2])
        gr_enabled2 = (range_[-1] < (rangeAvg[-1] / 1.5)) and (volume[-1] < (volumeA[-1] / 1.5))
        gr_enabled3 = (close[-1] > d1) and (close[-1] < u1)
        gr_enabled = gr_enabled1 or gr_enabled2 or gr_enabled3

        # v_color = gr_enabled ? 'gray': g_enabled ? 'green': r_enabled ? 'red': 'blue'
        try:
            if gr_enabled:
                v_color = 'gray'
            elif g_enabled:
                v_color = 'green'
            elif r_enabled:
                v_color = 'red'
            else:
                v_color = 'blue'
        except ValueError:
            return None

        return v_color

    @staticmethod
    def rsi(candles, prev=0, n=14):
        try:
            arr = np.array(candles)
            if prev > 0:
                arr = np.delete(arr, range(prev), axis=0)
            arr = arr[::-1]
            close = arr[:, 2]
            deltas = np.diff(close)
        except IndexError:
            return None

        seed = deltas[:n + 1]
        up = seed[seed >= 0].sum() / n
        down = -seed[seed < 0].sum() / n
        rs = up / down
        rsi = np.zeros_like(close)
        rsi[:n] = 100. - 100. / (1. + rs)

        for i in range(n, len(close)):
            delta = deltas[i - 1] 

            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (n - 1) + upval) / n
            down = (down * (n - 1) + downval) / n

            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)

        return rsi

    @staticmethod
    def true_range(candles):
        trs = list()
        arr = candles[::-1]
        for i in range(len(arr)):
            high = arr[i][3]
            low = arr[i][4]
            if i > 0:
                prev_close = arr[i][2]
            else:
                prev_close = arr[i - 1][2]
            tr = max(high - low, high - prev_close, prev_close - low)
            trs.append(tr)
        return trs

    @staticmethod
    def double_trend(candles, factor=3, ATR=12):
        atrs = Calc.average_true_range(candles, ATR)
        atrs_arr = np.array(atrs)
        arr = np.array(candles)
        arr = arr[::-1]
        high = arr[:, 3]
        low = arr[:, 4]
        close = arr[:, 2]

        hl2 = (high + low) / 2
        up = hl2 - (factor * atrs_arr)
        dn = hl2 + (factor * atrs_arr)

        tup = list()
        tdn = list()
        for i in range(len(arr)):
            if i > 0 and close[i - 1] > tup[i - 1]:
                tmp = max(up[i], tup[i - 1])
            else:
                tmp = up[i]
            tup.append(tmp)

        tmp = 0
        for i in range(len(arr)):
            if i > 0 and close[i - 1] < tdn[i - 1]:
                tmp = min(dn[i], tdn[i - 1])
            else:
                tmp = dn[i]
            tdn.append(tmp)

        trend = list()
        for i in range(len(arr)):
            if i > 0:
                if close[i] > tdn[i - 1]:
                    tmp = 1
                elif close[i] < tup[i - 1]:
                    tmp = -1
                else:
                    tmp = trend[i - 1]
            else:
                tmp = 1
            trend.append(tmp)

        return trend, tup, tdn

    @staticmethod
    def average_true_range(candles, length=14):
        trs = Calc.true_range(candles)
        atr = Calc.rma(trs, length)
        return atr

    @staticmethod
    def rma(closes, n=14):
        rma = []
        j = 0
        sma = sum(closes[:n]) / n
        rma.append(sma)

        for i in range(n, len(closes)):
            tmp = (rma[j] * (n - 1) + closes[i]) / n
            j += 1
            rma.append(tmp)

        rma = Calc.__fill_ma_beg(rma, len(closes), n)
        return rma

    @staticmethod
    def psar(candles, start=2, increment=2, maximum=2):
        start_calc = start * 0.01
        increment_calc = increment * 0.01
        maximum_calc = maximum * 0.1
        length = len(candles)
        arr = np.array(candles)
        arr = arr[::-1]
        high = arr[:, 3]
        low = arr[:, 4]
        close = arr[:, 2]

        psar = close[0:len(close)]
        psarbull = [None] * length
        psarbear = [None] * length
        bull = True
        af = start_calc
        hp = high[0]
        lp = low[0]
        for i in range(2, length):
            if bull:
                psar[i] = af * (hp - psar[i-1]) + psar[i-1]
            else:
                psar[i] = af * (lp - psar[i-1]) + psar[i-1]
            reverse = False
            if bull:
                if low[i] < psar[i]:
                    bull = False
                    reverse = True
                    psar[i] = hp
                    lp = low[i]
                    af = start_calc
            else:
                if high[i] > psar[i]:
                    bull = True
                    reverse = True
                    psar[i] = lp
                    hp = high[i]
                    af = start_calc

            if not reverse:
                if bull:
                    if high[i] > hp:
                        hp = high[i]
                        af = min(af + increment_calc, maximum_calc)
                    if low[i-1] < psar[i]:
                        psar[i] = low[i-1]
                    if low[i-2] < psar[i]:
                        psar[i] = low[i-2]
                else:
                    if low[i] < lp:
                        lp = low[i]
                        af = min(af + increment_calc, maximum_calc)
                    if high[i - 1] > psar[i]:
                        psar[i] = high[i - 1]
                    if high[i - 2] > psar[i]:
                        psar[i] = high[i - 2]

            if bull:
                psarbull[i] = psar[i]
            else:
                psarbear[i] = psar[i]

        return psarbull, psarbear