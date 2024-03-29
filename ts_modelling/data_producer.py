import random

import numpy as np


class DataProducer:
    def __init__(self, length, n_vars, path):
        self.length = length
        self.n_vars = n_vars
        self.path = path
        self.data = np.zeros([length, n_vars])

    def add_rw(self, amplitude):
        self.data += self._generate_random_walk(self.length, self.n_vars, amplitude)

    def _generate_random_walk(self, nbr_steps, n_vars, amplitude):
        steps = np.random.choice([-amplitude, amplitude], size=[nbr_steps, n_vars])
        random_walk = np.cumsum(steps)
        return random_walk

    def add_arma(self, ar_components, ma_components):
        for var in range(self.n_vars):
            self.data[var] += self._generate_arma(
                nbr_steps=self.length,
                ar_components=ar_components,
                ma_components=ma_components
            )

    def _generate_arma(self, nbr_steps, ar_components, ma_components):
        from statsmodels.tsa import arima_process
        import numpy as np
        arma = 0

        ar_components = ar_components or []
        ma_components = ma_components or []

        if len(ar_components) > 0 or len(ma_components) > 0:
            ma_components = np.array(ma_components)
            ma = np.r_[1, ma_components]  # add zero-lag

            ar_components = np.array(ar_components)
            ar = np.r_[1, -ar_components]  # add zero-lag and negate

            arma = arima_process.arma_generate_sample(
                ar=ar,
                ma=ma,
                nsample=nbr_steps,
                burnin=500,
            )
        return arma

    def add_sine(self, freq_amp: [(float,float)]):
        for var in range(self.n_vars):
            self.data += self._generate_sine(freq_amp)


    def _generate_sine(self, freq_amp: [(float,float)]):
        time_array = np.linspace(0, self.length, self.length)
        sine_component = np.zeros(self.length)
        for freq, amp in freq_amp:
            sine_component += amp * np.sin((time_array * 2 * np.pi) * freq + random.uniform(0, 1) * 2 * np.pi)
        return sine_component




    def generate_data(self,
                      root_path, data_file_name, plot=True,
                      nbr_days=7 * 52,
                      random_walk_amplitude=0, ar_components=[], ma_components=[],
                      sine=[(0, 0)], trends=0,
                      pure_noise_amp=0,
                      ):
        """Generates nbr_days of hourly sample data with different flavours
        root_path: path from root to directory, './datasets'
        data_file_name: name of the created data file, 'data_file_name.csv'
        random_walk_amplitude: the maximum "step" the rw process can take, y_{t+1} = y_{t}+U(-1,1)*random_walk_amplitude
        ar_-, ma_components: the coefficients in an ar/ma-process
        sine: an array of (frequency,amplitude) tuples for sines. For example: weekly period with amplitude 1: [(1/(24*7),1)]
        trends: number of different trend lines, change points are randomly distributed, slop is random between (-50,50)/total length
        pure_noise_amp: amplitude of pure noise component
        """
        import pandas as pd
        import numpy as np
        import random
        import os
        from matplotlib import pyplot as plt

        start_date = '2024-01-01'
        nbr_hours = 24 * nbr_days
        dates = pd.date_range(start=start_date, periods=nbr_hours, freq='H')

        time_array = np.linspace(0, nbr_hours, nbr_hours)

        ## sine
        sine_component = np.zeros(nbr_hours)
        if sine:
            for freq, amp in sine:
                sine_wave = amp * np.sin(((time_array) * 2 * np.pi) * freq + random.uniform(0, 1) * 2 * np.pi)
                sine_component += sine_wave

        ## random walk
        rw = generate_random_walk(nbr_hours, random_walk_amplitude)

        ## trend
        trend_amp = 3 / (24 * 7)  # three units per week (we want substantial change during one prediction window)
        trend = 0
        if trends > 0:
            trend = np.zeros(nbr_hours)
            breaks = random.sample(range(1, nbr_hours - 1), trends - 1)
            breaks.sort()

            idx = 0
            for b in breaks:
                trend[idx:b] = trend[idx] + trend_amp * random.uniform(-1, 1) * np.arange(b - idx)
                idx = b - 1

            trend[idx:] = trend[idx] + trend_amp * random.uniform(-1, 1) * np.arange(nbr_hours - idx)

        ## arma
        arma = generate_arma(nbr_steps=nbr_hours, ar_components=ar_components, ma_components=ma_components)

        ## pure noise
        noise = pure_noise_amp * np.random.normal(
            loc=0, scale=1, size=nbr_hours
        )

        signal = sine_component + rw + trend + noise + arma

        df = pd.DataFrame({'date': dates, 'signal': signal})
        if plot:
            plt.plot(df['date'][:24 * 7 * 3], df['signal'][:24 * 7 * 3])
            plt.title('First three weeks')
            plt.xlabel('Date')
            plt.ylabel('Signal')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            plt.plot(df['date'], df['signal'])
            plt.title('Full set')
            plt.xlabel('Date')
            plt.ylabel('Signal')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        os.makedirs(root_path, exist_ok=True)
        data_path = os.path.join(root_path, data_file_name)
        df.to_csv(data_path, index=False)
