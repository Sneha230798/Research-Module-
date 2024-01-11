import numpy as np
import pandas as pd

def generate_staggered_law_ma1_data(N, T, theta, num_individuals, mean=0, std_dev=1):
    # Generate random white noise for each individual
    white_noise = np.random.normal(mean, std_dev, size=(N, num_individuals, T))

    # Initialize the array to store the data
    data = np.zeros((N, num_individuals, T))

    # Generate the MA(1) process data for each individual
    for i in range(N):
        for j in range(num_individuals):
            for t in range(T):
                if t == 0:
                    data[i, j, t] = white_noise[i, j, t]
                else:
                    data[i, j, t] = white_noise[i, j, t] + theta * white_noise[i, j, t - 1]

    # The rest of the code remains the same for reshaping data and creating DataFrame

    reshaped_data = data.reshape((N * num_individuals, T))
    df = pd.DataFrame(reshaped_data, columns=[f'{t}' for t in range(T)])
    df['state'] = np.repeat(np.arange(1, N + 1), num_individuals)
    df['individual'] = np.tile(np.arange(1, num_individuals + 1), N)

    melted_df = pd.melt(df, id_vars=['state', 'individual'], var_name='time', value_name='value')
    melted_df['time'] = melted_df['time'].astype(int)
    data = melted_df.copy()
    data['time'] = data['time'].astype(int)

    # The rest of the treatment assignment code remains the same

    state_dummies = pd.get_dummies(data['state'], prefix='state', drop_first=True)
    time_dummies = pd.get_dummies(data['time'].astype(int), prefix='time', drop_first=True)
    data = pd.concat([data, state_dummies, time_dummies], axis=1)
    states = data['state'].unique()
    treatment_states = np.random.choice(states, size=len(states)//2, replace=False)
    treatment_years = np.random.choice(range(5, 15), size=len(treatment_states), replace=True)
    state_to_treatment_year = dict(zip(treatment_states, treatment_years))
    data['TREATMENT'] = data.apply(lambda x: 1 if x['state'] in treatment_states and x['time'] >= state_to_treatment_year[x['state']] else 0, axis=1)

    return data
