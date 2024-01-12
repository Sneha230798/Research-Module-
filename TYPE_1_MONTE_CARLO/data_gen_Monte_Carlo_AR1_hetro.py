



import numpy as np
import pandas as pd

def generate_staggered_law_ar1_data_hetero(N, T, num_individuals, mean=0, std_dev=1):
    # Generate random white noise for each individual
    white_noise = np.random.normal(mean, std_dev, size=(N, num_individuals, T))

    # Initialize the array to store the data
    data = np.zeros((N, num_individuals, T))

    # Generate the AR(1) process data for each individual and state
    rhos = np.random.uniform(0.2, 0.8, size=N)  # Generate rhos for each state
    for i in range(N):
        rho = rhos[i]
        for j in range(num_individuals):
            for t in range(T):
                if t == 0:
                    data[i, j, t] = white_noise[i, j, t]
                else:
                    data[i, j, t] = rho * data[i, j, t - 1] + white_noise[i, j, t]

    # Reshape the data array for easier DataFrame creation
    reshaped_data = data.reshape((N * num_individuals, T))

    # Create a DataFrame with column names as time periods
    df = pd.DataFrame(reshaped_data, columns=[f'{t}' for t in range(T)])

    # Add a new 'state' column with repeated state values
    df['state'] = np.repeat(np.arange(1, N + 1), num_individuals)

    # Add a new 'individual' column with repeated individual values
    df['individual'] = np.tile(np.arange(1, num_individuals + 1), N)

    melted_df = pd.melt(df, id_vars=['state', 'individual'], var_name='time', value_name='value')

    # Convert the 'time' column to int
    melted_df['time'] = melted_df['time'].astype(int)

    data = melted_df.copy()

    data['time'] = data['time'].astype(int)
    # Create state dummy variables
    state_dummies = pd.get_dummies(data['state'], prefix='state', drop_first=True)

    # Convert state dummy variables to int
    state_dummies = state_dummies.astype(int)

    # Create time dummy variables
    time_dummies = pd.get_dummies(data['time'].astype(int), prefix='time', drop_first=True)

    # Convert time dummy variables to int
    time_dummies = time_dummies.astype(int)

    data = pd.concat([data, state_dummies, time_dummies], axis=1)

    states = data['state'].unique()

    # Randomly select half of the states to be in the treatment group
    treatment_states = np.random.choice(states, size=len(states) // 2, replace=False)

    # Assign treatment year to each treatment state, staggered between 1985 and 1995
    treatment_years = np.random.choice(range(5, 15), size=len(treatment_states), replace=True)
    state_to_treatment_year = dict(zip(treatment_states, treatment_years))

    # Add a treatment column to the DataFrame
    data['TREATMENT'] = data.apply(
        lambda x: 1 if x['state'] in treatment_states and x['time'] >= state_to_treatment_year[x['state']] else 0,
        axis=1)

    return data
