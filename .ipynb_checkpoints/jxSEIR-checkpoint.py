import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("ticks")


'''
Parameters:
    n -> population size
    q_s -> fraction susceptible
    r_0 -> R_0
    gamma -> infectious rate
    beta -> initial uncontrolled transmission rate
    sdf -> social distancing factor 
    sdf_t -> social distancing reaction time
    lf = lockdown factor
    lf_t = lockdown reaction time
    ef = easing factor
    ef_t = easing time 
    timesteps = time step in model
    tau -> infection rate
'''

def b_t(timesteps, sdf=0.2, sdf_t=10, lf=0.9, lf_t=15, ef=0.5, ef_t=100):
    '''
    returns a series containing the timeseries of beta_t values given:
        timesteps = array to use as index
        sdf = social distancing factor
        sdf_t = social distancing reaction time
        lf = lockdown factor
        lf_t = lockdown reaction time
        ef = easing factor
        ef_t = easing time 
    '''
    # initialise a working dataframe with all zero values
    df = pd.DataFrame(index=timesteps)
    df['social_distancing_factor'] = 0
    df['lockdown_factor'] = 0
    df['easing_factor'] = 0 
    
    # set values 
    df.loc[sdf_t:, 'social_distancing_factor'] = sdf
    df.loc[lf_t:, 'lockdown_factor'] = lf
    df.loc[ef_t:, 'easing_factor'] = ef
    
    # compute b_t
    df['b_t'] = (1 - df.social_distancing_factor) * (1 - df.lockdown_factor) * ((df.lockdown_factor*df.easing_factor)/(1-df.lockdown_factor) + 1)
    
    return df['b_t']

def tau_t(timesteps, tau, isolation_reaction_time=30):
    '''
    returns a series containing the timeseries of beta_t values given:
        timesteps = array to use as index
        tau = isolation effectiveness
        isolation_reaction_time = isolation reaction time
    '''
    # initialise df with all zero values 
    df = pd.DataFrame(index=timesteps)
    df['tau_t'] = 0
    
    # tau = tau only from tau_rxn_t onwards
    df.loc[isolation_reaction_time:, 'tau_t'] = tau
    
    return df['tau_t']

def imported_cases(timesteps, arrival_rate=3, import_reaction_time=30):
    '''
    returns a random series of arrivals for imported cases 
    following a poisson distribution with arrival rate = arrival_rate per day
    '''
    step = timesteps[1] - timesteps[0]
    size = import_reaction_time / step
    
    # initialise df
    df = pd.DataFrame(index=timesteps)
    df['imported'] = 0
    
    a = np.random.poisson(arrival_rate, int(size))
    
    # generate data
    for i in range(len(a)):
        df.loc[i*step:i*step, 'imported'] = a[i]
    
    df['imported'] = df.imported * step
    
    return df['imported']

def plot_summary(metric, ax, dateindex=False):
    '''
    Make plots and define formatting for each metric in summary dataframe 
    '''
    ax.set_title(metric.columns[0])
    index = metric.index
    values = metric.values
    
    # Plot line
    ax.plot(index, values, c='b')
    
    # Formatting
    if dateindex: # capability for dateindex not ready yet
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        ax.set_xlim(pd.Timestamp('2020-03-01'), result.index.get_level_values('date')[-1]+pd.Timedelta(days=1))
    
    ax.yaxis.tick_left()
    ax.margins(0.05)
    ax.grid(which='major', axis='y', c='k')
    ax.set_facecolor('w')

def seirs(init_vals, params, params2, params3, t, int_only=False, imports=False):
    '''
    Runs a SEIRS deterministic model
    Returns dataframe with the levels and key statistics indexed by timestep
    '''
    # SEIR initialisation, unpacking
    S_0, E_0, I_0, R_0, D_0 = init_vals 
    S, E, I, R, D = [S_0], [E_0], [I_0], [R_0], [D_0]
    sigma, beta, gamma, xi, n, mu = params 
    
    # unpack and compute timeseries parameters
    q_s, tau, isolation_reaction_time, sdf, sdf_t, lf, lf_t, ef, ef_t = params2
    bT = b_t(timesteps=t, sdf=sdf, sdf_t=sdf_t, lf=lf, lf_t=lf_t, ef=ef, ef_t=ef_t)
    tauT = tau_t(timesteps=t, tau=tau, isolation_reaction_time=isolation_reaction_time)
    rt_beta = beta * bT * q_s
    rt_infection_rate = rt_beta * (1 - tauT)
    
    # unpack and compute imported cases 
    arrival_rate, import_reaction_time = params3
    imported_cases_series = imported_cases(timesteps=t, arrival_rate=arrival_rate, import_reaction_time=import_reaction_time)
        
    # initialise additional statistics
    infections_daily_0 = 0
    reinfected_daily_0 = 0
    I_daily, R_daily = [infections_daily_0], [reinfected_daily_0]
    
    dt = t[1] - t[0]
    for i, _ in enumerate(t[1:]):
        #base SEIR model
        next_S = S[-1] - (rt_infection_rate[i*dt]*S[-1]*I[-1])*dt + (xi*R[-1])*dt
        next_E = E[-1] + (rt_infection_rate[i*dt]*S[-1]*I[-1] - sigma*E[-1])*dt
        next_I = I[-1] + (sigma*E[-1] - gamma*I[-1] - mu*I[-1])*dt 
        if imports:
            next_I += imported_cases_series[i*dt]/(n*q_s) 
        next_R = R[-1] + (gamma*I[-1])*dt - (xi*R[-1])*dt
        next_D = D[-1] + mu*dt*I[-1]*gamma
        
        # print(I[-1], sigma*E[-1]*dt, -gamma*I[-1]*dt, -mu*I[-1]*dt, sep=', ') # debugging time lol
        
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
        D.append(next_D)
        
        #additional stats
        next_I_d = sigma*E[-1]*dt
        next_R_d = (xi*R[-1])*dt
        
        I_daily.append(next_I_d)
        R_daily.append(next_R_d)

    # convert to pandas Series with indexing, 
    S_series = pd.Series(S, index=t)
    E_series = pd.Series(E, index=t)
    I_series = pd.Series(I, index=t)
    R_series = pd.Series(R, index=t)
    D_series = pd.Series(D, index=t)
    
    # calculate additional metrics
    I_daily_series = pd.Series(I_daily, index=t)
    I_cum = I_daily_series.cumsum()
    R_daily_series = pd.Series(R_daily, index=t)
    log_cum = np.log10(I_cum[1:]*n)
    log_cum = log_cum.append(pd.Series(np.nan, index=[t[0]])).sort_index()
    r_growth = I_cum.iloc[1:]/I_cum.iloc[:-1].values - 1
    r_growth = r_growth.append(pd.Series(np.nan, index=[t[0]])).sort_index()
    r_decay = I_daily_series.iloc[1:]/I_daily_series.iloc[:-1].values - 1
    r_decay = r_decay.append(pd.Series(np.nan, index=[t[0]])).sort_index()
    r_growth.replace(np.inf, np.nan, inplace=True)
    r_decay.replace(np.inf, np.nan, inplace=True)

    df = pd.DataFrame({
        'Susceptible population': S,
        'Exposed population': E,
        'Infected population': I,
        'Recovered population': R,
        'Deceased population': D,
        'Daily infections': I_daily_series,
        'Daily reinfections': R_daily_series,
        'Cumulative infections': I_cum,
        'Log10 cumulative infections': log_cum,
        'Rate of growth': r_growth,
        'Rate of decay': r_decay,
    })
    
    unit_people = ['Susceptible population', 
                   'Exposed population', 
                   'Infected population', 
                   'Recovered population', 
                   'Deceased population',
                   'Daily infections', 
                   'Cumulative infections',
                   'Daily reinfections'
                  ]
    
    for e in unit_people: # convert percentages back to no. of people
        df[e] = df[e] * n*q_s
        if int_only: # condition to return only integers
            df[e] = df[e].apply(np.ceil)
    
    return df

def seirs_interactive(n=5846993, I_0=1, incubation_period=4, infectious_period=7, r_0=3.8, mu=24/133, q_s=1, 
                      sdf=0.2, sdf_t=10, lf=0.9, lf_t=15, ef=0.5, ef_t=100, tau=0.75, isolation_reaction_time=30,
                      arrival_rate=1, import_reaction_time=15,
                      immunity_period=36500, t_max = 730, 
                      S=False, E=True, I=True, R=False, D=False, imports=True, extra=True):
    
    # pack initial values
    init_vals = 1-I_0/(n*q_s), 0, I_0/(n*q_s), 0, 0
    
    # configure and pack parameters
    sigma = 1/incubation_period
    gamma = 1/infectious_period
    beta = r_0*gamma
    xi = 1/immunity_period if immunity_period >= 1 else 0
    params = sigma, beta, gamma, xi, n, mu
    
    # pack additional parameters
    params2 = q_s, tau, isolation_reaction_time, sdf, sdf_t, lf, lf_t, ef, ef_t
    
    # pack imported cases metrics
    params3 = arrival_rate, import_reaction_time
    
    # setup the environment
    dt = 1 # timestep
    t = np.linspace(0, t_max, int(t_max/dt) + 1)
    
    # run the simulation and plot results
    results = seirs(init_vals, params, params2, params3, t, int_only=False, imports=True)
    col_names = ['Susceptible population', 'Exposed population', 'Infected population', 'Recovered population', 'Deceased population']
    boolist = [S, E, I, R, D]
    SEIRD_col = [i for (i, v) in zip(col_names, boolist) if v]
    ax = results.loc[:, SEIRD_col].plot(figsize=(14,6), legend=True, 
                                       title='SEIRS epidemic course')
    
    if extra:
        ncols = 4
        nrows = int(np.ceil(len(results.columns) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, nrows*4))
        for i, seriesname in enumerate(results.columns):
            plot_summary(results.loc[:, [seriesname]], axes.flat[i])
    
    return results
