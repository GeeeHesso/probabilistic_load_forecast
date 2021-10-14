import pandas as pd
import numpy as np
from datetime import datetime
from holidays import CountryHoliday

def get_temporal_features(index, features):
    temporal_features = pd.DataFrame(index=index)
    if 'hour' in features:
        hour = index.hour
        temporal_features['hour'] = hour.astype(int)
    if 'weekday' in features:
        weekday = index.weekday
        temporal_features['weekday'] = weekday.astype(int)
    if 'month' in features:
        month = index.month
        temporal_features['month'] = month.astype(int)
    if 'dayofyear' in features:
        dayofyear = index.dayofyear
        temporal_features['dayofyear'] = dayofyear.astype(int)
    if 'holiday' in features:
        ch_holidays = CountryHoliday('CH')
        holiday = pd.Series(index.date, 
                            index=index).apply(lambda x: x in ch_holidays)
        temporal_features['holiday'] = holiday.astype(float)
    return temporal_features

def get_temporal_features_one_hot(index, features):
    temporal_features = pd.DataFrame(index=index)
    if 'hour' in features:
        hour = index.hour
        hour_sin = np.sin((2/24)*np.pi*hour).astype(float)
        hour_cos = np.cos((2/24)*np.pi*hour).astype(float)
        temporal_features['hour_sin'] = hour_sin
        temporal_features['hour_cos'] = hour_cos
    if 'weekday' in features:
        weekday = index.weekday
        weekdays = pd.get_dummies(weekday, prefix='weekday').astype(float)
        weekdays.index = temporal_features.index
        temporal_features = pd.concat([temporal_features, weekdays], axis=1)
    if 'month' in features:
        month = index.month
        months = pd.get_dummies(month, prefix='month').astype(float)
        months.index = temporal_features.index
        temporal_features = pd.concat([temporal_features, months], axis=1)
    if 'dayofyear' in features:
        dayofyear = index.dayofyear
        dayofyear_sin = np.sin((2/365)*np.pi*dayofyear).astype(float)
        dayofyear_cos = np.cos((2/365)*np.pi*dayofyear).astype(float)
        temporal_features['dayofyear_sin'] = dayofyear_sin
        temporal_features['dayofyear_cos'] = dayofyear_cos
    if 'holiday' in features:
        ch_holidays = CountryHoliday('CH')
        holiday = pd.Series(index.date, 
                            index=index).apply(lambda x: x in ch_holidays)
        temporal_features['holiday'] = holiday.astype(float)
    return temporal_features

def get_temporal_features_fourier(index, features, 
                                  n_hourofweek=42,
                                  n_hour=6,
                                  n_weekday=3,
                                  n_dayofyear=1, 
                                  n_month=1,
                                  n_ch_holiday=1,
                                  n_ch_break=2):
    temporal_features = pd.DataFrame(index=index)
    if 'hourofweek' in features:
        hour = index.hour
        weekday = index.weekday
        hourofweek = 24*weekday + hour
        for n in range(n_hourofweek):
            sin = np.sin((n+1)/168*2*np.pi*hourofweek).astype(float)
            cos = np.cos((n+1)/168*2*np.pi*hourofweek).astype(float)
            temporal_features['hourofweek_sin_{}'.format(n+1)] = sin
            temporal_features['hourofweek_cos_{}'.format(n+1)] = cos
    else:
        if 'hour' in features:
            hour = index.hour
            for n in range(n_hour):
                sin = np.sin((n+1)/24*2*np.pi*hour).astype(float)
                cos = np.cos((n+1)/24*2*np.pi*hour).astype(float)
                temporal_features['hour_sin_{}'.format(n+1)] = sin
                temporal_features['hour_cos_{}'.format(n+1)] = cos
        if 'weekday' in features:
            weekday = index.weekday
            for n in range(n_weekday):
                sin = np.sin((n+1)/7*2*np.pi*weekday).astype(float)
                cos = np.cos((n+1)/7*2*np.pi*weekday).astype(float)
                temporal_features['weekday_sin_{}'.format(n+1)] = sin
                temporal_features['weekday_cos_{}'.format(n+1)] = cos
    if 'dayofyear' in features:
        dayofyear = index.dayofyear
        for n in range(n_dayofyear):
            sin = np.sin((n+1)/365*2*np.pi*dayofyear).astype(float)
            cos = np.cos((n+1)/365*2*np.pi*dayofyear).astype(float)
            temporal_features['dayofyear_sin_{}'.format(n+1)] = sin
            temporal_features['dayofyear_cos_{}'.format(n+1)] = cos
    else:
        if 'month' in features:
            month = index.month
            for n in range(n_month):
                sin = np.sin((n+1)/7*2*np.pi*month).astype(float)
                cos = np.cos((n+1)/7*2*np.pi*month).astype(float)
                temporal_features['month_sin_{}'.format(n+1)] = sin
                temporal_features['month_cos_{}'.format(n+1)] = cos
    if 'trend' in features:
        trend = np.arange(len(index))/len(index)
        temporal_features['trend'] = trend
    if 'ch_holiday' in features:
        ch_holidays = CountryHoliday('CH')
        holiday = pd.Series(index.date, 
                            index=index).apply(lambda x: x in ch_holidays)
        hour = index.hour
        for n in range(n_ch_holiday):
            sin = (holiday*np.sin((n+1)/24*2*np.pi*hour)).astype(float)
            cos = (holiday*np.cos((n+1)/24*2*np.pi*hour)).astype(float)
            temporal_features['ch_holiday_sin_{}'.format(n+1)] = sin
            temporal_features['ch_holiday_cos_{}'.format(n+1)] = cos
            temporal_features['ch_holiday_constant'] = holiday.astype(float)
    if 'ch_break' in features:
        freq = 'H'
        break_dates = [(datetime(2015, 12, 19), datetime(2016, 1, 11)),
                       (datetime(2016, 12, 24), datetime(2017, 1, 9)),
                       (datetime(2017, 12, 23), datetime(2018, 1, 8)),
                       (datetime(2018, 12, 22), datetime(2019, 1, 7)),
                       (datetime(2019, 12, 21), datetime(2020, 1, 6)),
                       (datetime(2020, 12, 19), datetime(2021, 1, 4)),
                       (datetime(2021, 12, 18), datetime(2022, 1, 10)),
                       (datetime(2022, 12, 24), datetime(2023, 1, 9)),
                       (datetime(2023, 12, 23), datetime(2024, 1, 8))]
        
        # Build constant feature for break
        break_constant = pd.Series(0, index=index)
        for break_ in break_dates:
            phase_index = pd.date_range(break_[0], break_[-1], freq=freq)
            mask = [i in phase_index for i in index.round(freq)]
            timestamps = index[mask]
            break_constant[timestamps] = 1
        temporal_features['ch_break_constant'] = break_constant.astype(float)
        
        # Build sine and cosine features for break
        for n in range(n_ch_break):
            break_sin = pd.Series(0, index=index)
            break_cos = pd.Series(0, index=index)
            for break_ in break_dates:
                phase_index = pd.date_range(break_[0], break_[-1], freq=freq)
                phase = pd.Series(np.arange(len(phase_index))/len(phase_index), 
                                  index=phase_index)          
                mask = [i in phase_index for i in index.round(freq)]
                timestamps = index[mask]
                phase = phase[timestamps.round(freq)].values
                sin = np.sin((n+1)*2*np.pi*phase)
                cos = np.cos((n+1)*2*np.pi*phase)
                break_sin[timestamps] = sin
                break_cos[timestamps] = cos
            break_sin = break_sin.astype(float)
            break_cos = break_cos.astype(float)
            temporal_features['ch_break_sin_{}'.format(n+1)] = break_sin
            temporal_features['ch_break_cos_{}'.format(n+1)] = break_cos
        
    return temporal_features

