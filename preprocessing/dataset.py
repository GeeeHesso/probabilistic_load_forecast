import numpy as np
import pandas as pd
        
def load_dataframe_from_csv(path):
    dataframe = pd.read_csv(path, index_col=0)
    dataframe.index = pd.to_datetime(dataframe.index)
    dataframe.index = dataframe.index.floor('H')
    dataframe.index = dataframe.index.set_names('Timestamp')
    return dataframe
        
class Dataset:
    def __init__(self, data, scales=None, output_format='lag_last'):
        
        # Store output format
        format_options = {'lag_last', 'feature_last'}
        if not output_format in format_options:
            raise ValueError('''Output format must be 'lag_last' or 
                             'feature_last'.''')
        else:
            self.output_format = output_format
        
        # Instantiate scaler
        self.s = self._Scaler(self)
                
        # Initialize dictionaries
        self.data = {}
        self.scales = {}
        
        # Check if inputs are in a dictionary
        if not isinstance(data, dict):
            raise TypeError('''Data must a dictionary of string labels 
                            mapping to dataframes.''')
            
        # Store copies of data
        for label, frame in data.items():
            if isinstance(frame, pd.DataFrame):
                self.data[label] = frame.copy()
            else:
                raise TypeError('Data must be a dataframe')
        
        if scales is not None:
            
            # Check if inputs are in a dictionary
            if not isinstance(scales, dict):
                raise TypeError('''Data must a dictionary of string labels 
                                mapping to either series or scalars.''')
                      
            # Store copies of scales
            for label, scale in scales.items():
                if isinstance(scale, pd.Series):
                    self.scales[label] = scale.copy()
                elif isinstance(scale, (int, float)):
                    self.scales[label] = scale
                else:
                    raise TypeError('Scale must be either a series or scalar')
                
    # Scaler as inner class with analoguous getter and setter methods        
    class _Scaler:
        def __init__(self, parent_dataset):
            self._parent_dataset = parent_dataset
            
        def __call__(self, scaled_data, idx, scale_key):
            # Build new multiindex if data has more dimensions
            if scaled_data.ndim > len(idx.names) + 1:
                
                # Initialize multiindex length and lists of levels and names
                idx_len = scaled_data.shape[0]
                cols_len = scaled_data.shape[-1]
                if len(idx.names) == 1:
                    levels = [idx]
                else:
                    levels = list(idx.levels)
                names = ['Timestamp']
                
                # Iterate over all dimensions except the first and last
                for n, dim in enumerate(scaled_data.shape[1:-1]):
                    idx_len *= dim
                    levels.append(range(dim))
                    
                    # Interpret second dimension as axis
                    if n == 0:
                        names.append('Sample')
                        
                    # Leave any higher dimensions unnamed but numbered
                    else:
                        names.append(str(n))
                    
                    
                # Build multiindex
                idx = pd.MultiIndex.from_product(levels, names=names)
                
                # Reshape array
                scaled_data = scaled_data.reshape([idx_len, cols_len])
                
            # Build dataframe
            cols = self._parent_dataset.data[scale_key].columns
            scaled_data = pd.DataFrame(scaled_data, index=idx, columns=cols)
            
            # Unscale data
            scale = self._parent_dataset.scales.get(scale_key, 1)
            unscaled_data = scaled_data*scale
            
            return unscaled_data
            
        def __getitem__(self, key):
            if isinstance(key, str):
                unscaled_data = self._parent_dataset[key]
                scale = self._parent_dataset.scales.get(key, 1)
            elif isinstance(key, tuple) and len(key) == 2:
                unscaled_data = self._parent_dataset[key]
                scale = self._parent_dataset.scales.get(key[0], 1)
            else:
                raise TypeError('''Key must be either a string or a length
                               two tuple of a string and an index to be passed 
                               to the .loc indexer of the dataframe.''')
            
            # Scale the data and convert to array
            unscaled_data = unscaled_data.sort_index()
            scaled_data = (unscaled_data/scale).to_numpy()
            idx = unscaled_data.index
            
            # Check for multiindex
            if isinstance(unscaled_data.index, pd.MultiIndex):
                
                # Assert regular shape of dataframe
                n_timestamps = len(idx.levels[0])
                n_lags = len(idx.levels[1])
                if not n_timestamps*n_lags == len(idx):
                    raise ValueError('''Impossible to convert to
                                     regularly-shaped array: not all lags
                                     are available for all timestamps.''')
                
                # Build new shape of the output array
                new_shape = []
                for level in unscaled_data.index.levels:
                    new_shape.append(len(level))
                new_shape.append(len(unscaled_data.columns))
                
                # Reshape output array
                scaled_data = scaled_data.reshape(new_shape)
                
                if self._parent_dataset.output_format == 'lag_last':
                    scaled_data = np.transpose(scaled_data, [0, 2, 1])
                
                idx = idx.levels[0]

            return scaled_data, idx
        
        def __setitem__(self, key, value):
            if isinstance(value, tuple) and len(value) == 3:
                scaled_data, idx, scale_key = value
            else:
                raise TypeError('''Value must be a length three tuple
                                of an array, an index and key to obtain
                                columns and scale.''')           
            self._parent_dataset[key] = self(scaled_data, idx, scale_key)
            
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]
        elif isinstance(key, tuple) and len(key) == 2:
            label, idx = key
            return self.data[label].loc[idx]
        else:
            raise TypeError('''Key must be either a string or a length
                            two tuple of a string and an index to be passed 
                            to the .loc indexer of the dataframe.''')
                           
    def __setitem__(self, key, value):
        if isinstance(value, pd.DataFrame):
            self.data[key] = value
        elif isinstance(value, tuple) and len(value) == 2:
            frame, scale = value
            
            if isinstance(frame, pd.DataFrame):
                self.data[key] = frame.copy()
            else:
                raise TypeError('Data must be a dataframe')
                
            if isinstance(scale, pd.Series):
                self.scales[key] = scale.copy()
            elif isinstance(scale, (int, float)):
                self.scales[key] = scale
            else:
                raise TypeError('''Scale must be either a series or scalar''')
        else:
            raise TypeError('''Value must be either a dataframe or a length
                             two tuple of a dataframe and either a series
                             or scalar.''')
        
    def drop_nans(self, labels):
        for label, frame in self.data.items():
            if label in labels:
                frame.dropna(inplace=True)
                
                # If multiindex, remove unused levels
                if isinstance(frame.index, pd.MultiIndex):
                    frame.index = frame.index.remove_unused_levels()
    
    def fill_nans(self, labels, fill_value=0):        
        for label, frame in self.data.items():
            if label in labels:
                frame.fillna(value=fill_value, inplace=True)
                
    def drop_specific_timestamps(self, labels, timestamps):
        for label, frame in self.data.items():
            if label in labels:
                
                # If multiindex, select Timestamp level
                if isinstance(frame.index, pd.MultiIndex):
                    ts = frame.index.get_level_values('Timestamp').unique()
                    drop_ts = ts.intersection(timestamps)
                else:
                    drop_ts = timestamps
                    
                frame.drop(drop_ts, inplace=True)
                
                # If multiindex, remove unused levels
                if isinstance(frame.index, pd.MultiIndex):
                    frame.index = frame.index.remove_unused_levels()
    
    def drop_unshared_timestamps(self, labels):
        
        # Find shared timestamps
        shared_ts = None
        for label, frame in self.data.items():
            if label in labels:
                
                # Consider all time lags if multiindex
                if isinstance(frame.index, pd.MultiIndex):
                    lags = frame.index.get_level_values('Lag').unique()
                    for lag in lags:
                        ts = frame.xs(lag, level='Lag').index.unique()
                        
                        # Compute intersection between timestamps
                        if shared_ts is None:
                            shared_ts = ts
                        else:
                            shared_ts = shared_ts.intersection(ts)
                    
                # Directly compute intersection if not a multiindex
                else:
                    if shared_ts is None:
                        shared_ts = frame.index
                    else:
                        shared_ts = shared_ts.intersection(frame.index)
            
        # Select shared timestamps
        for label, frame in self.data.items():
            if label in labels:
                frame = frame.loc[shared_ts]
                
                # If multiindex, remove unused levels
                if isinstance(frame.index, pd.MultiIndex):
                    frame.index = frame.index.remove_unused_levels()
                    
                self.data[label] = frame
            
    def add_lag(self, labels, period, lags, 
                drop_nans=True, drop_unshared_timestamps=True):
        for label, frame in self.data.items():
            if label in labels:
                idx = frame.index
                
                # Add multiindex if not already present
                if not isinstance(idx, pd.MultiIndex):
                    if not isinstance(idx, pd.DatetimeIndex):
                        raise TypeError('''Index must be a datetime index 
                                        or a multiindex containing a
                                        datetime index and an index
                                        containing time lags.''')
                                        
                    new_idx = pd.MultiIndex.from_product([idx, ['0']],
                                                         names=['Timestamp',
                                                                'Lag'])
                    frame.index = new_idx
                    
                # Add lags only from zero lag values in the data
                for lag in lags:
                    lagged_frame = frame.xs('0', level='Lag')
                    lagged_frame = lagged_frame.shift(periods=lag,
                                                      freq=period)
                    new_idx = pd.MultiIndex.from_product([lagged_frame.index,
                                                          [f'{lag}{period}']],
                                                         names=['Timestamp',
                                                                'Lag'])
                    lagged_frame.index = new_idx
                    
                    frame = pd.concat([frame, lagged_frame])
                        
                self.data[label] = frame
                    
                if drop_nans:
                    self.drop_nans({label})
                        
                if drop_unshared_timestamps:
                    self.drop_unshared_timestamps({label})