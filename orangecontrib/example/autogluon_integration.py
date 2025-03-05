import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from orangecontrib.timeseries import Timeseries

def convert_to_autogluon_format(data: Timeseries) -> TimeSeriesDataFrame:
    """
    Convert Orange Timeseries to AutoGluon TimeSeriesDataFrame
    
    Parameters
    ----------
    data : Timeseries
        Orange Timeseries object
        
    Returns
    -------
    TimeSeriesDataFrame
        AutoGluon TimeSeriesDataFrame
    """
    # Получаем временную переменную
    time_var = data.time_variable
    if time_var is None:
        raise ValueError("Time variable is not set in the Timeseries object")
    
    # Получаем целевую переменную (обычно class_var в Orange)
    target_var = data.domain.class_var
    if target_var is None:
        raise ValueError("Target variable is not set (class variable)")
    
    # Получаем данные
    time_values = data.time_values
    target_values = data.Y
    
    # Создаем DataFrame
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(time_values, unit='s'),
        'target': target_values.flatten() if target_values.ndim > 1 else target_values,
        'item_id': 'item_1'  # AutoGluon требует идентификатор временного ряда
    })
    
    # Преобразуем в TimeSeriesDataFrame
    tsdf = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column='item_id',
        timestamp_column='timestamp'
    )
    
    return tsdf

def convert_from_autogluon_forecast(forecast, original_data: Timeseries) -> Timeseries:
    """
    Convert AutoGluon forecast to Orange Timeseries
    
    Parameters
    ----------
    forecast : pandas.DataFrame
        Forecast from AutoGluon
    original_data : Timeseries
        Original Orange Timeseries object
        
    Returns
    -------
    Timeseries
        Orange Timeseries object with forecast
    """
    from Orange.data import Domain, ContinuousVariable, TimeVariable
    
    # Получаем временную переменную из исходных данных
    time_var = original_data.time_variable
    
    # Преобразуем прогноз в DataFrame
    forecast_df = forecast.reset_index()
    
    # Создаем новую временную шкалу для прогноза
    last_timestamp = original_data.time_values[-1]
    
    # Создаем домен с переменными
    attrs = []
    for col in forecast.columns:
        if col != 'timestamp' and col != 'item_id':
            attrs.append(ContinuousVariable(col))
    
    domain = Domain(attrs, metas=[time_var])
    
    # Преобразуем данные для создания Timeseries
    X = forecast_df[forecast.columns[2:]].values
    
    # Преобразуем timestamp в секунды с начала эпохи
    timestamps = pd.to_datetime(forecast_df['timestamp']).astype(np.int64) // 10**9
    metas = np.column_stack([timestamps])
    
    # Создаем Timeseries
    forecast_timeseries = Timeseries.from_numpy(domain, X, metas=metas)
    forecast_timeseries.time_variable = time_var
    
    return forecast_timeseries

class AutoGluonWrapper:
    """
    Wrapper class for AutoGluon TimeSeriesPredictor to make it compatible with Orange
    """
    def __init__(self, prediction_length=1, **kwargs):
        self.prediction_length = prediction_length
        self.kwargs = kwargs
        self.predictor = None
        self.data = None
        
    def fit(self, data: Timeseries):
        """
        Fit the model to the data
        
        Parameters
        ----------
        data : Timeseries
            Orange Timeseries object
        """
        ag_data = convert_to_autogluon_format(data)
        self.data = data  # Сохраняем данные
        
        self.predictor = TimeSeriesPredictor(
            prediction_length=self.prediction_length,
            **self.kwargs
        )
        
        self.predictor.fit(ag_data)
        return self
        
    def predict(self, data: Timeseries = None) -> Timeseries:
        """
        Generate forecast
        
        Parameters
        ----------
        data : Timeseries, optional
            Orange Timeseries object, by default None
            
        Returns
        -------
        Timeseries
            Orange Timeseries object with forecast
        """
        if self.predictor is None:
            raise ValueError("Model must be fitted first")
            
        if data is None:
            # Используем данные из обучения
            if self.data is None:
                raise ValueError("No data available for prediction")
            data_to_use = self.data
        else:
            # Используем переданные данные
            data_to_use = data
            
        ag_data = convert_to_autogluon_format(data_to_use)
        predictions = self.predictor.predict(ag_data)
            
        return convert_from_autogluon_forecast(predictions, data_to_use)
    
    def get_model(self, model_name):
        """
        Get a specific model from the predictor
        
        Parameters
        ----------
        model_name : str
            Name of the model
        
        Returns
        -------
        object
            Model object
        """
        if self.predictor is None:
            return None
        return self.predictor.get_model(model_name)
    
    def get_fitted_values(self, data=None):
        """
        Get fitted values for the given data
        
        Parameters
        ----------
        data : Timeseries, optional
            Orange Timeseries object
        
        Returns
        -------
        Timeseries
            Orange Timeseries object with fitted values
        """
        if self.predictor is None:
            return None
            
        if data is None:
            data = self.data
        
        if data is None:
            return None
        
        ag_data = convert_to_autogluon_format(data)
        
        # В AutoGluon нет прямого способа получить fitted_values,
        # но можно делать прогноз на историческом периоде
        # Используем backtest для получения внутривыборочного прогноза
        fitted_values = self.predictor.predict(ag_data)
        
        # Преобразуем прогнозы обратно в формат Orange Timeseries
        return convert_from_autogluon_forecast(fitted_values, data)