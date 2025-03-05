from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QFormLayout

from Orange.widgets import widget, gui, settings
from Orange.widgets.widget import Input, Output, Msg

from orangecontrib.timeseries import Timeseries
from orangecontrib.example.autogluon_integration import AutoGluonWrapper, convert_to_autogluon_format

class OWAutoGluonForecast(widget.OWWidget):
    name = "AutoGluon Forecast"
    description = "Automatic time series forecasting with AutoGluon"
    icon = "icons/AutoGluonForecast.svg"
    priority = 200
    
    class Inputs:
        time_series = Input("Time series", Timeseries)
        
    class Outputs:
        forecast = Output("Forecast", Timeseries)
        predictor = Output("Predictor", object, auto_summary=False)
        fitted_values = Output("Fitted values", Timeseries)
    
    class Error(widget.OWWidget.Error):
        no_time_variable = Msg('Data contains no time variable')
        no_target = Msg('Input time series does not contain a target variable')
        fitting_failed = Msg('Failed to fit the model: {}')
        
    class Warning(widget.OWWidget.Warning):
        data_size = Msg('Data has only {} instances')
    
    # Settings
    prediction_length = settings.Setting(24)
    eval_metric = settings.Setting("MASE")
    preset = settings.Setting("medium_quality")
    time_limit = settings.Setting(600)
    autocommit = settings.Setting(True)
    
    METRICS = ["MASE", "RMSE", "MAE", "MAPE", "SMAPE", "WAPE"]
    PRESETS = ["fast_training", "medium_quality", "high_quality", "best_quality"]
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.predictor = None
        
        # UI setup
        box = gui.vBox(self.controlArea, "Forecast Settings")
        
        form = QFormLayout()
        gui.widgetBox(box, orientation=form)
        
        self.pred_spin = gui.spin(
            box, self, "prediction_length", 1, 1000, 
            label="Forecast horizon:", 
            callback=self.apply.deferred
        )
        form.addRow("Forecast horizon:", self.pred_spin)
        
        self.metric_combo = gui.comboBox(
            box, self, "eval_metric", items=self.METRICS,
            callback=self.apply.deferred
        )
        form.addRow("Evaluation metric:", self.metric_combo)
        
        self.preset_combo = gui.comboBox(
            box, self, "preset", items=self.PRESETS,
            callback=self.apply.deferred
        )
        form.addRow("Training preset:", self.preset_combo)
        
        self.time_spin = gui.spin(
            box, self, "time_limit", 10, 3600, 
            label="Time limit (seconds):", 
            callback=self.apply.deferred
        )
        form.addRow("Time limit (seconds):", self.time_spin)
        
        self.autocommit_checkbox = gui.auto_commit(self.controlArea, self, "autocommit", "Apply", box=False)
        
    @Inputs.time_series
    def set_data(self, data):
        self.Error.clear()
        self.Warning.clear()
        
        self.data = None
        if data is not None:
            self.data = Timeseries.from_data_table(data)
            
            if not self.data.time_variable:
                self.Error.no_time_variable()
                self.data = None
                
            elif not self.data.domain.class_var:
                self.Error.no_target()
                self.data = None
                
            elif len(self.data) < 10:
                self.Warning.data_size(len(self.data))
        
        self.apply.now()
    
    @gui.deferred
    def apply(self):
        self.Error.clear()
        
        if self.data is None:
            self.Outputs.forecast.send(None)
            self.Outputs.predictor.send(None)
            self.Outputs.fitted_values.send(None)
            return
        
        try:
            # Создаем и обучаем модель
            self.predictor = AutoGluonWrapper(
                prediction_length=self.prediction_length,
                eval_metric=self.eval_metric,
                path="autogluon-timeseries-model",
                presets=self.preset,
                time_limit=self.time_limit
            )
            
            with self.progressBar(1) as progress:
                self.predictor.fit(self.data)
                progress.advance()
                
                # Генерируем прогноз
                forecast = self.predictor.predict(self.data)
                fitted_values = self.predictor.get_fitted_values(self.data)
                
                self.Outputs.forecast.send(forecast)
                self.Outputs.predictor.send(self.predictor)
                self.Outputs.fitted_values.send(fitted_values)
                
        except Exception as e:
            self.Error.fitting_failed(str(e))
            self.Outputs.forecast.send(None)
            self.Outputs.predictor.send(None)
            self.Outputs.fitted_values.send(None)
    
    def commit(self):
        """Выполнить прогнозирование и передать результаты."""
        if self.data is None:
            self.Outputs.forecast.send(None)
            return
        
        try:
            # Создаем и обучаем модель прогнозирования
            self.predictor = AutoGluonWrapper(
                prediction_length=self.prediction_length,
                eval_metric=self.eval_metric,
                path="autogluon-timeseries-model",
                presets=self.preset,
                time_limit=self.time_limit
            )
            
            self.predictor.fit(self.data)
            
            # Делаем прогноз
            forecast_data = self.predictor.predict(self.data)
            
            # Отправляем результат
            self.Outputs.forecast.send(forecast_data)
            self.Outputs.predictor.send(self.predictor)
            
            # Получаем fitted values
            fitted_values = self.predictor.get_fitted_values(self.data)
            self.Outputs.fitted_values.send(fitted_values)
            
        except Exception as e:
            self.Error.fitting_failed(str(e))
            self.Outputs.forecast.send(None)
            self.Outputs.predictor.send(None)
            self.Outputs.fitted_values.send(None)


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview
    from Orange.widgets.utils.owbasespinbox import DoubleSpinBox
    data = Timeseries.from_file('airpassengers')
    WidgetPreview(OWAutoGluonForecast).run(set_data=data)