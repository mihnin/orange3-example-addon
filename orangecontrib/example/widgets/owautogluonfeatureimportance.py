from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QFormLayout, QComboBox

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import Input, Output, Msg
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase

from orangecontrib.timeseries import Timeseries

class OWAutoGluonFeatureImportance(OWScatterPlotBase):
    name = "AutoGluon Feature Importance"
    description = "Visualize feature importance for AutoGluon TimeSeries models"
    icon = "icons/AutoGluonFeatureImportance.svg"
    priority = 220
    
    class Inputs:
        predictor = Input("Predictor", object)
        time_series = Input("Time series", Timeseries)
        
    class Outputs:
        feature_importance = Output("Feature importance", Table)
        selected_features = Output("Selected features", Table)
        
    class Error(OWScatterPlotBase.Error):
        invalid_predictor = Msg("Input is not a valid AutoGluon predictor")
        computation_failed = Msg("Feature importance computation failed: {}")
    
    # Settings
    method = settings.Setting("permutation")
    num_iterations = settings.Setting(5)
    subsample_size = settings.Setting(50)
    
    METHODS = ["permutation", "naive"]
    
    def __init__(self):
        super().__init__()
        
        self.predictor = None
        self.data = None
        self.importance_data = None
        
        # Control area setup
        box = gui.vBox(self.controlArea, "Feature Importance Method")
        
        form = QFormLayout()
        gui.widgetBox(box, orientation=form)
        
        self.method_combo = gui.comboBox(
            box, self, "method", items=self.METHODS,
            callback=self.compute_importance,
        )
        form.addRow("Method:", self.method_combo)
        
        self.iter_spin = gui.spin(
            box, self, "num_iterations", 1, 50, 
            label="Number of iterations:", 
            callback=self.compute_importance,
        )
        form.addRow("Number of iterations:", self.iter_spin)
        
        self.subsample_spin = gui.spin(
            box, self, "subsample_size", 10, 1000, 
            label="Subsample size:", 
            callback=self.compute_importance,
        )
        form.addRow("Subsample size:", self.subsample_spin)
        
    @Inputs.predictor
    def set_predictor(self, predictor):
        self.Error.clear()
        self.predictor = predictor
        
        if predictor is None:
            self.importance_data = None
            self.Outputs.feature_importance.send(None)
            return
            
        try:
            # Проверяем, что это действительно AutoGluon предиктор
            if not hasattr(predictor, 'predictor'):
                self.Error.invalid_predictor()
                self.predictor = None
                return
                
            if self.data is not None:
                self.compute_importance()
                
        except Exception as e:
            self.Error.invalid_predictor()
            self.predictor = None
            
    @Inputs.time_series
    def set_data(self, data):
        self.data = data
        if self.predictor is not None and self.data is not None:
            self.compute_importance()
            
    def compute_importance(self):
        if self.predictor is None or self.data is None:
            return
            
        self.Error.clear()
        
        try:
            # Вычисляем важность признаков
            importance = self.predictor.feature_importance(
                data=self.data,
                method=self.method,
                num_iterations=self.num_iterations,
                subsample_size=self.subsample_size,
            )
            
            # Преобразуем в таблицу Orange
            domain = Domain(
                [ContinuousVariable("Importance"), 
                 ContinuousVariable("StdDev")],
                [],
                [StringVariable("Feature")]
            )
            
            metas = [[feature] for feature in importance.index]
            X = [[importance.loc[feature, 'importance'], 
                  importance.loc[feature, 'stddev']] 
                 for feature in importance.index]
            
            self.importance_data = Table.from_numpy(domain, X, None, metas)
            self.Outputs.feature_importance.send(self.importance_data)
            
            # Обновляем график
            self.update_graph()
            
        except Exception as e:
            self.Error.computation_failed(str(e))
            
    def update_graph(self):
        """
        Обновляет график важности признаков
        """
        if self.importance_data is None:
            return
            
        # Использует функциональность OWScatterPlotBase для отображения графика
        # ...
        
    def selection_changed(self):
        """
        Обрабатывает выбор признаков на графике
        """
        if self.importance_data is None:
            return
            
        # Получаем выбранные признаки и отправляем их на выход
        # ...
        
if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview
    # Создаем фиктивный предиктор для теста
    WidgetPreview(OWAutoGluonFeatureImportance).run()