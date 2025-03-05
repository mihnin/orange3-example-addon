from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QTableView, QHeaderView

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import Input, Output, Msg, OWWidget
from Orange.widgets.utils.itemmodels import PyTableModel

from orangecontrib.timeseries import Timeseries

class OWAutoGluonLeaderboard(OWWidget):
    name = "AutoGluon Leaderboard"
    description = "Display model leaderboard for AutoGluon TimeSeries"
    icon = "icons/AutoGluonLeaderboard.svg"
    priority = 210
    
    class Inputs:
        predictor = Input("Predictor", object, auto_summary=False)
        time_series = Input("Time series", Timeseries)
        
    class Outputs:
        selected_model = Output("Selected model", object, auto_summary=False)
        evaluation_results = Output("Evaluation results", Table)
        
    class Error(OWWidget.Error):
        invalid_predictor = Msg("Input is not a valid AutoGluon predictor")
        evaluation_failed = Msg("Evaluation failed: {}")
    
    # Settings
    sorting = settings.Setting((0, Qt.AscendingOrder))
    selected_model = settings.Setting(None)
    
    def __init__(self):
        super().__init__()
        
        self.predictor = None
        self.test_data = None
        
        # Создаем таблицу
        self.table = QTableView()
        self.model = PyTableModel(parent=self.table)
        
        self.model.setHorizontalHeaderLabels(["Model", "Score", "Fit Time", "Inference Time"])
        
        self.table.setModel(self.model)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().sectionClicked.connect(self.sort_by_column)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.selectionModel().selectionChanged.connect(self.selection_changed)
        
        # Добавляем таблицу в главную область виджета
        self.mainArea.layout().addWidget(self.table)
        
    @Inputs.predictor
    def set_predictor(self, predictor):
        self.Error.clear()
        self.predictor = predictor
        
        if predictor is None:
            self.model.clear()
            self.Outputs.selected_model.send(None)
            self.Outputs.evaluation_results.send(None)
            return
        
        try:
            # Проверяем, что это действительно AutoGluon предиктор
            if not hasattr(predictor, 'predictor') or not hasattr(predictor.predictor, 'leaderboard'):
                self.Error.invalid_predictor()
                self.predictor = None
                return
                
            self.update_leaderboard()
            
        except Exception as e:
            self.Error.invalid_predictor()
            self.predictor = None
    
    @Inputs.time_series
    def set_data(self, data):
        self.test_data = data
        if self.predictor is not None:
            self.update_leaderboard()
    
    def update_leaderboard(self):
        if self.predictor is None:
            return
            
        try:
            # Получаем таблицу лидеров
            if self.test_data is not None:
                # Оцениваем на тестовых данных
                from orangecontrib.example.autogluon_integration import convert_to_autogluon_format
                ag_data = convert_to_autogluon_format(self.test_data)
                leaderboard = self.predictor.predictor.leaderboard(ag_data)
            else:
                # Используем встроенную таблицу лидеров
                leaderboard = self.predictor.predictor.leaderboard()
                
            # Заполняем модель
            self.model.clear()
            rows = []
            
            for _, row in leaderboard.iterrows():
                rows.append([
                    row['model'],
                    row['score_test' if self.test_data is not None else 'score_val'],
                    row.get('fit_time', 0),
                    row.get('pred_time_val', 0)
                ])
                
            self.model.wrap(rows)
            
            # Восстанавливаем сортировку
            self.sort_by_column(*self.sorting)
            
            # Создаем таблицу Orange для вывода
            domain = Domain(
                [ContinuousVariable("Score"), 
                 ContinuousVariable("Fit Time"),
                 ContinuousVariable("Inference Time")],
                [],
                [StringVariable("Model")]
            )
            
            metas = [[row[0]] for row in rows]
            X = [[row[1], row[2], row[3]] for row in rows]
            
            results_table = Table.from_numpy(domain, X, None, metas)
            self.Outputs.evaluation_results.send(results_table)
            
        except Exception as e:
            self.Error.evaluation_failed(str(e))
            
    def sort_by_column(self, column, order=None):
        if order is None:
            order = Qt.AscendingOrder if self.sorting[1] == Qt.DescendingOrder else Qt.DescendingOrder
            
        self.sorting = (column, order)
        self.model.sort(column, order)
        
    def selection_changed(self):
        indexes = self.table.selectionModel().selectedRows()
        if not indexes:
            self.Outputs.selected_model.send(None)
            return
            
        model_name = self.model[indexes[0].row()][0]
        self.selected_model = model_name
        
        if self.predictor is not None:
            # Получаем выбранную модель
            try:
                selected_model = self.predictor.get_model(model_name)
                self.Outputs.selected_model.send(selected_model)
            except:
                self.Error.evaluation_failed(f"Could not access model {model_name}")
                self.Outputs.selected_model.send(None)
            

if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview
    # Создаем фиктивный предиктор для теста
    WidgetPreview(OWAutoGluonLeaderboard).run()