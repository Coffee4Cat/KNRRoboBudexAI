import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, 
                               QVBoxLayout, QWidget, QHBoxLayout, QLabel,
                               QPushButton, QSpinBox, QGroupBox, QScrollArea)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineScript, QWebEnginePage
from PySide6.QtCore import QUrl, Qt, Slot, QObject
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWebChannel import QWebChannel
import plotly.graph_objects as go
import json
import torch
import torch.nn as nn



class RankNet(nn.Module):
    def __init__(self, input_dim):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.model(x)
input_dim = 5    
model = RankNet(input_dim=input_dim)
model.load_state_dict(torch.load("usa_model_200_400.pt", map_location="cpu"))
model.eval()


class WebChannelHandler(QObject):
    """Obsługuje komunikację między JavaScript a Qt"""
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
    
    @Slot(str)
    def handleClick(self, data_json):
        """Odbiera dane z JavaScript"""
        try:
            data = json.loads(data_json)
            if self.callback:
                self.callback(data)
        except Exception as e:
            print(f"Error handling click: {e}")


class PlotlyWidget(QWebEngineView):
    """Widget do wyświetlania wykresów Plotly"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 400)
        self.setMaximumHeight(500)
        self.click_callback = None
        self.channel = QWebChannel()
        self.handler = None
        
    def set_figure(self, fig, longitudes, latitudes, predictions):
        """Ustawia wykres Plotly w widoku"""
        # Zapisujemy dane do użycia w callback
        self.longitudes = longitudes
        self.latitudes = latitudes
        self.predictions = predictions
        
        # Tworzymy handler z callbackiem
        self.handler = WebChannelHandler(self.on_js_click)
        self.channel.registerObject('handler', self.handler)
        self.page().setWebChannel(self.channel)
        
        html = fig.to_html(include_plotlyjs='cdn')
        
        # Dodajemy QWebChannel i kod do obsługi kliknięć
        html = html.replace('</head>', '''
        <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
        <script>
        var qt_handler = null;
        
        new QWebChannel(qt.webChannelTransport, function(channel) {
            qt_handler = channel.objects.handler;
        });
        </script>
        </head>''')
        
        html = html.replace('</body>', '''
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
            if (plotDiv) {
                plotDiv.on('plotly_click', function(data) {
                    var point = data.points[0];
                    var clickData = {
                        lon: point.lon,
                        lat: point.lat,
                        pointIndex: point.pointIndex
                    };
                    
                    // Czekamy aż handler będzie gotowy
                    var sendData = function() {
                        if (qt_handler) {
                            qt_handler.handleClick(JSON.stringify(clickData));
                        } else {
                            setTimeout(sendData, 100);
                        }
                    };
                    sendData();
                });
            }
        });
        </script>
        </body>''')
        
        self.setHtml(html)
    
    def on_js_click(self, data):
        """Przetwarza dane z JavaScript i wywołuje callback"""
        try:
            point_index = data.get('pointIndex', 0)
            lon = data.get('lon')
            lat = data.get('lat')
            
            # Pobieramy wartość z naszych zapisanych danych
            if point_index < len(self.predictions):
                value = self.predictions[point_index]
                
                click_data = {
                    'lon': lon,
                    'lat': lat,
                    'value': value,
                    'index': point_index
                }
                
                if self.click_callback:
                    self.click_callback(click_data)
        except Exception as e:
            print(f"Error in on_js_click: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wizualizacja Predykcji - USA")
        self.setGeometry(100, 100, 1400, 900)
        
        # Dane dla wszystkich modeli
        self.models_data = {}
        self.optimized_points = {}  # Przechowujemy punkty optymalizacji osobno
        self.selected_coords = None
        self.plot_widgets = {}  # Przechowujemy referencje do widgetów
        
        # Ustawienie ciemnego motywu
        self.setup_dark_theme()
        
        # Tworzenie głównego widgetu
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout główny
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Tworzenie zakładek z wykresami
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        main_layout.addWidget(self.tab_widget)
        
        # Sekcja z informacjami o klikniętym punkcie
        self.create_info_section(main_layout)
        
        # Sekcja z optymalizacją
        self.create_optimization_section(main_layout)
        
        # Dodawanie zakładek z wykresami
        self.add_plot_tabs()
        
    def setup_dark_theme(self):
        """Ustawia ciemny motyw aplikacji"""
        palette = QPalette()
        
        # Ciemne kolory
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(40, 40, 40))
        palette.setColor(QPalette.ToolTipBase, QColor(220, 220, 220))
        palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.Button, QColor(40, 40, 40))
        palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        
        self.setPalette(palette)
        
        # Style dla zakładek i innych elementów
        self.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3a3a3a;
                background: #1e1e1e;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #2a2a2a;
                color: #dcdcdc;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background: #1e1e1e;
                color: #2a82da;
                border-bottom: 2px solid #2a82da;
            }
            QTabBar::tab:hover {
                background: #353535;
            }
            QGroupBox {
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #dcdcdc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
            QPushButton {
                background: #2a82da;
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #3a92ea;
            }
            QPushButton:pressed {
                background: #1a72ca;
            }
            QPushButton:disabled {
                background: #4a4a4a;
                color: #888888;
            }
            QSpinBox {
                background: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 5px;
                color: #dcdcdc;
            }
            QLabel {
                color: #dcdcdc;
            }
        """)
    
    def create_info_section(self, parent_layout):
        """Tworzy sekcję z informacjami o klikniętym punkcie"""
        info_group = QGroupBox("Informacje o wybranym punkcie")
        info_layout = QVBoxLayout()
        
        self.coords_label = QLabel("Kliknij punkt na mapie aby zobaczyć szczegóły")
        self.coords_label.setStyleSheet("font-size: 14px; padding: 10px;")
        info_layout.addWidget(self.coords_label)
        
        # Scroll area dla wartości z wszystkich modeli
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(150)
        scroll_widget = QWidget()
        self.values_layout = QVBoxLayout(scroll_widget)
        self.values_layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(scroll_widget)
        info_layout.addWidget(scroll)
        
        info_group.setLayout(info_layout)
        parent_layout.addWidget(info_group)
    
    def create_optimization_section(self, parent_layout):
        """Tworzy sekcję z optymalizacją"""
        opt_group = QGroupBox("Optymalizacja")
        opt_layout = QHBoxLayout()
        
        opt_label = QLabel("Liczba punktów do dodania:")
        opt_layout.addWidget(opt_label)
        
        self.point_count_spinbox = QSpinBox()
        self.point_count_spinbox.setMinimum(1)
        self.point_count_spinbox.setMaximum(1000)
        self.point_count_spinbox.setValue(10)
        self.point_count_spinbox.setMinimumWidth(100)
        opt_layout.addWidget(self.point_count_spinbox)
        
        self.optimize_button = QPushButton("Optimize")
        self.optimize_button.clicked.connect(self.optimize_points)
        self.optimize_button.setEnabled(False)
        opt_layout.addWidget(self.optimize_button)
        
        opt_layout.addStretch()
        
        opt_group.setLayout(opt_layout)
        parent_layout.addWidget(opt_group)
    
    def create_sample_data(self, seed=42):
        """Tworzy przykładowe dane dla wykresów"""
        np.random.seed(seed)
        n_points = 1000
        
        # Generowanie losowych współrzędnych w granicach USA
        longitudes = np.random.uniform(-125, -65, n_points)
        latitudes = np.random.uniform(25, 50, n_points)
        predictions = np.random.uniform(-3, 7, n_points)
        
        return longitudes, latitudes, predictions
    
    def create_geo_plot(self, longitudes, latitudes, predictions, title, opt_lons=None, opt_lats=None, opt_preds=None):
        """Tworzy wykres geograficzny USA"""
        # Podstawowe punkty danych
        scatter = go.Scattergeo(
            lon=longitudes,
            lat=latitudes,
            text=[f"Score: {v:.2f}" for v in predictions],
            marker=dict(
                size=4,
                color=predictions,
                cmin=-3,
                cmax=7,
                colorscale='Viridis',
                colorbar=dict(
                    title='Predicted Score',
                    # titlefont=dict(color='white'),
                    tickfont=dict(color='white')
                ),
                line=dict(width=0)
            ),
            mode='markers',
            hovertemplate='<b>Lon:</b> %{lon:.2f}<br><b>Lat:</b> %{lat:.2f}<br><b>Score:</b> %{text}<extra></extra>',
            name='Original Data'
        )
        
        data = [scatter]
        
        # Jeśli są punkty optymalizacji, dodajemy je jako osobną warstwę
        if opt_lons is not None and len(opt_lons) > 0:
            scatter_opt = go.Scattergeo(
                lon=opt_lons,
                lat=opt_lats,
                text=[f"Score: {v:.2f}" for v in opt_preds],
                marker=dict(
                    size=6,
                    color='#FF6B6B',  # Czerwony kolor
                    line=dict(width=1, color='white')
                ),
                mode='markers',
                hovertemplate='<b>Optimized Point</b><br><b>Lon:</b> %{lon:.2f}<br><b>Lat:</b> %{lat:.2f}<br><b>Score:</b> %{text}<extra></extra>',
                name='Optimized Points'
            )
            data.append(scatter_opt)
        
        layout = go.Layout(
            title=dict(
                text=title,
                font=dict(color='white', size=18)
            ),
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='white'),
            height=450,
            showlegend=True,
            legend=dict(
                font=dict(color='white'),
                bgcolor='#2a2a2a',
                bordercolor='#3a3a3a',
                borderwidth=1
            ),
            geo=dict(
                scope='usa',
                projection=dict(type='albers usa'),
                showland=True,
                landcolor='#2a2a2a',
                showcountries=True,
                showsubunits=True,
                subunitcolor='#4a4a4a',
                bgcolor='#1e1e1e',
                coastlinecolor='#4a4a4a',
                lakecolor='#1e1e1e'
            )
        )
        
        return go.Figure(data=data, layout=layout)
    
    def on_plot_click(self, data):
        """Obsługuje kliknięcie w punkt na wykresie"""
        lon = data['lon']
        lat = data['lat']
        value = data['value']
        
        self.selected_coords = (lon, lat)
        
        print(f"Kliknięto punkt: Lon={lon:.4f}, Lat={lat:.4f}, Value={value:.4f}")
        
        # Aktualizacja informacji o współrzędnych
        self.coords_label.setText(f"Wybrane współrzędne: Longitude: {lon:.4f}, Latitude: {lat:.4f}")
        
        # Czyszczenie poprzednich wartości
        while self.values_layout.count():
            child = self.values_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Znajdowanie najbliższego punktu w każdym modelu i wyświetlanie wartości
        for model_name, (lons, lats, preds) in self.models_data.items():
            # Obliczanie odległości do wszystkich punktów
            distances = np.sqrt((lons - lon)**2 + (lats - lat)**2)
            nearest_idx = np.argmin(distances)
            
            value = preds[nearest_idx]
            actual_lon = lons[nearest_idx]
            actual_lat = lats[nearest_idx]
            
            value_label = QLabel(
                f"<b>{model_name}:</b> Score = {value:.4f} "
                f"(Lon: {actual_lon:.4f}, Lat: {actual_lat:.4f})"
            )
            value_label.setStyleSheet("padding: 5px; font-size: 12px;")
            self.values_layout.addWidget(value_label)
        
        # Aktywacja przycisku optymalizacji
        self.optimize_button.setEnabled(True)
    
    @Slot()
    def optimize_points(self):
        """Dodaje nowe punkty wokół wybranych współrzędnych"""
        if self.selected_coords is None:
            print("Brak wybranych współrzędnych")
            return
        
        n_points = self.point_count_spinbox.value()
        lon, lat = self.selected_coords
        
        print(f"Dodawanie {n_points} punktów wokół ({lon:.4f}, {lat:.4f})")
        
        # Generowanie nowych punktów wokół wybranych współrzędnych
        # Rozrzut ~1 stopień
        new_lons = np.random.normal(lon, 1.0, n_points)
        new_lats = np.random.normal(lat, 1.0, n_points)
        
        # Ograniczenie do granic USA
        new_lons = np.clip(new_lons, -125, -65)
        new_lats = np.clip(new_lats, 25, 50)
        
        # Generowanie wartości dla nowych punktów (można tu dodać własną logikę)
        new_preds = np.random.uniform(-3, 7, n_points)
        
        # Aktualizacja danych wszystkich modeli
        for model_name in self.models_data.keys():
            # Dodawanie nowych punktów do oddzielnej struktury
            if model_name not in self.optimized_points:
                self.optimized_points[model_name] = (new_lons.copy(), new_lats.copy(), new_preds.copy())
            else:
                old_lons, old_lats, old_preds = self.optimized_points[model_name]
                self.optimized_points[model_name] = (
                    np.concatenate([old_lons, new_lons]),
                    np.concatenate([old_lats, new_lats]),
                    np.concatenate([old_preds, new_preds])
                )
        
        # Odświeżanie wszystkich wykresów
        self.refresh_all_plots()
        
        print(f"Pomyślnie dodano {n_points} punktów")
    
    def refresh_all_plots(self):
        """Odświeża wszystkie wykresy z nowymi danymi"""
        for model_name, widget in self.plot_widgets.items():
            if model_name in self.models_data:
                lons, lats, preds = self.models_data[model_name]
                
                # Pobieramy punkty optymalizacji jeśli istnieją
                opt_lons, opt_lats, opt_preds = None, None, None
                if model_name in self.optimized_points:
                    opt_lons, opt_lats, opt_preds = self.optimized_points[model_name]
                
                fig = self.create_geo_plot(lons, lats, preds, f'{model_name} - Predykcje USA', 
                                          opt_lons, opt_lats, opt_preds)
                
                # Dla set_figure przekazujemy wszystkie punkty razem
                all_lons = np.concatenate([lons, opt_lons]) if opt_lons is not None else lons
                all_lats = np.concatenate([lats, opt_lats]) if opt_lats is not None else lats
                all_preds = np.concatenate([preds, opt_preds]) if opt_preds is not None else preds
                
                widget.set_figure(fig, all_lons, all_lats, all_preds)
    
    def add_plot_tabs(self):
        """Dodaje zakładki z różnymi wykresami"""
        
        # Zakładka 1: Model A
        lon1, lat1, pred1 = self.create_sample_data(seed=42)
        self.models_data['Model A'] = (lon1, lat1, pred1)
        plot1 = PlotlyWidget()
        plot1.click_callback = self.on_plot_click
        fig1 = self.create_geo_plot(lon1, lat1, pred1, 'Model A - Predykcje USA')
        plot1.set_figure(fig1, lon1, lat1, pred1)
        self.tab_widget.addTab(plot1, "Model A")
        self.plot_widgets['Model A'] = plot1
        
        # Zakładka 2: Model B
        lon2, lat2, pred2 = self.create_sample_data(seed=123)
        self.models_data['Model B'] = (lon2, lat2, pred2)
        plot2 = PlotlyWidget()
        plot2.click_callback = self.on_plot_click
        fig2 = self.create_geo_plot(lon2, lat2, pred2, 'Model B - Predykcje USA')
        plot2.set_figure(fig2, lon2, lat2, pred2)
        self.tab_widget.addTab(plot2, "Model B")
        self.plot_widgets['Model B'] = plot2
        
        # Zakładka 3: Model C
        lon3, lat3, pred3 = self.create_sample_data(seed=456)
        self.models_data['Model C'] = (lon3, lat3, pred3)
        plot3 = PlotlyWidget()
        plot3.click_callback = self.on_plot_click
        fig3 = self.create_geo_plot(lon3, lat3, pred3, 'Model C - Predykcje USA')
        plot3.set_figure(fig3, lon3, lat3, pred3)
        self.tab_widget.addTab(plot3, "Model C")
        self.plot_widgets['Model C'] = plot3
        
        # Zakładka 4: Porównanie
        lon4, lat4, pred4 = self.create_sample_data(seed=789)
        pred4 = pred4 * 0.8
        self.models_data['Porównanie'] = (lon4, lat4, pred4)
        plot4 = PlotlyWidget()
        plot4.click_callback = self.on_plot_click
        fig4 = self.create_geo_plot(lon4, lat4, pred4, 'Porównanie - Średnia Modeli')
        plot4.set_figure(fig4, lon4, lat4, pred4)
        self.tab_widget.addTab(plot4, "Porównanie")
        self.plot_widgets['Porównanie'] = plot4


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("USA Predictions Viewer")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()