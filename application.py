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
from torch.utils.data import DataLoader, Dataset

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
    


class EvolutionaryOptimizer:
    """Algorytm ewolucyjny do optymalizacji lokalizacji punktów"""
    def __init__(self, model, data_tensor, target_lon, target_lat, 
                 population_size=50, generations=30, min_distance=0.5):
        self.model = model
        self.data_tensor = data_tensor
        self.target_lon = target_lon
        self.target_lat = target_lat
        self.population_size = population_size
        self.generations = generations
        self.min_distance = min_distance  # minimalna odległość między punktami
        
        # Granice USA
        self.lon_min, self.lon_max = -125, -65
        self.lat_min, self.lat_max = 25, 50
        
        # Ekstrakcja cech z data_tensor (indeksy 2-6)
        self.feature_indices = [2, 3, 4, 5, 6]
        self.features_mean = data_tensor[:, self.feature_indices].mean(dim=0).numpy()
        self.features_std = data_tensor[:, self.feature_indices].std(dim=0).numpy()
    
    def initialize_population(self, n_points):
        """Inicjalizuje populację wokół punktu docelowego"""
        population = []
        for _ in range(self.population_size):
            # Losowe punkty wokół target z większym rozrzutem
            lons = np.random.normal(self.target_lon, 2.0, n_points)
            lats = np.random.normal(self.target_lat, 2.0, n_points)
            
            # Clip do granic USA
            lons = np.clip(lons, self.lon_min, self.lon_max)
            lats = np.clip(lats, self.lat_min, self.lat_max)
            
            population.append((lats, lons))
        
        return population
    
    def create_features_for_points(self, lats, lons):
        """Tworzy cechy dla punktów na podstawie lokalizacji"""
        n_points = len(lats)
        features = np.zeros((n_points, 5))
        
        # Generujemy cechy bazując na lokalizacji i średnich z datasetu
        for i in range(n_points):
            # Dodajemy niewielką wariancję do średnich cech
            for j in range(5):
                features[i, j] = np.random.normal(
                    self.features_mean[j], 
                    self.features_std[j] * 0.3
                )
        
        return features
    
    def calculate_fitness(self, lats, lons):
        """Oblicza fitness: score z modelu - kara za bliskość"""
        n_points = len(lats)
        
        # Tworzymy tensor wejściowy: [lat, lon, features...]
        features = self.create_features_for_points(lats, lons)
        input_data = np.column_stack([lats, lons, features])
        input_tensor = torch.FloatTensor(input_data)
        
        # Score z modelu
        with torch.no_grad():
            scores = self.model(input_tensor).squeeze().numpy()
        
        # Średni score (im wyższy tym lepiej)
        avg_score = np.mean(scores)
        
        # Kara za bliskość punktów (nieliniowa)
        proximity_penalty = 0
        if n_points > 1:
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    dist = np.sqrt((lons[i] - lons[j])**2 + (lats[i] - lats[j])**2)
                    
                    # Nieliniowa kara - im bliżej tym większa kara
                    if dist < self.min_distance:
                        # Bardzo silna kara gdy punkty są bardzo blisko
                        proximity_penalty += (self.min_distance - dist) ** 2
                    elif dist < self.min_distance * 2:
                        # Mniejsza kara gdy są w średniej odległości
                        proximity_penalty += (self.min_distance * 2 - dist) * 0.5
        
        # Normalizujemy karę przez liczbę par punktów
        if n_points > 1:
            proximity_penalty /= (n_points * (n_points - 1) / 2)
        
        # Fitness = score - penalty (chcemy maksymalizować)
        fitness = avg_score - proximity_penalty * 5.0  # waga kary
        
        return fitness, avg_score, proximity_penalty
    
    def mutate(self, lats, lons, mutation_rate=0.3):
        """Mutacja punktów"""
        new_lats = lats.copy()
        new_lons = lons.copy()
        
        for i in range(len(lats)):
            if np.random.random() < mutation_rate:
                new_lats[i] += np.random.normal(0, 0.5)
                new_lons[i] += np.random.normal(0, 0.5)
        
        # Clip do granic
        new_lats = np.clip(new_lats, self.lat_min, self.lat_max)
        new_lons = np.clip(new_lons, self.lon_min, self.lon_max)
        
        return new_lats, new_lons
    
    def crossover(self, parent1, parent2):
        """Krzyżowanie dwóch rodziców"""
        lats1, lons1 = parent1
        lats2, lons2 = parent2
        
        n_points = len(lats1)
        crossover_point = np.random.randint(1, n_points)
        
        # Tworzymy potomka
        child_lats = np.concatenate([lats1[:crossover_point], lats2[crossover_point:]])
        child_lons = np.concatenate([lons1[:crossover_point], lons2[crossover_point:]])
        
        return child_lats, child_lons
    
    def optimize(self, n_points):
        """Główna pętla algorytmu ewolucyjnego"""
        print(f"\n=== Rozpoczynam optymalizację ewolucyjną ===")
        print(f"Populacja: {self.population_size}, Generacje: {self.generations}")
        print(f"Punkty do wygenerowania: {n_points}")
        
        # Inicjalizacja populacji
        population = self.initialize_population(n_points)
        best_fitness_history = []
        
        for gen in range(self.generations):
            # Ocena fitness dla całej populacji
            fitness_scores = []
            for lats, lons in population:
                fitness, score, penalty = self.calculate_fitness(lats, lons)
                fitness_scores.append((fitness, score, penalty, (lats, lons)))
            
            # Sortowanie według fitness (malejąco)
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            best_fitness, best_score, best_penalty, best_individual = fitness_scores[0]
            best_fitness_history.append(best_fitness)
            
            if gen % 5 == 0:
                print(f"Gen {gen}: Best Fitness={best_fitness:.4f}, "
                      f"Score={best_score:.4f}, Penalty={best_penalty:.4f}")
            
            # Selekcja (top 50%)
            elite_size = self.population_size // 2
            elite = [ind for _, _, _, ind in fitness_scores[:elite_size]]
            
            # Tworzenie nowej populacji
            new_population = elite.copy()
            
            # Krzyżowanie i mutacja
            while len(new_population) < self.population_size:
                # Wybieramy dwóch rodziców z elity
                parent1 = elite[np.random.randint(0, elite_size)]
                parent2 = elite[np.random.randint(0, elite_size)]
                
                # Krzyżowanie
                child_lats, child_lons = self.crossover(parent1, parent2)
                
                # Mutacja
                child_lats, child_lons = self.mutate(child_lats, child_lons)
                
                new_population.append((child_lats, child_lons))
            
            population = new_population
        
        # Zwracamy najlepszego osobnika z ostatniej generacji
        final_fitness = []
        for lats, lons in population:
            fitness, score, penalty = self.calculate_fitness(lats, lons)
            final_fitness.append((fitness, score, penalty, (lats, lons)))
        
        final_fitness.sort(key=lambda x: x[0], reverse=True)
        best_fitness, best_score, best_penalty, (best_lats, best_lons) = final_fitness[0]
        
        print(f"\n=== Optymalizacja zakończona ===")
        print(f"Final Best Fitness: {best_fitness:.4f}")
        print(f"Final Best Score: {best_score:.4f}")
        print(f"Final Proximity Penalty: {best_penalty:.4f}")
        
        # Obliczamy predykcje dla najlepszych punktów
        features = self.create_features_for_points(best_lats, best_lons)
        input_data = np.column_stack([best_lats, best_lons, features])
        input_tensor = torch.FloatTensor(input_data)
        
        with torch.no_grad():
            predictions = self.model(input_tensor).squeeze().numpy()
        
        return best_lons, best_lats, predictions
# input_dim = 7    
# model = RankNet(input_dim=input_dim)
# model.load_state_dict(torch.load("usa_model_200_400.pt", map_location="cpu"))
# model.eval()

class USADatasetLoaded(Dataset):
    def __init__(self, path):
        self.data = torch.load(path).data    
        self.lendata = self.data.shape[0]

    def __len__(self):
        return self.lendata

    def __getitem__(self, idx):
        return self.data[idx, :]   
    

# dataset = USADatasetLoaded("usa_dataset_200_400_g.pt")
# dataloader = DataLoader(
#     dataset,
#     batch_size=128,
#     shuffle=True
# )
# print(f"DATASET SHAPE: {len(dataset)}")
# data_tensor = dataset.data 

# with torch.no_grad():
#     predictions = model(data_tensor).squeeze().numpy()

# latitudes = data_tensor[:, 0].numpy()
# longitudes = data_tensor[:, 1].numpy()
    




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
        
        # Wczytywanie modelu i danych
        self.load_model_and_data()
        
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
    
    def load_model_and_data(self):
        """Wczytuje model i dane"""
        input_dim = 7
        self.model = RankNet(input_dim=input_dim)
        self.model.load_state_dict(torch.load("usa_model_200_400.pt", map_location="cpu"))
        self.model.eval()
        
        # TODO: Wczytaj swoje data_tensor tutaj
        # Na razie używam placeholder - zamień to na właściwe wczytywanie danych
        # Przykład: self.data_tensor = torch.load("your_data.pt")
        dataset = USADatasetLoaded("usa_dataset_200_400_g.pt")
        # dataloader = DataLoader(
        #     dataset,
        #     batch_size=128,
        #     shuffle=True
        # )
        print(f"DATASET SHAPE: {len(dataset)}")
        self.data_tensor = dataset.data[::3] 

    

        
        # Placeholder - generowanie losowych danych o właściwym kształcie
        # Kolumny: [lat, lon, wind_eff, solar, fiber, temp, pop_density]
        
        # Nazwy cech
        self.feature_names = [
            "Wind efficiency",
            "Solar power", 
            "Fiber optics",
            "Temperature",
            "Population density"
        ]
        self.feature_indices = [2, 3, 4, 5, 6]
        
        # Obliczanie predykcji
        with torch.no_grad():
            self.predictions = self.model(self.data_tensor).squeeze().cpu().numpy()
        
        # Ekstraktacja danych
        self.latitudes = self.data_tensor[:, 0].cpu().numpy()
        self.longitudes = self.data_tensor[:, 1].cpu().numpy()
        self.features = [self.data_tensor[:, idx].cpu().numpy() for idx in self.feature_indices]
        
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
        """Tworzy przykładowe dane dla wykresów - DEPRECATED, używamy prawdziwych danych"""
        # Ta metoda nie jest już używana - dane pochodzą z load_model_and_data()
        pass
    
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
        """Dodaje nowe punkty wokół wybranych współrzędnych używając algorytmu ewolucyjnego"""
        if self.selected_coords is None:
            print("Brak wybranych współrzędnych")
            return
        
        n_points = self.point_count_spinbox.value()
        target_lon, target_lat = self.selected_coords
        
        print(f"\n{'='*60}")
        print(f"Rozpoczynam optymalizację ewolucyjną")
        print(f"Cel: ({target_lon:.4f}, {target_lat:.4f})")
        print(f"Liczba punktów do wygenerowania: {n_points}")
        print(f"{'='*60}")
        
        # Tworzymy optimizer z aktualnym modelem i danymi
        optimizer = EvolutionaryOptimizer(
            model=self.model,
            data_tensor=self.data_tensor,
            target_lon=target_lon,
            target_lat=target_lat,
            population_size=50,      # Możesz dostosować
            generations=30,          # Możesz dostosować
            min_distance=0.5         # Minimalna odległość między punktami
        )
        
        # Uruchamiamy optymalizację
        try:
            new_lons, new_lats, new_preds = optimizer.optimize(n_points)
            
            print(f"\n{'='*60}")
            print(f"Optymalizacja zakończona pomyślnie!")
            print(f"Wygenerowano {len(new_lons)} punktów")
            print(f"Średni score: {np.mean(new_preds):.4f}")
            print(f"Min score: {np.min(new_preds):.4f}")
            print(f"Max score: {np.max(new_preds):.4f}")
            
            # Sprawdzamy odległości między punktami
            if n_points > 1:
                min_dist = float('inf')
                for i in range(n_points):
                    for j in range(i + 1, n_points):
                        dist = np.sqrt((new_lons[i] - new_lons[j])**2 + 
                                    (new_lats[i] - new_lats[j])**2)
                        min_dist = min(min_dist, dist)
                print(f"Minimalna odległość między punktami: {min_dist:.4f}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"Błąd podczas optymalizacji: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Aktualizacja danych wszystkich modeli (tylko Model Predictions)
        # Dla Model Predictions używamy rzeczywistych predykcji
        model_name = 'Model Predictions'
        if model_name not in self.optimized_points:
            self.optimized_points[model_name] = (new_lons.copy(), new_lats.copy(), new_preds.copy())
        else:
            old_lons, old_lats, old_preds = self.optimized_points[model_name]
            self.optimized_points[model_name] = (
                np.concatenate([old_lons, new_lons]),
                np.concatenate([old_lats, new_lats]),
                np.concatenate([old_preds, new_preds])
            )
        
        # Dla cech - musimy wyekstrahować wartości cech dla nowych punktów
        # Tworzymy tensor wejściowy i pobieramy cechy
        features = optimizer.create_features_for_points(new_lats, new_lons)
        
        for i, feature_name in enumerate(self.feature_names):
            feature_values = features[:, i]
            
            if feature_name not in self.optimized_points:
                self.optimized_points[feature_name] = (
                    new_lons.copy(), 
                    new_lats.copy(), 
                    feature_values.copy()
                )
            else:
                old_lons, old_lats, old_vals = self.optimized_points[feature_name]
                self.optimized_points[feature_name] = (
                    np.concatenate([old_lons, new_lons]),
                    np.concatenate([old_lats, new_lats]),
                    np.concatenate([old_vals, feature_values])
                )
        
        # Odświeżanie wszystkich wykresów
        self.refresh_all_plots()
        
        print(f"Wykresy zaktualizowane - czerwone punkty to wyniki optymalizacji")
    
    def refresh_all_plots(self):
        """Odświeża wszystkie wykresy z nowymi danymi"""
        for model_name, widget in self.plot_widgets.items():
            if model_name in self.models_data:
                lons, lats, vals = self.models_data[model_name]
                
                # Pobieramy punkty optymalizacji jeśli istnieją
                opt_lons, opt_lats, opt_vals = None, None, None
                if model_name in self.optimized_points:
                    opt_lons, opt_lats, opt_vals = self.optimized_points[model_name]
                
                # Dla cech używamy create_feature_plot, dla predykcji create_geo_plot
                if model_name in self.feature_names:
                    # Obliczamy vmin/vmax dla tej cechy
                    try:
                        vmin = float(np.nanpercentile(vals, 1))
                        vmax = float(np.nanpercentile(vals, 99))
                        if np.isclose(vmin, vmax):
                            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
                    except Exception:
                        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
                    
                    fig = self.create_feature_plot(lons, lats, vals, model_name, vmin, vmax,
                                                   opt_lons, opt_lats, opt_vals)
                else:
                    fig = self.create_geo_plot(lons, lats, vals, f'{model_name} - USA', 
                                              opt_lons, opt_lats, opt_vals)
                
                # Dla set_figure przekazujemy wszystkie punkty razem
                all_lons = np.concatenate([lons, opt_lons]) if opt_lons is not None else lons
                all_lats = np.concatenate([lats, opt_lats]) if opt_lats is not None else lats
                all_vals = np.concatenate([vals, opt_vals]) if opt_vals is not None else vals
                
                widget.set_figure(fig, all_lons, all_lats, all_vals)
    
    def add_plot_tabs(self):
        """Dodaje zakładki z różnymi wykresami"""
        
        # Zakładka 1: Model Predictions
        self.models_data['Model Predictions'] = (self.longitudes, self.latitudes, self.predictions)
        plot_pred = PlotlyWidget()
        plot_pred.click_callback = self.on_plot_click
        fig_pred = self.create_geo_plot(
            self.longitudes, self.latitudes, self.predictions, 
            'Model Predictions - USA'
        )
        plot_pred.set_figure(fig_pred, self.longitudes, self.latitudes, self.predictions)
        self.tab_widget.addTab(plot_pred, "Model Predictions")
        self.plot_widgets['Model Predictions'] = plot_pred
        
        # Zakładki dla każdej cechy
        for i, (feature_name, feature_data) in enumerate(zip(self.feature_names, self.features)):
            # Obliczamy rozsądne min/max dla colorscale
            try:
                vmin = float(np.nanpercentile(feature_data, 1))
                vmax = float(np.nanpercentile(feature_data, 99))
                if np.isclose(vmin, vmax):
                    vmin, vmax = float(np.nanmin(feature_data)), float(np.nanmax(feature_data))
            except Exception:
                vmin, vmax = float(np.nanmin(feature_data)), float(np.nanmax(feature_data))
            
            self.models_data[feature_name] = (self.longitudes, self.latitudes, feature_data)
            plot_feat = PlotlyWidget()
            plot_feat.click_callback = self.on_plot_click
            
            # Tworzymy wykres z custom vmin/vmax
            fig_feat = self.create_feature_plot(
                self.longitudes, self.latitudes, feature_data,
                feature_name, vmin, vmax
            )
            plot_feat.set_figure(fig_feat, self.longitudes, self.latitudes, feature_data)
            self.tab_widget.addTab(plot_feat, feature_name)
            self.plot_widgets[feature_name] = plot_feat
    
    def create_feature_plot(self, longitudes, latitudes, values, title, vmin, vmax, 
                           opt_lons=None, opt_lats=None, opt_vals=None):
        """Tworzy wykres dla konkretnej cechy z custom zakresem kolorów"""
        # Podstawowe punkty danych
        scatter = go.Scattergeo(
            lon=longitudes,
            lat=latitudes,
            text=[f"{title}: {v:.3f}" for v in values],
            marker=dict(
                size=2,
                color=values,
                cmin=vmin,
                cmax=vmax,
                colorscale='Viridis',
                colorbar=dict(
                    title=title,
                    # titlefont=dict(color='white'),
                    tickfont=dict(color='white')
                ),
                line=dict(width=0)
            ),
            mode='markers',
            hovertemplate='<b>Lon:</b> %{lon:.2f}<br><b>Lat:</b> %{lat:.2f}<br><b>Value:</b> %{text}<extra></extra>',
            name='Original Data'
        )
        
        data = [scatter]
        
        # Jeśli są punkty optymalizacji, dodajemy je jako osobną warstwę
        if opt_lons is not None and len(opt_lons) > 0:
            scatter_opt = go.Scattergeo(
                lon=opt_lons,
                lat=opt_lats,
                text=[f"{title}: {v:.3f}" for v in opt_vals],
                marker=dict(
                    size=6,
                    color='#FF6B6B',
                    line=dict(width=1, color='white')
                ),
                mode='markers',
                hovertemplate='<b>Optimized Point</b><br><b>Lon:</b> %{lon:.2f}<br><b>Lat:</b> %{lat:.2f}<br><b>Value:</b> %{text}<extra></extra>',
                name='Optimized Points'
            )
            data.append(scatter_opt)
        
        layout = go.Layout(
            title=dict(
                text=f'{title} - USA Distribution',
                font=dict(color='white', size=18)
            ),
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='white'),
            height=450,
            showlegend=True if opt_lons is not None else False,
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
                lakecolor='#1e1e1e',
                lataxis=dict(range=[25, 50]),
                lonaxis=dict(range=[-125, -67])
            )
        )
        
        return go.Figure(data=data, layout=layout)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("USA Predictions Viewer")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()