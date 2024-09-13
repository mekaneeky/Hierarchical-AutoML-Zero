
from automl.evolutionary_algorithm import AutoMLZero
from automl.memory import HierarchicalMemoryArrays
from automl.function_decoder import FunctionDecoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import deque
from copy import deepcopy

import torch
import torch.nn as nn
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QTextEdit, QProgressBar, 
                             QTabWidget, QStyleFactory, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Import the necessary modules from main_activation_2.py

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    val_data = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1000, shuffle=False)
    return train_loader, val_loader


def train(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for idx, (inputs, targets) in enumerate(train_loader):
        if idx == 10:
            break
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        

def evaluate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total


class EvolvableNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, evolved_activation):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.evolved_activation = evolved_activation

    def forward(self, x):
        x = x.view(-1,28*28)
        x = self.fc1(x)
        x = self.evolved_activation(x)
        x = self.fc2(x)
        return x

class AppleStyleButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #007AFF;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #003d80;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
        """)

class AppleStyleSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QSpinBox {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 20px;
                background-color: #F0F0F0;
            }
        """)

class AutoMLWorker(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(object)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        # Initialize components
        hierarchical_memory = HierarchicalMemoryArrays(
            self.config['num_meta_levels'],
            self.config['num_scalars'],
            self.config['num_vectors'],
            self.config['num_tensors'],
            self.config['scalar_size'],
            self.config['vector_size'],
            self.config['tensor_size']
        )
        function_decoder = FunctionDecoder()

        # Initialize AutoML-Zero
        automl = AutoMLZero(
            population_size=self.config['population_size'],
            num_meta_levels=self.config['num_meta_levels'],
            genome_length=self.config['genome_length'],
            tournament_size=self.config['tournament_size'],
            hierarchical_memory=hierarchical_memory,
            function_decoder=function_decoder
        )

        # Load data
        train_loader, val_loader = load_data()

        # Initialize population and evaluate initial fitness
        population = automl.hierarchical_genome.genomes[-1]
        self.evaluate_population(population, train_loader, val_loader)

        best_genome_all_time = max(population, key=lambda g: g.fitness)
        best_fitness_history = [best_genome_all_time.fitness]

        # Evolution loop
        for generation in range(self.config['num_generations']):
            for _ in range(len(population)):
                parent = automl.tournament_selection(population)
                offspring = automl.mutate(parent, 0)

                model = EvolvableNN(
                    input_size=28*28, 
                    hidden_size=128, 
                    output_size=10, 
                    evolved_activation=offspring.function()
                )

                try:
                    train(model, train_loader)
                    accuracy = evaluate(model, val_loader)
                    offspring.fitness = accuracy
                except:
                    offspring.fitness = -9999

                population.append(offspring)
                population.pop(0)

            best_genome = max(population, key=lambda g: g.fitness)
            if best_genome.fitness > best_genome_all_time.fitness:
                best_genome_all_time = best_genome

            best_fitness_history.append(best_genome_all_time.fitness)
            self.progress.emit(int((generation + 1) / self.config['num_generations'] * 100))

        self.result.emit((best_genome_all_time, best_fitness_history))

    def evaluate_population(self, population, train_loader, val_loader):
        for genome in population:
            try:
                model = EvolvableNN(
                    input_size=28*28, 
                    hidden_size=128, 
                    output_size=10, 
                    evolved_activation=genome.function()
                )
                train(model, train_loader)
                genome.fitness = evaluate(model, val_loader)
            except:
                genome.fitness = -9999

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoML")
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F7;
            }
            QLabel {
                font-size: 14px;
                color: #333333;
            }
            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                font-size: 14px;
                padding: 5px;
            }
            QTabWidget::pane {
                border: 1px solid #CCCCCC;
                background-color: #FFFFFF;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #E5E5E5;
                color: #333333;
                padding: 8px 20px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #FFFFFF;
            }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # Left panel for configuration and control
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel, 1)

        # Configuration inputs
        self.config_inputs = {}
        config_params = [
            ('population_size', 'Population Size', 100, 1000),
            ('num_meta_levels', 'Meta Levels', 1, 5),
            ('genome_length', 'Genome Length', 5, 100),
            ('tournament_size', 'Tournament Size', 2, 100),
            ('num_generations', 'Generations', 10, 1000)
        ]

        for param, label, min_val, max_val in config_params:
            layout = QHBoxLayout()
            layout.addWidget(QLabel(f"{label}:"))
            self.config_inputs[param] = AppleStyleSpinBox()
            self.config_inputs[param].setRange(min_val, max_val)
            layout.addWidget(self.config_inputs[param])
            left_layout.addLayout(layout)

        # Start button
        self.start_button = AppleStyleButton("Start Evolution")
        self.start_button.clicked.connect(self.start_evolution)
        left_layout.addWidget(self.start_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #007AFF;
                border-radius: 5px;
            }
        """)
        left_layout.addWidget(self.progress_bar)

        # Right panel for results and visualization
        right_panel = QTabWidget()
        main_layout.addWidget(right_panel, 2)

        # Results tab
        results_widget = QWidget()
        results_layout = QVBoxLayout()
        results_widget.setLayout(results_layout)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        right_panel.addTab(results_widget, "Results")

        # Visualization tab
        viz_widget = QWidget()
        viz_layout = QVBoxLayout()
        viz_widget.setLayout(viz_layout)
        self.figure = plt.figure(facecolor='#F5F5F7')
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        right_panel.addTab(viz_widget, "Visualization")

    def start_evolution(self):
        config = {
            'population_size': self.config_inputs['population_size'].value(),
            'num_meta_levels': self.config_inputs['num_meta_levels'].value(),
            'genome_length': self.config_inputs['genome_length'].value(),
            'tournament_size': self.config_inputs['tournament_size'].value(),
            'num_generations': self.config_inputs['num_generations'].value(),
            'num_scalars': 5,
            'num_vectors': 5,
            'num_tensors': 5,
            'scalar_size': 1,
            'vector_size': (28,),
            'tensor_size': (28, 28)
        }

        self.worker = AutoMLWorker(config)
        self.worker.progress.connect(self.update_progress)
        self.worker.result.connect(self.show_results)
        self.worker.start()

        self.start_button.setEnabled(False)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def show_results(self, result):
        best_genome, fitness_history = result
        self.results_text.setText(f"Best genome fitness: {best_genome.fitness:.4f}")

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(fitness_history, color='#007AFF')
        ax.set_facecolor('#F5F5F7')
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Best Fitness', fontsize=12)
        ax.set_title('Evolution Progress', fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.canvas.draw()

        self.start_button.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())