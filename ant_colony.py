import random
import numpy as np
import math
import io
import base64
import plotly.graph_objs as go
from plotly.offline import plot

class AntColony:
    def __init__(self, cities, num_ants, num_iterations, alpha=1, beta=2, evaporation_rate=0.5, Q=100):
        self.cities = cities
        self.distances = self.calculate_distance_matrix(cities)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.num_cities = len(cities)
        self.pheromones = np.ones((self.num_cities, self.num_cities))
        self.best_path = None
        self.iteration_distances = []
        self.best_distance = float('inf')
        self.start_city = random.randint(0, self.num_cities - 1)

    def calculate_distance_matrix(self, cities):
        n = len(cities)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distances[i][j] = distances[j][i] = self.att_distance(cities[i], cities[j])
        return distances

    @staticmethod
    def att_distance(city1, city2):
        xd = city1[0] - city2[0]
        yd = city1[1] - city2[1]
        rij = math.sqrt((xd ** 2 + yd ** 2) / 10.0)
        tij = round(rij)
        return tij + 1 if tij < rij else tij

    def run(self):
        for i in range(self.num_iterations):
            all_paths = []
            all_distances = []

            for _ in range(self.num_ants):
                path, distance = self.generate_path()
                all_paths.append(path)
                all_distances.append(distance)

                if distance < self.best_distance:
                    self.best_path = path
                    self.best_distance = distance

            self.update_pheromones(all_paths, all_distances)
            self.iteration_distances.append({
            "iteration_number": i + 1,
            "iteration_distance": self.best_distance
        })

    def generate_path(self):
        path = [self.start_city]
        visited = set(path)
        total_distance = 0

        for _ in range(self.num_cities - 1):
            current_city = path[-1]
            next_city = self.choose_next_city(current_city, visited)
            path.append(next_city)
            visited.add(next_city)
            total_distance += self.distances[current_city, next_city]

        total_distance += self.distances[path[-1], self.start_city]
        path.append(self.start_city)

        return path, total_distance

    def choose_next_city(self, current_city, visited):
        probabilities = []
        total_pheromone = 0

        for city in range(self.num_cities):
            if city not in visited:
                pheromone = self.pheromones[current_city, city] ** self.alpha
                heuristic = (1 / self.distances[current_city, city]) ** self.beta
                probability = pheromone * heuristic
                probabilities.append((city, probability))
                total_pheromone += probability

        probabilities = [(city, prob / total_pheromone) for city, prob in probabilities]
        cities, probs = zip(*probabilities)
        next_city = random.choices(cities, probs)[0]
        return next_city

    def update_pheromones(self, all_paths, all_distances):
        self.pheromones *= (1 - self.evaporation_rate)
        for path, distance in zip(all_paths, all_distances):
            pheromone_contribution = self.Q / distance
            for i in range(len(path) - 1):
                self.pheromones[path[i], path[i + 1]] += pheromone_contribution
                self.pheromones[path[i + 1], path[i]] += pheromone_contribution

    def plot_best_path(self):
        print(self.best_path)

        # Extracting coordinates for the best path
        x_coords = [self.cities[city][0] for city in self.best_path]
        y_coords = [self.cities[city][1] for city in self.best_path]

        # Creating a Plotly figure
        fig = go.Figure()

        # Adding a scatter plot for the best path with larger markers and black lines
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+lines+text',
            marker=dict(size=20, color='black'),  # Increased size of the dots
            line=dict(color='blue', width=2),  # Black path lines
            name='Best Path',
            text=[str(index) for index in range(len(self.best_path))],  # Order numbers as text
            textposition='middle center',  # Position text in the middle of the marker
            textfont=dict(color='white')  # White color for the text
        ))

        # Highlight the starting city
        start_city_x = self.cities[self.best_path[0]][0]
        start_city_y = self.cities[self.best_path[0]][1]
        
        fig.add_trace(go.Scatter(
            x=[start_city_x],
            y=[start_city_y],
            mode='markers',
            marker=dict(size=25, color='red', symbol='circle'),  # Larger size for start city
            name='Start City'
        ))

        # Adding layout details
        fig.update_layout(
            title=f"Best path found with distance {self.best_distance}",
            xaxis_title="X",
            yaxis_title="Y",
            showlegend=True,
            width=1200,  # Increase width
            height=800  # Increase height
        )

        # Generate HTML div for the Plotly figure
        plot_div = plot(fig, include_plotlyjs=False, output_type='div')
        
        return plot_div