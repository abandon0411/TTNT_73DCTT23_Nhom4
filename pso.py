import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Particle:
    def __init__(self, num_cities, cities):
        self.position = list(range(num_cities))
        random.shuffle(self.position)
        self.pbest = self.position.copy()
        self.velocity = [0] * num_cities
        self.current_fitness = self.calculate_fitness(cities)
        self.best_fitness = self.current_fitness

    def calculate_fitness(self, cities):
        total_distance = 0
        for i in range(len(self.position)):
            city1 = cities[self.position[i]]
            city2 = cities[self.position[(i + 1) % len(self.position)]]
            distance = math.sqrt((city1[0] - city2[0])**2 + 
                               (city1[1] - city2[1])**2 + 
                               (city1[2] - city2[2])**2)
            total_distance += distance
        return total_distance

def PSO_TSP(cities, num_particles=50, max_iterations=100):
    num_cities = len(cities)
    particles = [Particle(num_cities, cities) for _ in range(num_particles)]
    gbest = min(particles, key=lambda x: x.current_fitness).position.copy()
    gbest_fitness = float('inf')
    
    w = 0.8
    c1 = 2.0
    c2 = 2.0
    
    # Lưu lại lịch sử các đường đi tốt nhất để vẽ animation
    history = []

    for iteration in range(max_iterations):
        for particle in particles:
            for i in range(num_cities):
                r1, r2 = random.random(), random.random()
                particle.velocity[i] = (w * particle.velocity[i] + 
                                      c1 * r1 * (particle.pbest[i] - particle.position[i]) +
                                      c2 * r2 * (gbest[i] - particle.position[i]))
            
            new_position = particle.position.copy()
            for i in range(num_cities):
                j = int(abs(particle.velocity[i])) % num_cities
                new_position[i], new_position[j] = new_position[j], new_position[i]
            
            particle.position = new_position
            particle.current_fitness = particle.calculate_fitness(cities)
            
            if particle.current_fitness < particle.best_fitness:
                particle.pbest = particle.position.copy()
                particle.best_fitness = particle.current_fitness
                
                if particle.best_fitness < gbest_fitness:
                    gbest = particle.pbest.copy()
                    gbest_fitness = particle.best_fitness
                    history.append((gbest.copy(), gbest_fitness))
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Best distance: {gbest_fitness}")
            
    return gbest, gbest_fitness, history

def plot_route_3d(cities, route, title=""):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Trích xuất tọa độ
    x_coords = [cities[i][0] for i in route]
    y_coords = [cities[i][1] for i in route]
    z_coords = [cities[i][2] for i in route]
    
    # Thêm điểm đầu tiên vào cuối để hoàn thành chu trình
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])
    z_coords.append(z_coords[0])
    
    # Vẽ các thành phố
    ax.scatter(x_coords, y_coords, z_coords, c='red', marker='o', s=100)
    
    # Vẽ đường đi
    ax.plot(x_coords, y_coords, z_coords, c='blue', linewidth=2)
    
    # Đánh số các thành phố
    for i, (x, y, z) in enumerate(zip(x_coords[:-1], y_coords[:-1], z_coords[:-1])):
        ax.text(x, y, z, f'City {i}', fontsize=8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    return fig, ax

def animate_route_evolution(cities, history, num_frames=5):
    frames = np.linspace(0, len(history)-1, num_frames, dtype=int)
    
    for idx in frames:
        route, distance = history[idx]
        fig, ax = plot_route_3d(cities, route, 
                               f'Iteration {idx}\nDistance: {distance:.2f}')
        plt.show()
        plt.close()

def generate_random_cities(num_cities):
    return [(random.uniform(0, 100), 
             random.uniform(0, 100), 
             random.uniform(0, 100)) for _ in range(num_cities)]

if __name__ == "__main__":
    # Thiết lập seed để có kết quả reproducible
    random.seed(42)
    np.random.seed(42)
    
    # Tạo dữ liệu thành phố
    num_cities = 10
    cities = generate_random_cities(num_cities)
    
    # Chạy thuật toán PSO
    best_route, best_distance, history = PSO_TSP(cities, 
                                                num_particles=50, 
                                                max_iterations=100)
    
    print("\nKết quả cuối cùng:")
    print(f"Đường đi tốt nhất: {best_route}")
    print(f"Tổng khoảng cách: {best_distance}")
    
    # Hiển thị quá trình tiến hóa của đường đi
    print("\nHiển thị quá trình tối ưu hóa đường đi...")
    animate_route_evolution(cities, history, num_frames=5)
    
    # Hiển thị kết quả cuối cùng
    print("\nHiển thị đường đi cuối cùng...")
    plot_route_3d(cities, best_route, 
                  f'Final Route\nTotal Distance: {best_distance:.2f}')
    plt.show()