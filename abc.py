import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

class ABC:
    def __init__(self, function, lb, ub, colony_size=50, max_iterations=100, dimensions=3):
        self.function = function
        self.lb = lb
        self.ub = ub
        self.colony_size = colony_size
        self.max_iterations = max_iterations
        self.dimensions = dimensions
        
        # Số ong thợ = số ong trinh sát = colony_size/2
        self.food_sources = colony_size // 2
        self.limit = colony_size * dimensions  # Giới hạn thử trước khi từ bỏ nguồn thức ăn
        
        # Khởi tạo các nguồn thức ăn
        self.solutions = np.random.uniform(lb, ub, (self.food_sources, dimensions))
        self.fitness = np.zeros(self.food_sources)
        self.trials = np.zeros(self.food_sources)
        
        # Lưu lại lịch sử để vẽ đồ thị
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []

    def calculate_fitness(self, solution):
        function_value = self.function(solution)
        if function_value >= 0:
            fitness = 1 / (1 + function_value)
        else:
            fitness = 1 + abs(function_value)
        return fitness

    def employed_bees(self):
        for i in range(self.food_sources):
            # Tạo giải pháp mới
            k = random.randint(0, self.food_sources-1)
            while k == i:
                k = random.randint(0, self.food_sources-1)
            
            phi = random.uniform(-1, 1)
            j = random.randint(0, self.dimensions-1)
            
            new_solution = self.solutions[i].copy()
            new_solution[j] = self.solutions[i][j] + phi * (self.solutions[i][j] - self.solutions[k][j])
            
            # Kiểm tra giới hạn
            new_solution = np.clip(new_solution, self.lb, self.ub)
            
            # So sánh và cập nhật nếu tốt hơn
            new_fitness = self.calculate_fitness(new_solution)
            if new_fitness > self.fitness[i]:
                self.solutions[i] = new_solution
                self.fitness[i] = new_fitness
                self.trials[i] = 0
            else:
                self.trials[i] += 1

    def onlooker_bees(self):
        probabilities = self.fitness / np.sum(self.fitness)
        
        for _ in range(self.food_sources):
            i = np.random.choice(range(self.food_sources), p=probabilities)
            
            # Tương tự như employed bees
            k = random.randint(0, self.food_sources-1)
            while k == i:
                k = random.randint(0, self.food_sources-1)
            
            phi = random.uniform(-1, 1)
            j = random.randint(0, self.dimensions-1)
            
            new_solution = self.solutions[i].copy()
            new_solution[j] = self.solutions[i][j] + phi * (self.solutions[i][j] - self.solutions[k][j])
            new_solution = np.clip(new_solution, self.lb, self.ub)
            
            new_fitness = self.calculate_fitness(new_solution)
            if new_fitness > self.fitness[i]:
                self.solutions[i] = new_solution
                self.fitness[i] = new_fitness
                self.trials[i] = 0
            else:
                self.trials[i] += 1

    def scout_bees(self):
        for i in range(self.food_sources):
            if self.trials[i] >= self.limit:
                self.solutions[i] = np.random.uniform(self.lb, self.ub, self.dimensions)
                self.fitness[i] = self.calculate_fitness(self.solutions[i])
                self.trials[i] = 0

    def optimize(self):
        # Tính fitness ban đầu
        for i in range(self.food_sources):
            self.fitness[i] = self.calculate_fitness(self.solutions[i])
        
        for iteration in range(self.max_iterations):
            # Giai đoạn ong thợ
            self.employed_bees()
            
            # Giai đoạn ong giám sát
            self.onlooker_bees()
            
            # Giai đoạn ong trinh sát
            self.scout_bees()
            
            # Cập nhật giải pháp tốt nhất
            current_best_idx = np.argmax(self.fitness)
            if self.function(self.solutions[current_best_idx]) < self.best_fitness:
                self.best_solution = self.solutions[current_best_idx].copy()
                self.best_fitness = self.function(self.best_solution)
            
            self.history.append(self.best_fitness)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.best_fitness}")
        
        return self.best_solution, self.best_fitness

def plot_optimization_3d(abc, title="Optimization Progress"):
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Convergence history
    ax1 = fig.add_subplot(121)
    ax1.plot(abc.history)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Best Fitness')
    ax1.set_title('Convergence History')
    
    # Plot 2: 3D scatter of solutions
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(abc.solutions[:, 0], abc.solutions[:, 1], abc.solutions[:, 2], 
                c='blue', marker='o', label='Food Sources')
    ax2.scatter(abc.best_solution[0], abc.best_solution[1], abc.best_solution[2], 
                c='red', marker='*', s=200, label='Best Solution')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Solution Space')
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Hàm test (Sphere function)
def sphere_function(x):
    return np.sum(x**2)

# Chạy thuật toán
if __name__ == "__main__":
    # Thiết lập tham số
    dimensions = 3
    lb = -5.0  # Lower bound
    ub = 5.0   # Upper bound
    colony_size = 50
    max_iterations = 100
    
    # Khởi tạo ABC
    abc = ABC(sphere_function, lb, ub, colony_size, max_iterations, dimensions)
    
    # Tối ưu hóa
    best_solution, best_fitness = abc.optimize()
    
    print("\nKết quả cuối cùng:")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    
    # Vẽ đồ thị
    plot_optimization_3d(abc, "ABC Optimization Results")