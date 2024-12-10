import java.util.Random;

public class PSO {
    private int numParticles; // Số lượng hạt
    private int maxIterations; // Số lần lặp tối đa
    private double w; // Hệ số quán tính
    private double c1; // Hệ số học cá nhân
    private double c2; // Hệ số học xã hội
    private int dimension; // Số chiều của không gian tìm kiếm
    private double[] globalBestPosition; // Vị trí tốt nhất toàn cục
    private double globalBestFitness; // Giá trị fitness tốt nhất toàn cục
    private Particle[] particles; // Mảng các hạt
    
    public class Particle {
        double[] position; // Vị trí hiện tại
        double[] velocity; // Vận tốc
        double[] personalBestPosition; // Vị trí tốt nhất cá nhân
        double personalBestFitness; // Giá trị fitness tốt nhất cá nhân
        
        public Particle(int dimension) {
            position = new double[dimension];
            velocity = new double[dimension];
            personalBestPosition = new double[dimension];
            Random rand = new Random();
            
            // Khởi tạo ngẫu nhiên vị trí và vận tốc
            for (int i = 0; i < dimension; i++) {
                position[i] = rand.nextDouble() * 100; // Giả sử phạm vi 0-100
                velocity[i] = rand.nextDouble() * 2 - 1; // Vận tốc từ -1 đến 1
                personalBestPosition[i] = position[i];
            }
            personalBestFitness = evaluateFitness(position);
        }
    }
    
    public PSO(int numParticles, int maxIterations, int dimension) {
        this.numParticles = numParticles;
        this.maxIterations = maxIterations;
        this.dimension = dimension;
        this.w = 0.729; // Giá trị thường dùng
        this.c1 = 1.49445; // Giá trị thường dùng
        this.c2 = 1.49445; // Giá trị thường dùng
        
        particles = new Particle[numParticles];
        globalBestPosition = new double[dimension];
        globalBestFitness = Double.NEGATIVE_INFINITY;
        
        // Khởi tạo các hạt
        for (int i = 0; i < numParticles; i++) {
            particles[i] = new Particle(dimension);
            if (particles[i].personalBestFitness > globalBestFitness) {
                globalBestFitness = particles[i].personalBestFitness;
                System.arraycopy(particles[i].position, 0, globalBestPosition, 0, dimension);
            }
        }
    }
    
    // Hàm đánh giá fitness (cần điều chỉnh theo bài toán cụ thể)
    private double evaluateFitness(double[] position) {
        // Ví dụ: tìm max của tổng các thành phần
        double sum = 0;
        for (double pos : position) {
            sum += pos;
        }
        return sum;
    }
    
    public void optimize() {
        Random rand = new Random();
        
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            for (Particle particle : particles) {
                // Cập nhật vận tốc
                for (int d = 0; d < dimension; d++) {
                    double r1 = rand.nextDouble();
                    double r2 = rand.nextDouble();
                    
                    particle.velocity[d] = w * particle.velocity[d] +
                            c1 * r1 * (particle.personalBestPosition[d] - particle.position[d]) +
                            c2 * r2 * (globalBestPosition[d] - particle.position[d]);
                }
                
                // Cập nhật vị trí
                for (int d = 0; d < dimension; d++) {
                    particle.position[d] += particle.velocity[d];
                }
                
                // Đánh giá fitness mới
                double fitness = evaluateFitness(particle.position);
                
                // Cập nhật personal best
                if (fitness > particle.personalBestFitness) {
                    particle.personalBestFitness = fitness;
                    System.arraycopy(particle.position, 0, particle.personalBestPosition, 0, dimension);
                    
                    // Cập nhật global best
                    if (fitness > globalBestFitness) {
                        globalBestFitness = fitness;
                        System.arraycopy(particle.position, 0, globalBestPosition, 0, dimension);
                    }
                }
            }
            
            // In kết quả sau mỗi lần lặp
            System.out.println("Iteration " + iteration + ": Best fitness = " + globalBestFitness);
        }
    }
    
    public static void main(String[] args) {
        int numParticles = 30;
        int maxIterations = 100;
        int dimension = 2; // Số chiều của không gian tìm kiếm
        
        PSO pso = new PSO(numParticles, maxIterations, dimension);
        pso.optimize();
        
        // In kết quả cuối cùng
        System.out.println("\nKết quả tối ưu:");
        System.out.println("Fitness: " + pso.globalBestFitness);
        System.out.print("Vị trí: ");
        for (double pos : pso.globalBestPosition) {
            System.out.print(pos + " ");
        }
    }
}