#include "bailey/precision_traits.hpp"
#include "bailey/dd_arithmetic.hpp"
#include "bailey/dq_arithmetic.hpp"
#include "bailey/quad_double.hpp"
#include "algorithms/conjugate_gradient.hpp"
#include "io/matrix_market.hpp"

#include <iostream>
#include <string>
#include <variant>
#include <iomanip>
#include <sstream>
#include <algorithm>

// Command line configuration
struct SolverConfig {
    std::string matrix_name;
    std::string precision_level{"qx"};  // dd, dq, qx
    double tolerance{1.0e-12};
    std::variant<int, double> max_iter{2.0};  // Default: 2*n
    std::string input_dir{"/work/inputs"};
};

// Command line parser
SolverConfig parseCommandLine(int argc, char* argv[]) {
    SolverConfig config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--matrix" && i + 1 < argc) {
            config.matrix_name = argv[++i];
        }
        else if (arg == "--precision" && i + 1 < argc) {
            config.precision_level = argv[++i];
            if (config.precision_level != "dd" && 
                config.precision_level != "dq" && 
                config.precision_level != "qx" &&
                config.precision_level != "double") {
                throw std::runtime_error("Invalid precision level. Use: dd, dq, qx, or double");
            }
        }
        else if (arg == "--tol" && i + 1 < argc) {
            try {
                config.tolerance = std::stod(argv[++i]);
            } catch (...) {
                throw std::runtime_error("Invalid tolerance value");
            }
        }
        else if (arg == "--max-iter" && i + 1 < argc) {
            std::string value = argv[++i];
            try {
                if (value.find('.') != std::string::npos) {
                    config.max_iter = std::stod(value);
                } else {
                    config.max_iter = std::stoi(value);
                }
            } catch (...) {
                throw std::runtime_error("Invalid max-iter value");
            }
        }
        else if (arg == "--input-dir" && i + 1 < argc) {
            config.input_dir = argv[++i];
        }
        else if (arg == "--help" || arg == "-h") {
            throw std::runtime_error("help");  // Special case for help
        }
        else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    
    if (config.matrix_name.empty()) {
        throw std::runtime_error("Matrix name is required (--matrix)");
    }
    
    return config;
}

void printUsage(const char* program_name) {
    std::cout << "\nUsage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --matrix NAME         Matrix name (required, e.g., nos5 for nos5.mtx)\n";
    std::cout << "  --precision LEVEL     Precision level: dd, dq, qx, double (default: qx)\n";
    std::cout << "  --tol VALUE           Convergence tolerance (default: 1.0e-12)\n";
    std::cout << "  --max-iter VALUE      Maximum iterations:\n";
    std::cout << "                        - Integer: absolute number of iterations\n";
    std::cout << "                        - Float: coefficient * matrix_size (default: 2.0)\n";
    std::cout << "  --input-dir PATH      Input directory path (default: /work/inputs)\n";
    std::cout << "  --help, -h            Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " --matrix nos5 --precision qx --tol 1e-15\n";
    std::cout << "  " << program_name << " --matrix nos7 --precision dq --max-iter 1000\n";
    std::cout << "  " << program_name << " --matrix test --precision dd --max-iter 2.5\n";
    std::cout << "  " << program_name << " --matrix nos5 --precision double --tol 1e-10\n\n";
}

// Template solver function
template<typename T>
int solveCG(const SolverConfig& config) {
    using Traits = bailey::PrecisionTraits<T>;
    using MatrixType = typename Traits::matrix_type;
    using VectorType = typename Traits::vector_type;
    
    std::string matrix_path = io::constructMatrixPath(config.matrix_name, config.input_dir);
    
    std::cout << "Loading matrix: " << matrix_path << " (precision: " << Traits::name() << ")" << std::endl;
    
    MatrixType A = io::loadMatrixMarket<T>(matrix_path);
    int n = A.rows();
    
    std::cout << "Matrix size: " << n << " x " << A.cols() << std::endl;
    std::cout << "Non-zeros: " << A.nonZeros() << std::endl;
    
    // Calculate max iterations
    int max_iterations = algorithms::resolve_max_iterations(config.max_iter, n);
    
    std::cout << "Max iterations: " << max_iterations << std::endl;
    std::cout << std::scientific << std::setprecision(2) << "Tolerance: " << config.tolerance << std::endl;
    
    // Set up problem: Ax = b where x_true = ones(n)
    VectorType x_true = VectorType::Ones(n);
    VectorType b = A * x_true;
    VectorType x = VectorType::Zero(n);  // Initial guess
    
    std::cout << "\nStarting CG iterations...\n";
    
    auto result = algorithms::conjugateGradient<T>(A, b, x, x_true, max_iterations, config.tolerance);
    
    // Print results
    algorithms::print_results(result, config.matrix_name + ".mtx");
    
    return result.converged ? 0 : 2;  // Exit code 2 for non-convergence (not an error)
}

// Solver dispatcher using std::variant
int runSolver(const SolverConfig& config) {
    if (config.precision_level == "dd") {
        return solveCG<bailey::DDNumber>(config);
    } else if (config.precision_level == "dq") {
        return solveCG<bailey::DQNumber>(config);
    } else if (config.precision_level == "qx") {
        return solveCG<QuadDouble>(config);
    } else if (config.precision_level == "double") {
        return solveCG<double>(config);
    } else {
        std::cerr << "Invalid precision level: " << config.precision_level << std::endl;
        return 1;
    }
}

int main(int argc, char* argv[]) {
    try {
        SolverConfig config = parseCommandLine(argc, argv);
        return runSolver(config);
    } catch (const std::exception& e) {
        std::string error_msg = e.what();
        if (error_msg == "help") {
            printUsage(argv[0]);
            return 0;
        }
        std::cerr << "Error: " << error_msg << std::endl;
        if (error_msg.find("argument") != std::string::npos || error_msg.find("required") != std::string::npos) {
            printUsage(argv[0]);
        }
        return 1;
    }
}