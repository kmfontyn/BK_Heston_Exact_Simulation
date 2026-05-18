#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <string>
#include <sstream>

using namespace std;

// 1. Data structure: MCResult

// Stores one row of Monte Carlo simulation output.

// Fields:
// M: Number of Monte Carlo simulation paths.
// time_steps: Number of Euler time steps used per simulated path.
// price: Estimated discounted European call option price.
// bias: Difference between estimated price and benchmark true price; bias = price - C_true.
// std_error: Monte Carlo standard error of the estimated option price.
// rmse: Root mean squared error combining bias and sampling error; rmse = sqrt(bias^2 + std_error^2).
// seconds: Wall-clock computation time needed to produce this row.

// This structure is used to reproduce the Broadie-Kaya-style Euler comparison tables.

struct MCResult {
    long long M;
    int time_steps;
    double price;
    double bias;
    double std_error;
    double rmse;
    double seconds;
};

// 2. Function: volatility_time_t

// Computes one Euler-discretized update of the Heston variance process.
// Continuous-time variance process: dV_t = kappa(theta - V_t)dt + sigma sqrt(V_t)dW_t^(1).
// Euler approximation: V_{t+dt} = V_t + kappa(theta - V_t)dt + sigma sqrt(V_t) Delta W_t^(1).

// Inputs:
// dt: Length of one discrete time step.
// V_prev: Variance value at the beginning of the time step.
// delta_Wt_1: Brownian increment driving the variance process; distributed N(0, dt).
// kappa: Speed of mean reversion in the variance process.
// theta: Long-run mean variance level.
// sigma: Volatility of variance, or vol-of-vol.

// Output:
// V_next: Euler-updated variance value at the next time step, floored at zero.

double volatility_time_t(double dt,double V_prev,double delta_Wt_1,double kappa=6.21,double theta=0.019,double sigma=0.61){
    double V_for_sqrt=max(V_prev,0.0);
    double V_next=V_prev+kappa*(theta-V_prev)*dt+sqrt(V_for_sqrt)*sigma*delta_Wt_1;
    V_next=max(V_next,0.0);
    return V_next;
}
// 3. Function: price_time_t

// Computes one Euler-discretized update of the Heston stock-price process.
// Continuous-time stock-price process: dS_t = rS_t dt + sqrt(V_t)S_t[rho dW_t^(1) + sqrt(1-rho^2)dW_t^(2)].
// Euler approximation: S_{t+dt} = S_t + rS_t dt + sqrt(V_t)S_t[rho Delta W_t^(1) + sqrt(1-rho^2) Delta W_t^(2)].

// Inputs:
// dt: Length of one discrete time step.
// delta_Wt_1: Brownian increment shared with the variance process.
// delta_Wt_2: Independent Brownian increment used only in the stock process.
// V_prev: Variance value used over the current time interval.
// S_prev: Stock price at the beginning of the time step.
// r: Risk-free rate under the risk-neutral pricing measure.
// rho: Instantaneous correlation between the stock and variance shocks; must satisfy -1 <= rho <= 1.

// Output:
// S_next: Euler-updated stock price at the next time step.

double price_time_t(double dt,double delta_Wt_1,double delta_Wt_2,double V_prev,double S_prev,double r=0.0319,double rho=-0.7){
    double V_for_sqrt=max(V_prev,0.0);
    double S_next=S_prev+r*S_prev*dt+sqrt(V_for_sqrt)*S_prev*(rho*delta_Wt_1+sqrt(1.0-rho*rho)*delta_Wt_2);
    return S_next;
}

// 4. Function: get_ST

// Simulates one terminal stock price S_T under the Euler-discretized Heston model.
// The interval [0,T] is divided into time_steps equal subintervals. At each step, two independent standard normal random variables are generated and scaled by sqrt(dt), producing Brownian increments distributed N(0, dt).

// Inputs:
// time_steps: Number of Euler subintervals used to approximate [0,T].
// rng: Pseudorandom number generator passed by reference.
// standard_normal: Standard normal distribution object used to generate N(0,1) variates.
// S_0: Initial stock price.
// V_0: Initial variance.
// T: Option maturity, in years.
// r: Risk-free interest rate.
// kappa: Speed of mean reversion of the variance process.
// theta: Long-run mean variance.
// sigma: Volatility of variance.
// rho: Correlation between stock-return and variance shocks.

// Output:
// S_prev: Simulated terminal stock price S_T.

double get_ST(
    int time_steps,
    mt19937_64& rng,
    normal_distribution<double>& standard_normal,
    double S_0 = 100.0,
    double V_0 = 0.010201,
    double T = 1.0,
    double r = 0.0319,
    double kappa = 6.21,
    double theta = 0.019,
    double sigma = 0.61,
    double rho = -0.7
){
    // Length of each Euler time step and its square root for Brownian increments.
    double dt = T / static_cast<double>(time_steps);
    double sqrt_dt = sqrt(dt);

    // Initialize the variance and stock price at time 0.
    double V_prev = V_0;
    double S_prev = S_0;

    // Step the Heston system forward on the discrete Euler grid.
    for (int step = 0; step < time_steps; ++step) {
        // Store the current variance value before updating; this is the value
        // used in both the variance and stock-price Euler steps.
        double V_old = V_prev;

        // Simulate the two independent Brownian increments over the current step.
        double delta_Wt_1 = sqrt_dt * standard_normal(rng);
        double delta_Wt_2 = sqrt_dt * standard_normal(rng);

        // Update the variance process using the Euler approximation.
        double V_next = volatility_time_t(dt, V_old, delta_Wt_1, kappa, theta, sigma);

        // Update the stock-price process using the Euler approximation.
        double S_next = price_time_t(dt, delta_Wt_1, delta_Wt_2, V_old, S_prev, r, rho);

        // Move the process state forward to the next time point.
        V_prev = V_next;
        S_prev = S_next;
    }

    // Return the terminal stock price S_T after all Euler steps.
    return S_prev;
}

// 5. Function: get_MC_call_price
// Estimates a European call option price using Euler-discretized Monte Carlo simulation under the Heston model.
// European call payoff: max(S_T - K, 0).
// Monte Carlo price estimate: exp(-rT) * average payoff.
// Note that this function avoids storing all terminal prices by accumulating payoff sums and squared payoff sums.

// Inputs:
// M: Number of Monte Carlo paths.
// time_steps: Number of Euler time steps per path.
// S_0: Initial stock price.
// V_0: Initial variance.
// T: Option maturity, in years.
// r: Risk-free interest rate.
// K: Strike price of the European call option.
// C_true: Benchmark option value used to compute bias and RMSE.
// kappa: Speed of mean reversion of the variance process.
// theta: Long-run mean variance.
// sigma: Volatility of variance.
// rho: Correlation between stock-return and variance shocks.
// seed: Random seed used to initialize the generator.

// Output:
// MCResult result containing M, time_steps, price, bias, std_error, rmse, and seconds.

MCResult get_MC_call_price(
    long long M,
    int time_steps,
    double S_0 = 100.0,
    double V_0 = 0.010201,
    double T = 1.0,
    double r = 0.0319,
    double K = 100.0,
    double C_true = 6.8061,
    double kappa = 6.21,
    double theta = 0.019,
    double sigma = 0.61,
    double rho = -0.7,
    unsigned long long seed = 12345
){
    // Initialize the random number generator and standard normal draw object
    // used to simulate Brownian increments in the Euler scheme.
    mt19937_64 rng(seed);
    normal_distribution<double> standard_normal(0.0,1.0);

    // Risk-neutral discount factor for converting the expected payoff to present value.
    double discount = exp(-r * T);

    double payoff_sum = 0.0;
    double payoff_sq_sum = 0.0;

    // Simulate M terminal stock prices under the Euler discretization and
    // accumulate the corresponding European call payoffs.
    for (long long i = 0; i < M; ++i) {
        double ST = get_ST(time_steps, rng, standard_normal, S_0, V_0, T, r, kappa, theta, sigma, rho);
        double payoff = max(ST - K, 0.0);
        payoff_sum += payoff;
        payoff_sq_sum += payoff * payoff;
    }

    // Monte Carlo mean of the undiscounted payoff.
    double mean_payoff = payoff_sum / static_cast<double>(M);

    // Unbiased sample variance of the payoff values.
    double sample_variance_payoff =
        (payoff_sq_sum - static_cast<double>(M) * mean_payoff * mean_payoff)
        / static_cast<double>(M - 1);

    // Discounted option price estimate.
    double price = discount * mean_payoff;

    // Standard error of the discounted Monte Carlo estimator.
    double std_error = discount * sqrt(sample_variance_payoff / static_cast<double>(M));

    // Bias relative to the benchmark option price.
    double bias = price - C_true;

    // Root mean squared error combines sampling error and discretization bias.
    double rmse = sqrt(std_error * std_error + bias * bias);

    MCResult result;
    result.M = M;
    result.time_steps = time_steps;
    result.price = price;
    result.bias = bias;
    result.std_error = std_error;
    result.rmse = rmse;
    result.seconds = 0.0;
    return result;
}

// 6. Function: construct_broadie_kaya_table

// Constructs the Broadie-Kaya-style Euler discretization table.

// Inputs:
// S_0: Initial stock price.
// V_0: Initial variance.
// T: Option maturity.
// r: Risk-free rate.
// K: Strike price.
// C_true: Benchmark true option price used for bias/RMSE calculations.
// kappa: Speed of variance mean reversion.
// theta: Long-run mean variance.
// sigma: Volatility of variance.
// rho: Correlation between stock and variance shocks.
// seed: Base random seed; a deterministic offset is added for each row.

// Output:
// results: Vector of MCResult objects, where each element corresponds to one table row.

vector<MCResult> construct_broadie_kaya_table(
    double S_0 = 100.0,
    double V_0 = 0.010201,
    double T = 1.0,
    double r = 0.0319,
    double K = 100.0,
    double C_true = 6.8061,
    double kappa = 6.21,
    double theta = 0.019,
    double sigma = 0.61,
    double rho = -0.7,
    unsigned long long seed = 12345,
    vector<long long> M_vals = {10000,40000,160000,640000,2560000,10240000},
    vector<int> time_steps_vals = {100,200,400,800,1600,3200}
){
    // Store one Monte Carlo result row for each (M, time_steps) experiment.
    vector<MCResult> results;

    // Run the Euler simulation across the Broadie–Kaya replication grid.
    for (size_t i = 0; i < M_vals.size(); ++i) {
        cout << "Running M = " << M_vals[i]
             << ", Time Steps = " << time_steps_vals[i] << "..." << endl;

        auto start = chrono::high_resolution_clock::now();

        // Estimate the call price and associated error measures for the current grid point.
        MCResult row = get_MC_call_price(
            M_vals[i],
            time_steps_vals[i],
            S_0,
            V_0,
            T,
            r,
            K,
            C_true,
            kappa,
            theta,
            sigma,
            rho,
            seed + static_cast<unsigned long long>(i)
        );

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;

        // Record the runtime for the current experiment.
        row.seconds = elapsed.count();
        results.push_back(row);

        // Print a summary line for the completed run.
        cout << fixed << setprecision(6);
        cout << "Done: "
             << "Price = " << row.price
             << ", Bias = " << row.bias
             << ", Standard Error = " << row.std_error
             << ", RMSE = " << row.rmse
             << ", Time = " << row.seconds << " sec"
             << endl << endl;
    }

    // Return the full table of Euler simulation results.
    return results;
}

// 7. Function: write_table_to_csv
// Writes the Monte Carlo results table to a CSV file.

// Inputs:
// results: Vector of MCResult objects produced by construct_broadie_kaya_table.
// output_path: Full path of the CSV file to be written.

// Output:
// None directly; writes a CSV file to a prespecified directory.

// Side effects:
// Creates or overwrites output_path, prints confirmation if successful, and prints an error if the file cannot be opened.
// CSV columns: M, Time_Steps, Price, Bias, Standard Error, RMSE, Time (Sec).

void write_table_to_csv(const vector<MCResult>& results, const string& output_path){
    // Open the output CSV file.
    ofstream file(output_path);
    if (!file.is_open()){
        cerr << "Error: could not open file for writing: " << output_path << endl;
        return;
    }

    // Write the column headers expected for the Euler results table.
    file << "M,Time_Steps,Price,Bias,Standard Error,RMSE,Time (Sec)\n";
    file << fixed << setprecision(10);

    // Write one row per Monte Carlo experiment.
    for (const auto& row : results){
        file << row.M << ","
             << row.time_steps << ","
             << row.price << ","
             << row.bias << ","
             << row.std_error << ","
             << row.rmse << ","
             << row.seconds << "\n";
    }

    file.close();

    // Confirm where the file was written.
    cout << "Table written to: " << output_path << endl;
}

// Function: 8. main

// Purpose:
// Reads optional command-line parameters, sets default Broadie-Kaya Table 1 parameters if none are supplied, constructs the Euler simulation table, and writes the table to CSV.

// Implementing the full procedure of creating the table requires command line usage:

// Custom parameter values with a user-designed experiment grid:
// ./heston_euler S_0 K V_0 kappa theta sigma rho r T C_true output_filename seed M_vals time_steps_vals

// The experiment-grid arguments must be comma-separated lists with no spaces.
//      (i) M_vals gives the Monte Carlo path counts.
//      (ii) time_steps_vals gives the Euler time-step counts.

// The code assumes that the i-th value of M_vals is paired with the i-th value of time_steps_vals.

// This makes the code adaptable beyond the fixed Broadie-Kaya tables because the user can choose the Monte Carlo effort and discretization level without editing the source code.

// Required command-line inputs:
// S_0: Initial stock price.
// K: Strike price.
// V_0: Initial variance.
// kappa: Speed of variance mean reversion.
// theta: Long-run mean variance.
// sigma: Volatility of variance.
// rho: Correlation between stock and variance shocks.
// r: Risk-free rate.
// T: Maturity in years.
// C_true: Benchmark true call option price.
// Optional command-line inputs:
// output_filename: Name of the output CSV file.
// seed: Random seed.
// M_vals: Comma-separated list of Monte Carlo path counts.
// time_steps_vals: Comma-separated list of Euler time-step counts.

// Output:
// Prints parameter values and simulation progress, then writes the final table to CSV.

int main(int argc, char* argv[]){
    // Create the output directory.
    string output_dir = "Tables";
    filesystem::create_directories(output_dir);

    // Default Broadie–Kaya Table 1 parameter values.
    double S_0 = 100.0;
    double K = 100.0;
    double V_0 = 0.010201;
    double kappa = 6.21;
    double theta = 0.019;
    double sigma = 0.61;
    double rho = -0.70;
    double r = 0.0319;
    double T = 1.0;
    double C_true = 6.8061;

    // Default output settings and replication grid.
    string output_filename = "broadie_kaya_euler_table.csv";
    unsigned long long seed = 12345;
    vector<long long> M_vals = {10000,40000,160000,640000,2560000,10240000};
    vector<int> time_steps_vals = {100,200,400,800,1600,3200};

    // No command-line arguments: use all default settings.
    if (argc == 1){
        cout << "Using default Broadie-Kaya Table 1 parameters." << endl;
        cout << "Using default Broadie-Kaya-style experiment grid." << endl;

    // Parameter input with optional output name, seed, and custom experiment grid.
    } else if (argc == 11 || argc == 12 || argc == 13 || argc == 15){
        S_0 = stod(argv[1]);
        K = stod(argv[2]);
        V_0 = stod(argv[3]);
        kappa = stod(argv[4]);
        theta = stod(argv[5]);
        sigma = stod(argv[6]);
        rho = stod(argv[7]);
        r = stod(argv[8]);
        T = stod(argv[9]);
        C_true = stod(argv[10]);

        // Optional output filename.
        if (argc >= 12){
            output_filename = argv[11];
        }

        // Optional random seed.
        if (argc >= 13){
            seed = stoull(argv[12]);
        }

        // Optional custom experiment grid:
        // argv[13] = comma-separated Monte Carlo sizes
        // argv[14] = comma-separated time-step counts
        if (argc == 15){
            M_vals.clear();
            time_steps_vals.clear();

            stringstream M_stream(argv[13]);
            stringstream steps_stream(argv[14]);
            string token;

            while (getline(M_stream, token, ',')){
                M_vals.push_back(stoll(token));
            }

            while (getline(steps_stream, token, ',')){
                time_steps_vals.push_back(stoi(token));
            }

            // The two experiment lists must align row by row.
            if (M_vals.size() != time_steps_vals.size()){
                cerr << "Invalid experiment grid." << endl;
                cerr << "M_vals and time_steps_vals must have the same number of entries." << endl;
                return 1;
            }

            // Reject empty custom grids.
            if (M_vals.empty()){
                cerr << "Invalid experiment grid." << endl;
                cerr << "M_vals and time_steps_vals cannot be empty." << endl;
                return 1;
            }

            cout << "Using command-line parameter values." << endl;
            cout << "Using user-specified experiment grid." << endl;
        } else {
            // Parameter values were supplied, but the default experiment grid is retained.
            cout << "Using command-line parameter values." << endl;
            cout << "Using default Broadie-Kaya-style experiment grid." << endl;
        }

    } else {
        // Any other argument count is invalid; print usage instructions.
        cerr << "Invalid number of arguments." << endl;
        cerr << endl;
        cerr << "Usage:" << endl;
        cerr << "  ./heston_euler" << endl;
        cerr << endl;
        cerr << "or:" << endl;
        cerr << "  ./heston_euler S_0 K V_0 kappa theta sigma rho r T C_true output_filename seed" << endl;
        cerr << endl;
        cerr << "or:" << endl;
        cerr << "  ./heston_euler S_0 K V_0 kappa theta sigma rho r T C_true output_filename seed M_vals time_steps_vals" << endl;
        cerr << endl;
        cerr << "Required parameter order:" << endl;
        cerr << "  S_0 K V_0 kappa theta sigma rho r T C_true" << endl;
        cerr << endl;
        cerr << "Optional:" << endl;
        cerr << "  output_filename seed" << endl;
        cerr << "  M_vals time_steps_vals" << endl;
        cerr << endl;
        cerr << "Experiment grid format:" << endl;
        cerr << "  M_vals and time_steps_vals must be comma-separated lists with no spaces." << endl;
        cerr << "  The two lists must have the same number of entries." << endl;
        cerr << endl;
        cerr << "Example for Broadie-Kaya Table 2 with default grid:" << endl;
        cerr << "  ./heston_euler 100 100 0.09 2.00 0.09 1.00 -0.30 0.05 5.0 34.9998 broadie_kaya_euler_table2.csv 12345" << endl;
        cerr << endl;
        cerr << "Example for custom experiment grid:" << endl;
        cerr << "  ./heston_euler 100 100 0.09 2.00 0.09 1.00 -0.30 0.05 5.0 34.9998 custom_grid.csv 12345 10000,50000,100000 100,250,500" << endl;
        return 1;
    }

    // Full path for the CSV output file.
    string output_path = output_dir + "/" + output_filename;

    // Echo the experiment settings before running the simulation.
    cout << fixed << setprecision(6);
    cout << "Parameter values:" << endl;
    cout << "  S_0      = " << S_0 << endl;
    cout << "  K        = " << K << endl;
    cout << "  V_0      = " << V_0 << endl;
    cout << "  kappa    = " << kappa << endl;
    cout << "  theta    = " << theta << endl;
    cout << "  sigma    = " << sigma << endl;
    cout << "  rho      = " << rho << endl;
    cout << "  r        = " << r << endl;
    cout << "  T        = " << T << endl;
    cout << "  C_true   = " << C_true << endl;
    cout << "  seed     = " << seed << endl;
    cout << "  output   = " << output_path << endl;
    cout << "  grid     = " << endl;
    for (size_t i = 0; i < M_vals.size(); ++i){
        cout << "             row " << i + 1
             << ": M = " << M_vals[i]
             << ", Time Steps = " << time_steps_vals[i] << endl;
    }
    cout << endl;

    // Run the Euler simulation experiments across the requested grid.
    vector<MCResult> table = construct_broadie_kaya_table(
        S_0,
        V_0,
        T,
        r,
        K,
        C_true,
        kappa,
        theta,
        sigma,
        rho,
        seed,
        M_vals,
        time_steps_vals
    );

    // Write the completed results table to CSV.
    write_table_to_csv(table, output_path);

    return 0;
}