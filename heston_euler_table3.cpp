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

// Documentation for the reused Euler simulation components is available in
// heston_euler.cpp. This file re-implements the Euler engine only as needed
// to construct a Broadie-Kaya Table 3-style comparison, so comments below are
// added only for structures and routines that are new to this version.


// 1. Data structures

struct MCResult {
    long long M;
    int time_steps;
    double price;
    double bias;
    double std_error;
    double rmse;
    double seconds;
};

// Heston Parameter Structure: HestonParameters
// Stores one complete Heston parameter set, together with a label identifying
// which Broadie-Kaya parameter scheme is being used.

// Fields:
// scheme_name: Descriptive label for the parameter set.
// S_0: Initial stock price.
// K: Strike price.
// V_0: Initial variance.
// kappa: Speed of variance mean reversion.
// theta: Long-run mean variance.
// sigma: Volatility of variance.
// rho: Correlation between stock and variance shocks.
// r: Risk-free rate.
// T: Option maturity.
// C_true: Benchmark option price used to compute Euler bias.

struct HestonParameters {
    string scheme_name;
    double S_0;
    double K;
    double V_0;
    double kappa;
    double theta;
    double sigma;
    double rho;
    double r;
    double T;
    double C_true;
};

// Table Structure: Table3Row
// Stores one row of the Broadie-Kaya Table 3-style Euler bias comparison.

// Table 3 compares the effect of using different numbers of Euler time steps
// for the same number of Monte Carlo simulation trials N.

// Fields:
// scheme_name: Identifies whether the row uses Table 1 or Table 2 parameters.
// N: Number of Monte Carlo simulation trials.
// standard_error: Standard error from the middle grid, steps = sqrt(N).
// steps_0_1_sqrtN: Euler time steps equal to approximately 0.1sqrt(N).
// bias_0_1_sqrtN: Bias using steps_0_1_sqrtN.
// steps_sqrtN: Euler time steps equal to approximately sqrt(N).
// bias_sqrtN: Bias using steps_sqrtN.
// steps_10_sqrtN: Euler time steps equal to approximately 10sqrt(N).
// bias_10_sqrtN: Bias using steps_10_sqrtN.
// seconds: Total runtime for all three step-count experiments in that row.

struct Table3Row {
    string scheme_name;
    long long N;
    double standard_error;

    int steps_0_1_sqrtN;
    double bias_0_1_sqrtN;

    int steps_sqrtN;
    double bias_sqrtN;

    int steps_10_sqrtN;
    double bias_10_sqrtN;

    double seconds;
};


// 2. Function(s): For Euler-discretized Heston dynamics

double volatility_time_t(
    double dt,
    double V_prev,
    double delta_Wt_1,
    double kappa,
    double theta,
    double sigma
) {
    double V_for_sqrt = max(V_prev, 0.0);

    double V_next =
        V_prev
        + kappa * (theta - V_prev) * dt
        + sqrt(V_for_sqrt) * sigma * delta_Wt_1;

    V_next = max(V_next, 0.0);

    return V_next;
}

double price_time_t(
    double dt,
    double delta_Wt_1,
    double delta_Wt_2,
    double V_prev,
    double S_prev,
    double r,
    double rho
) {
    double V_for_sqrt = max(V_prev, 0.0);

    double S_next =
        S_prev
        + r * S_prev * dt
        + sqrt(V_for_sqrt) * S_prev *
            (rho * delta_Wt_1 + sqrt(1.0 - rho * rho) * delta_Wt_2);

    return S_next;
}

// 3. Function: Simulate terminal stock price:
// This is the same path simulator as in heston_euler.cpp, but it now receives
// all model inputs through a HestonParameters object rather than as separate
// scalar arguments. This reduces repetition when comparing multiple parameter
// schemes.
double simulate_terminal_stock_price(
    int time_steps,
    mt19937_64& rng,
    normal_distribution<double>& standard_normal,
    const HestonParameters& params
) {
    double dt = params.T / static_cast<double>(time_steps);
    double sqrt_dt = sqrt(dt);

    double V_prev = params.V_0;
    double S_prev = params.S_0;

    for (int step = 0; step < time_steps; ++step) {
        double V_old = V_prev;

        double delta_Wt_1 = sqrt_dt * standard_normal(rng);
        double delta_Wt_2 = sqrt_dt * standard_normal(rng);

        double V_next = volatility_time_t(
            dt,
            V_old,
            delta_Wt_1,
            params.kappa,
            params.theta,
            params.sigma
        );

        double S_next = price_time_t(
            dt,
            delta_Wt_1,
            delta_Wt_2,
            V_old,
            S_prev,
            params.r,
            params.rho
        );

        V_prev = V_next;
        S_prev = S_next;
    }

    return S_prev;
}


// 4. Function: Monte Carlo estimator

// New implementation detail:
// This is the same Euler Monte Carlo estimator as in heston_euler.cpp, but it
// now takes a HestonParameters object so that the same function can be reused
// for both Broadie-Kaya parameter sets in Table 3.

MCResult estimate_call_price_euler(
    long long M,
    int time_steps,
    const HestonParameters& params,
    unsigned long long seed
) {
    mt19937_64 rng(seed);
    normal_distribution<double> standard_normal(0.0, 1.0);

    double discount = exp(-params.r * params.T);

    double payoff_sum = 0.0;
    double payoff_sq_sum = 0.0;

    for (long long i = 0; i < M; ++i) {
        double ST = simulate_terminal_stock_price(
            time_steps,
            rng,
            standard_normal,
            params
        );

        double payoff = max(ST - params.K, 0.0);

        payoff_sum += payoff;
        payoff_sq_sum += payoff * payoff;
    }

    double mean_payoff = payoff_sum / static_cast<double>(M);

    double sample_variance_payoff =
        (payoff_sq_sum - static_cast<double>(M) * mean_payoff * mean_payoff)
        / static_cast<double>(M - 1);

    if (sample_variance_payoff < 0.0) {
        sample_variance_payoff = 0.0;
    }

    double price = discount * mean_payoff;
    double std_error = discount * sqrt(sample_variance_payoff / static_cast<double>(M));

    double bias = price - params.C_true;
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


// 5. Function: Table 3 construction

// Constructs the Table 3-style rows for one Heston parameter scheme.

// Inputs:
// params: HestonParameters object for either the Table 1 or Table 2 parameter set.
// seed: Base seed used to make the simulation reproducible.
// N_vals: User-specified Monte Carlo simulation trial counts.

// Output:
// rows: Vector of Table3Row objects for the selected parameter scheme.

// Method:
// For each Monte Carlo sample size N, this function evaluates Euler bias under
// three time-step choices: 0.1sqrt(N), sqrt(N), and 10sqrt(N). This reproduces
// the structure of Broadie-Kaya Table 3, which studies how the Euler bias changes
// as the time discretization is refined.

vector<Table3Row> construct_table3_for_scheme(
    const HestonParameters& params,
    unsigned long long seed,
    const vector<long long>& N_vals
) {
    // Store one Table3Row for each Monte Carlo size N.
    vector<Table3Row> rows;

    for (size_t i = 0; i < N_vals.size(); ++i) {
        long long N = N_vals[i];

        // Broadie–Kaya Table 3 compares Euler bias under three time-step choices:
        // 0.1*sqrt(N), sqrt(N), and 10*sqrt(N).
        int sqrtN = static_cast<int>(sqrt(static_cast<double>(N)));

        int steps_0_1_sqrtN = max(1, static_cast<int>(0.1 * sqrtN));
        int steps_sqrtN = max(1, sqrtN);
        int steps_10_sqrtN = max(1, static_cast<int>(10.0 * sqrtN));

        // Report the current experiment before running the three Euler cases.
        cout << "Running " << params.scheme_name
             << ", N = " << N
             << ", steps = "
             << steps_0_1_sqrtN << ", "
             << steps_sqrtN << ", "
             << steps_10_sqrtN
             << endl;

        auto start = chrono::high_resolution_clock::now();

        // Estimate the option price and bias at 0.1*sqrt(N) time steps.
        MCResult result_0_1 = estimate_call_price_euler(
            N,
            steps_0_1_sqrtN,
            params,
            seed + 100000ULL + static_cast<unsigned long long>(i)
        );

        // Estimate the option price and bias at sqrt(N) time steps.
        MCResult result_1 = estimate_call_price_euler(
            N,
            steps_sqrtN,
            params,
            seed + 200000ULL + static_cast<unsigned long long>(i)
        );

        // Estimate the option price and bias at 10*sqrt(N) time steps.
        MCResult result_10 = estimate_call_price_euler(
            N,
            steps_10_sqrtN,
            params,
            seed + 300000ULL + static_cast<unsigned long long>(i)
        );

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;

        Table3Row row;
        row.scheme_name = params.scheme_name;
        row.N = N;

        // Broadie–Kaya Table 3 reports the standard error alongside the three
        // discretization-bias columns. Here the standard error is taken from
        // the middle case, where time steps = sqrt(N).
        row.standard_error = result_1.std_error;

        row.steps_0_1_sqrtN = steps_0_1_sqrtN;
        row.bias_0_1_sqrtN = result_0_1.bias;

        row.steps_sqrtN = steps_sqrtN;
        row.bias_sqrtN = result_1.bias;

        row.steps_10_sqrtN = steps_10_sqrtN;
        row.bias_10_sqrtN = result_10.bias;

        // Record the total runtime for all three Euler experiments at this N.
        row.seconds = elapsed.count();

        rows.push_back(row);

        // Print a summary line for the completed row.
        cout << fixed << setprecision(6);
        cout << "Done: "
             << "SE = " << row.standard_error
             << ", Bias(0.1sqrtN) = " << row.bias_0_1_sqrtN
             << ", Bias(sqrtN) = " << row.bias_sqrtN
             << ", Bias(10sqrtN) = " << row.bias_10_sqrtN
             << ", Time = " << row.seconds << " sec"
             << endl << endl;
    }

    // Return the full Table 3 replication output for the chosen parameter set.
    return rows;
}


// 6. CSV writer

// Writes the combined Table 3-style results to a CSV file.

// Inputs:
// rows: Vector of Table3Row objects from one or more parameter schemes.
// output_path: Full path of the output CSV file.

// Output:
// None directly. The function writes a CSV file to disk.

// CSV columns:
// Scheme, Number of Trials, Standard Error, time-step choices, corresponding
// biases, and runtime.

void write_table3_to_csv(
    const vector<Table3Row>& rows,
    const string& output_path
) {
    ofstream file(output_path);

    if (!file.is_open()) {
        cerr << "Error: could not open file for writing: " << output_path << endl;
        return;
    }

    file << "Scheme,"
         << "Number of Trials,"
         << "Standard Error,"
         << "Time Steps 0.1sqrtN,"
         << "Bias 0.1sqrtN,"
         << "Time Steps sqrtN,"
         << "Bias sqrtN,"
         << "Time Steps 10sqrtN,"
         << "Bias 10sqrtN,"
         << "Time (Sec)\n";

    file << fixed << setprecision(10);

    for (const auto& row : rows) {
        file
            << row.scheme_name << ","
            << row.N << ","
            << row.standard_error << ","
            << row.steps_0_1_sqrtN << ","
            << row.bias_0_1_sqrtN << ","
            << row.steps_sqrtN << ","
            << row.bias_sqrtN << ","
            << row.steps_10_sqrtN << ","
            << row.bias_10_sqrtN << ","
            << row.seconds << "\n";
    }

    file.close();

    cout << "Table 3 written to: " << output_path << endl;
}


// 7. Main

// Program entry point.

// Purpose:
// Reads command-line parameters, constructs a Table 3-style Euler bias comparison for a user-specified Heston parameter set, and writes the table to CSV.

// Command-line usage (i.e. to implement the full procedure of creating the table):

// To implement the full procedure of creating the table, the user must supply
// custom parameter values with a user-designed experiment grid in the form:

// ./heston_table3 scheme_name S_0 K V_0 kappa theta sigma rho r T C_true output_filename seed N_vals

// The experiment-grid argument must be a comma-separated list with no spaces.
//      (i) N_vals gives the Monte Carlo simulation trial counts.

// For each value of N, the code automatically constructs the three Table 3 time-step designs:
//      (i)   approximately 0.1sqrt(N)
//      (ii)  approximately sqrt(N)
//      (iii) approximately 10sqrt(N)

// The code assumes that the i-th value of N_vals corresponds to the i-th row of the Table 3-style output.


// This makes the code adaptable beyond the fixed Broadie-Kaya tables because the user can choose the model parameters and Monte Carlo effort without editing the source code.

// Required command-line inputs:
// scheme_name: Descriptive label for the parameter set. Use underscores instead of spaces.
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
// output_filename: Name of the output CSV file.
// seed: Random seed.
// N_vals: Comma-separated list of Monte Carlo simulation trial counts.

// Output:
// Prints parameter values, the induced Table 3 time-step design, and simulation progress, then writes the final table to CSV.

int main(int argc, char* argv[]) {
    // Create the output directory.
    string output_dir = "Tables";
    filesystem::create_directories(output_dir);

    // This program requires a full command-line specification of the scheme name,
    // model parameters, output settings, and Monte Carlo grid.
    if (argc != 15) {
        cerr << "Invalid number of arguments." << endl;
        cerr << endl;
        cerr << "Usage:" << endl;
        cerr << "  ./heston_table3 scheme_name S_0 K V_0 kappa theta sigma rho r T C_true output_filename seed N_vals" << endl;
        cerr << endl;
        cerr << "Required parameter order:" << endl;
        cerr << "  scheme_name S_0 K V_0 kappa theta sigma rho r T C_true output_filename seed N_vals" << endl;
        cerr << endl;
        cerr << "Experiment grid format:" << endl;
        cerr << "  N_vals must be a comma-separated list with no spaces." << endl;
        cerr << "  Example: 10000,40000,160000" << endl;
        cerr << endl;
        cerr << "Example with custom non-Broadie-Kaya parameters:" << endl;
        cerr << "  ./heston_table3 Custom_Scheme 120 115 0.04 3.50 0.06 0.45 -0.40 0.025 2.0 18.50 custom_table3.csv 98765 5000,20000,80000" << endl;
        return 1;
    }

    // Read the Heston / option parameters from the command line.
    HestonParameters params;
    params.scheme_name = argv[1];
    params.S_0 = stod(argv[2]);
    params.K = stod(argv[3]);
    params.V_0 = stod(argv[4]);
    params.kappa = stod(argv[5]);
    params.theta = stod(argv[6]);
    params.sigma = stod(argv[7]);
    params.rho = stod(argv[8]);
    params.r = stod(argv[9]);
    params.T = stod(argv[10]);
    params.C_true = stod(argv[11]);

    // Read the output filename and random seed.
    string output_filename = argv[12];
    unsigned long long seed = stoull(argv[13]);

    // Parse the comma-separated list of Monte Carlo sizes N.
    vector<long long> N_vals;
    stringstream N_stream(argv[14]);
    string token;

    while (getline(N_stream, token, ',')) {
        N_vals.push_back(stoll(token));
    }

    // Reject an empty experiment grid.
    if (N_vals.empty()) {
        cerr << "Invalid experiment grid." << endl;
        cerr << "N_vals cannot be empty." << endl;
        return 1;
    }

    // Each N must be large enough for the Monte Carlo standard-error formula.
    for (size_t i = 0; i < N_vals.size(); ++i) {
        if (N_vals[i] <= 1) {
            cerr << "Invalid experiment grid." << endl;
            cerr << "Each N value must be greater than 1." << endl;
            return 1;
        }
    }

    // Full path for the CSV output file.
    string output_path = output_dir + "/" + output_filename;

    // Echo the experiment settings before running the Table 3 replication.
    cout << fixed << setprecision(6);
    cout << "Creating Table 3-style Euler bias table." << endl;
    cout << "Parameter values:" << endl;
    cout << "  scheme_name = " << params.scheme_name << endl;
    cout << "  S_0 = " << params.S_0 << endl;
    cout << "  K = " << params.K << endl;
    cout << "  V_0 = " << params.V_0 << endl;
    cout << "  kappa = " << params.kappa << endl;
    cout << "  theta = " << params.theta << endl;
    cout << "  sigma = " << params.sigma << endl;
    cout << "  rho = " << params.rho << endl;
    cout << "  r = " << params.r << endl;
    cout << "  T = " << params.T << endl;
    cout << "  C_true = " << params.C_true << endl;
    cout << "  seed = " << seed << endl;
    cout << "  output = " << output_path << endl;
    cout << "  grid = " << endl;

    // Show, for each N, the three time-step choices used in the Table 3 design:
    // 0.1*sqrt(N), sqrt(N), and 10*sqrt(N).
    for (size_t i = 0; i < N_vals.size(); ++i) {
        int sqrtN = static_cast<int>(sqrt(static_cast<double>(N_vals[i])));
        int steps_0_1_sqrtN = max(1, static_cast<int>(0.1 * sqrtN));
        int steps_sqrtN = max(1, sqrtN);
        int steps_10_sqrtN = max(1, static_cast<int>(10.0 * sqrtN));

        cout << " row " << i + 1
             << ": N = " << N_vals[i]
             << ", Time Steps = "
             << steps_0_1_sqrtN << ", "
             << steps_sqrtN << ", "
             << steps_10_sqrtN
             << endl;
    }

    cout << endl;

    // Construct the Table 3-style bias table for the chosen parameter set.
    vector<Table3Row> rows =
        construct_table3_for_scheme(
            params,
            seed,
            N_vals
        );

    // Write the completed table to CSV.
    write_table3_to_csv(rows, output_path);

    return 0;
}