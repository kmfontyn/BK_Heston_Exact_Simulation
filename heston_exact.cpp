#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>
#include <random>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <sstream>
using namespace std;

const double PI = 3.141592653589793238462643383279502884;


// 1. Data structure: MCResult

// Stores one row of exact Monte Carlo simulation output.

// Fields:
// M: Number of Monte Carlo simulation paths.
// price: Estimated discounted European call option price.
// std_error: Monte Carlo standard error of the estimated price.
// rms_error: Root mean squared error reported for the exact method.
// seconds: Wall-clock computation time needed to produce this row.

// This structure is used to reproduce the Broadie-Kaya-style exact simulation table.

struct MCResult {
    long long M;
    double price;
    double std_error;
    double rms_error;
    double seconds;
};


// 2. Data structure: MomentIV

// Stores conditional moments of the integrated variance.

// Fields:
// mu1: Conditional first moment of the integrated variance.
// mu2: Conditional second moment of the integrated variance.

// These moments are used to initialize the inverse-CDF search for the integrated variance.

struct MomentIV {
    double mu1;
    double mu2;
};


// 3. Data structure: CFOutput

// Stores the output from the branch-corrected characteristic-function calculation.

// Fields:
// f: Complex-valued characteristic-function or Bessel-adjusted output.
// z1: Current complex Bessel-function argument.
// branch1: Current branch-tracking value used for complex continuity.

// This structure is used because the Broadie-Kaya characteristic function involves
// modified Bessel functions with complex arguments, requiring branch tracking.

struct CFOutput {
    complex<double> f;
    complex<double> z1;
    double branch1;
};


// 4. Function: inverse_standard_normal_cdf

// Computes an Acklam inverse normal CDF approximation to the inverse standard normal CDF.

// Inputs:
// p: Probability value in the open interval (0,1).

// Output:
// Approximate z-value satisfying Phi(z) = p, where Phi is the standard normal CDF.

// Purpose:
// Used to construct a normal moment-matched initial guess for sampling the integrated variance.

double inverse_standard_normal_cdf(double p) {
    // The inverse standard normal CDF is only defined for probabilities strictly
    // between 0 and 1. Reject invalid inputs immediately.
    if (p <= 0.0 || p >= 1.0) {
        throw runtime_error("inverse_standard_normal_cdf requires p in (0,1).");
    }

    // Coefficients for the rational approximation in the central region.
    // These constants are from Acklam's approximation for the inverse Gaussian CDF.
    // Ref: https://stackedboxes.org/2017/05/01/acklams-normal-quantile-function/ 

    static const double a1 = -3.969683028665376e+01;
    static const double a2 =  2.209460984245205e+02;
    static const double a3 = -2.759285104469687e+02;
    static const double a4 =  1.383577518672690e+02;
    static const double a5 = -3.066479806614716e+01;
    static const double a6 =  2.506628277459239e+00;

    static const double b1 = -5.447609879822406e+01;
    static const double b2 =  1.615858368580409e+02;
    static const double b3 = -1.556989798598866e+02;
    static const double b4 =  6.680131188771972e+01;
    static const double b5 = -1.328068155288572e+01;

    // Coefficients for the rational approximation in the lower and upper tails.
    // Ref: https://stackedboxes.org/2017/05/01/acklams-normal-quantile-function/
    static const double c1 = -7.784894002430293e-03;
    static const double c2 = -3.223964580411365e-01;
    static const double c3 = -2.400758277161838e+00;
    static const double c4 = -2.549732539343734e+00;
    static const double c5 =  4.374664141464968e+00;
    static const double c6 =  2.938163982698783e+00;

    static const double d1 =  7.784695709041462e-03;
    static const double d2 =  3.224671290700398e-01;
    static const double d3 =  2.445134137142996e+00;
    static const double d4 =  3.754408661907416e+00;

    // Working variables:
    double q, r;

    // Split the probability range into three regions:
    //   1. lower tail
    //   2. central region
    //   3. upper tail
    // Different rational approximations are used in each region for accuracy.
    double plow = 0.02425;
    double phigh = 1.0 - plow;

    // Lower tail: p close to 0
    // Transform p so that the approximation is stable in the tail.
    if (p < plow) {
        q = sqrt(-2.0 * log(p));
        return (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
               ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0);
    }

    // Upper tail: p close to 1
    // Use symmetry of the standard normal distribution:
    //   Φ^{-1}(p) = -Φ^{-1}(1-p)
    if (p > phigh) {
        q = sqrt(-2.0 * log(1.0 - p));
        return -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
                ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0);
    }

    // Central region: p not too close to 0 or 1
    // Approximate using a rational function in q = p - 0.5.
    q = p - 0.5;
    r = q * q;

    return (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q /
           (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0);
}


// 5. Function: modified_bessel_first_kind_real

// Computes the modified Bessel function of the first kind for a real argument.

// Inputs:
// order: Order of the Bessel function.
// x: Real-valued argument.

// Output:
// Approximate value of I_order(x).

// Purpose:
// Used in the denominator of the Broadie-Kaya characteristic function and in
// the conditional moment calculations for integrated variance.

double modified_bessel_first_kind_real(double order, double x) {
    // Handle x = 0 separately.
    if (x == 0.0) {
        if (order == 0.0) return 1.0;
        return 0.0;
    }

    // Treat the real argument as a complex number so the same power-series
    // structure can be used consistently.
    complex<double> z(x, 0.0);

    // First term in the power series for I_order(x).
    complex<double> log_term0 = order * log(z / 2.0) - lgamma(order + 1.0);
    complex<double> term = exp(log_term0);
    complex<double> sum = term;

    // Add successive series terms until the increment is negligible.
    for (int m = 1; m < 10000; ++m) {
        term *= (z * z / 4.0) / (static_cast<double>(m) * (order + static_cast<double>(m)));
        sum += term;

        // Relative stopping rule for numerical convergence.
        if (abs(term) < 1e-14 * max(1.0, abs(sum))) {
            break;
        }
    }

    // Return the real part of the accumulated sum.
    return real(sum);
}


// 6. Function: modified_bessel_first_kind_complex

// Computes the modified Bessel function of the first kind for a complex argument.

// Inputs:
// order: Order of the Bessel function.
// z: Complex-valued argument.

// Output:
// Approximate complex value of I_order(z).

// Purpose:
// Used in the numerator of the Broadie-Kaya conditional characteristic function,
// where the Bessel-function argument is complex.

complex<double> modified_bessel_first_kind_complex(double order, complex<double> z) {
    // Handle z = 0 separately.
    if (abs(z) == 0.0) {
        if (order == 0.0) return complex<double>(1.0, 0.0);
        return complex<double>(0.0, 0.0);
    }

    // First term in the power series for I_order(z).
    complex<double> log_term0 = order * log(z / 2.0) - lgamma(order + 1.0);
    complex<double> term = exp(log_term0);
    complex<double> sum = term;

    // Add successive series terms until the increment is negligible.
    for (int m = 1; m < 10000; ++m) {
        term *= (z * z / 4.0) / (static_cast<double>(m) * (order + static_cast<double>(m)));
        sum += term;

        // Relative stopping rule for numerical convergence.
        if (abs(term) < 1e-14 * max(1.0, abs(sum))) {
            break;
        }
    }

    // Return the complex-valued series approximation.
    return sum;
}


// 7. Function: update_complex_branch

// Updates the branch-tracking value for a complex argument.

// Inputs:
// current_z: Current complex Bessel-function argument.
// previous_z: Previous complex Bessel-function argument.
// previous_branch: Previously accumulated branch correction.

// Output:
// Updated branch correction.

// Purpose:
// Prevents discontinuities caused by evaluating complex logarithms and Bessel functions
// only on their principal branches.

double update_complex_branch(
    complex<double> current_z,
    complex<double> previous_z,
    double previous_branch
) {
    double current_arg = arg(current_z);
    double previous_arg = arg(previous_z);

    if ((current_arg < 0.0) && (previous_arg > 0.0)) {
        return previous_branch + 2.0 * PI;
    }

    return previous_branch;
}


// 8. Function: evaluate_branch_corrected_bessel

// Evaluates the complex modified Bessel function with branch correction.

// Inputs:
// current_z: Current complex Bessel-function argument.
// previous_z: Previous complex Bessel-function argument.
// previous_branch: Previous branch correction.
// order: Order of the modified Bessel function.

// Output:
// CFOutput object containing the corrected Bessel value, current argument, and updated branch.

// Purpose:
// Maintains continuity in the characteristic-function inversion used to sample
// the integrated variance.

CFOutput evaluate_branch_corrected_bessel(
    complex<double> current_z,
    complex<double> previous_z,
    double previous_branch,
    double order
) {
    // Update the running branch correction based on whether the complex argument
    // has crossed the branch cut since the previous evaluation.
    double current_branch = update_complex_branch(
        current_z,
        previous_z,
        previous_branch
    );

    // Evaluate the modified Bessel function at the current complex argument.
    complex<double> raw_bessel =
        modified_bessel_first_kind_complex(order, current_z);

    // Apply the branch correction to preserve continuity across successive
    // characteristic-function evaluations.
    complex<double> corrected_bessel =
        raw_bessel * exp(complex<double>(0.0, -2.0 * PI * current_branch));

    CFOutput out;
    out.f = corrected_bessel;
    out.z1 = current_z;
    out.branch1 = current_branch;

    return out;
}


// 9. Function: sample_noncentral_chi_square

// Samples from a noncentral chi-square distribution using a Poisson-mixture representation.

// Inputs:
// degrees_of_freedom: Base degrees of freedom.
// noncentrality: Noncentrality parameter.
// rng: Random number generator passed by reference.

// Output:
// One sample from a noncentral chi-square distribution.

// Purpose:
// Used to sample the terminal variance in the Heston square-root variance process.

double sample_noncentral_chi_square(
    double degrees_of_freedom,
    double noncentrality,
    mt19937_64& rng
) {
    // Use the Poisson-mixture representation of a noncentral chi-square:
    // if N ~ Poisson(noncentrality / 2), then
    // X | N ~ ChiSquare(degrees_of_freedom + 2N).
    poisson_distribution<int> poisson_draw(noncentrality / 2.0);
    int N = poisson_draw(rng);

    // Adjust the degrees of freedom according to the Poisson draw.
    double adjusted_degrees_of_freedom =
        degrees_of_freedom + 2.0 * static_cast<double>(N);

    // A chi-square(df) variable is Gamma(df / 2, 2).
    gamma_distribution<double> gamma_draw(adjusted_degrees_of_freedom / 2.0, 2.0);

    // Return one draw from the implied noncentral chi-square distribution.
    return gamma_draw(rng);
}


// 10. Function: sample_terminal_variance

// Samples terminal variance V_T from the transition law of the CIR variance process.

// Inputs:
// initial_variance: Initial variance V_0.
// maturity: Time horizon T.
// kappa: Speed of mean reversion.
// theta: Long-run mean variance.
// sigma: Volatility of variance.
// rng: Random number generator passed by reference.

// Output:
// Simulated terminal variance V_T.

// Purpose:
// Implements the first step of the Broadie-Kaya exact simulation algorithm:
// sample V_T conditional on V_0.

double sample_terminal_variance(
    double initial_variance,
    double maturity,
    double kappa,
    double theta,
    double sigma,
    mt19937_64& rng
) {
    // Degrees of freedom in the exact CIR / Heston transition law.
    double degrees_of_freedom = 4.0 * theta * kappa / (sigma * sigma);

    // Common exponential decay term over the interval [0, T].
    double decay = exp(-kappa * maturity);

    // Scale factor in
    //   V_T = scale * ChiSquare'_df(noncentrality).
    double scale = sigma * sigma * (1.0 - decay) / (4.0 * kappa);

    // Broadie–Kaya noncentrality parameter:
    //   lambda = [4 kappa e^{-kappa T} / (sigma^2 (1 - e^{-kappa T}))] * V_0
    double noncentrality =
        (4.0 * kappa * decay * initial_variance)
        / (sigma * sigma * (1.0 - decay));

    // Sample from the exact transition distribution of V_T | V_0.
    return scale * sample_noncentral_chi_square(
        degrees_of_freedom,
        noncentrality,
        rng
    );
}

// 11. Function: compute_integrated_variance_moments

// Computes conditional moments of the integrated variance.

// Method:
// The function computes
//   (i) mu1 = E[∫_0^T V_s ds | V_0, V_T]
//   (ii) mu2 = E[(∫_0^T V_s ds)^2 | V_0, V_T]
// using closed-form expressions implied by the CIR bridge structure of the
// Heston variance process. The calculation uses the model parameters
// (kappa, theta, sigma), the endpoint variances (V_0, V_T), the exponential
// decay term exp(-kappa * T), and modified Bessel functions evaluated at the
// bridge-specific argument. Nearby Bessel orders are also used to form the
// derivative terms that appear in the moment formulas. The resulting first
// and second conditional moments are then returned for use in the moment-
// matched normal approximation that initializes inverse-CDF sampling.

// Inputs:
// terminal_variance: Simulated terminal variance V_T.
// initial_variance: Initial variance V_0.
// maturity: Time horizon T.
// kappa: Speed of mean reversion.
// theta: Long-run mean variance.
// sigma: Volatility of variance.

// Output:
// MomentIV object containing:
// mu1: Conditional first moment of integrated variance.
// mu2: Conditional second moment of integrated variance.

// Purpose:
// These moments are used to approximate the conditional integrated-variance
// distribution by a normal distribution, giving an initial value for Newton inversion.

MomentIV compute_integrated_variance_moments(
    double terminal_variance,
    double initial_variance,
    double maturity,
    double kappa,
    double theta,
    double sigma
) {
    // Parameters appearing in the conditional law of the integrated variance.
    double d = 4.0 * theta * kappa / (sigma * sigma);
    double order = 0.5 * d - 1.0;

    double decay = exp(-kappa * maturity);

    // Bessel-function argument implied by the CIR bridge structure.
    double x = sqrt(initial_variance * terminal_variance)
        * 4.0 * kappa * exp(-0.5 * kappa * maturity)
        / (sigma * sigma * (1.0 - decay));

    // Evaluate nearby Bessel terms needed for the derivative identities.
    double I_order =
        modified_bessel_first_kind_real(order, x);
    double I_order_minus_1 =
        modified_bessel_first_kind_real(order - 1.0, x);
    double I_order_plus_1 =
        modified_bessel_first_kind_real(order + 1.0, x);
    double I_order_minus_2 =
        modified_bessel_first_kind_real(order - 2.0, x);
    double I_order_plus_2 =
        modified_bessel_first_kind_real(order + 2.0, x);

    // First and second derivative terms of the Bessel ratio entering the moments.
    double first_bessel_ratio_derivative =
        I_order_minus_1 / (2.0 * I_order) + I_order_plus_1 / (2.0 * I_order);

    double second_bessel_ratio_derivative =
        I_order_minus_2 / (4.0 * I_order)
        + I_order / (2.0 * I_order)
        + I_order_plus_2 / (4.0 * I_order);

    // Components of the conditional first moment.
    double df1 = (sigma * sigma / kappa)
        * (-1.0 / kappa + maturity / 2.0 + maturity * decay / (1.0 - decay));

    double df2 = (initial_variance + terminal_variance) * (
        (1.0 / kappa) * (1.0 + decay) / (1.0 - decay)
        - 2.0 * maturity * decay / pow(1.0 - decay, 2.0)
    );

    double df3 = (-4.0 / kappa + 2.0 * maturity) * exp(-0.5 * kappa * maturity)
        + 4.0 * maturity * exp(-1.5 * kappa * maturity) / (1.0 - decay);

    df3 = (df3 / (1.0 - decay))
        * sqrt(initial_variance * terminal_variance)
        * first_bessel_ratio_derivative;

    double first_moment = df1 + df2 + df3;

    // Components used in the conditional second moment calculation.
    double ddf1 = (
        1.0 / (kappa * kappa)
        + maturity / (2.0 * kappa)
        - maturity * maturity / 4.0
        + (maturity / kappa - 2.0 * maturity * maturity) * decay / (1.0 - decay)
    );

    ddf1 = ddf1
        - 2.0 * maturity * maturity * decay * decay / pow(1.0 - decay, 2.0);

    ddf1 = ddf1 * pow(sigma, 4.0) / (kappa * kappa);

    double t1 = (1.0 + decay) / (kappa * kappa)
        + maturity * decay / kappa
        - maturity * maturity * decay;

    double t2 = (
        maturity * (1.0 + decay) * decay / kappa
        - 3.0 * maturity * maturity * decay * decay
        - maturity * maturity * decay
    ) / (1.0 - decay);

    double t3 = 2.0 * maturity * maturity * decay * decay * (1.0 + decay)
        / pow(1.0 - decay, 2.0);

    double ddf2 = -df2 * df2
        - ((initial_variance + terminal_variance) * sigma * sigma / (kappa * (1.0 - decay)))
        * (t1 + t2 - t3);

    double dz = (-4.0 / kappa + 2.0 * maturity) * exp(-0.5 * kappa * maturity)
        + 4.0 * maturity * exp(-1.5 * kappa * maturity) / (1.0 - decay);

    dz = dz * sqrt(initial_variance * terminal_variance) / (1.0 - decay);

    t1 = (4.0 / (kappa * kappa) + 2.0 * maturity / kappa - maturity * maturity)
        * exp(-0.5 * kappa * maturity);

    t2 = (4.0 * maturity / kappa - 8.0 * maturity * maturity)
        * exp(-1.5 * kappa * maturity) / (1.0 - decay);

    t3 = 8.0 * maturity * exp(-2.5 * kappa * maturity)
        / pow(1.0 - decay, 2.0);

    double ddz = sqrt(initial_variance * terminal_variance)
        * sigma * sigma / (kappa * (1.0 - decay))
        * (t1 + t2 - t3);

    double ddf3 = -dz * dz * second_bessel_ratio_derivative
        + first_bessel_ratio_derivative * ddz;

    // Assemble the conditional second moment.
    double second_moment = ddf1 + ddf2 + ddf3
        - 2.0 * (df1 * df2 + df1 * df3 + df2 * df3);

    second_moment = -second_moment;

    // Return the first two conditional moments of the integrated variance.
    MomentIV out;
    out.mu1 = first_moment;
    out.mu2 = second_moment;
    return out;
}


// 12. Function: evaluate_integrated_variance_cf

// Evaluates the conditional characteristic function of integrated variance.

// Inputs:
// frequency: Fourier frequency at which the characteristic function is evaluated.
// previous_z: Previous complex Bessel argument.
// previous_branch: Previous branch correction.
// initial_variance: Initial variance V_0.
// terminal_variance: Terminal variance V_T.
// maturity: Time horizon T.
// kappa: Speed of variance mean reversion.
// theta: Long-run mean variance.
// sigma: Volatility of variance.

// Output:
// CFOutput object containing the characteristic-function value, current Bessel argument,
// and updated branch correction.

// Purpose:
// Implements the core Broadie-Kaya Fourier inversion object for sampling
// integrated variance conditional on V_0 and V_T.

CFOutput evaluate_integrated_variance_cf(
    double frequency,
    complex<double> previous_z,
    double previous_branch,
    double initial_variance,
    double terminal_variance,
    double maturity,
    double kappa,
    double theta,
    double sigma
) {
    complex<double> imaginary_unit(0.0, 1.0);

    // Complex gamma term appearing in the conditional characteristic function.
    complex<double> gamma_term =
        sqrt(complex<double>(kappa * kappa, 0.0)
        - 2.0 * sigma * sigma * frequency * imaginary_unit);

    // Real and complex decay factors over the maturity horizon.
    double decay_kappa = exp(-kappa * maturity);
    complex<double> decay_gamma = exp(-gamma_term * maturity);

    // Leading multiplicative term in the characteristic function.
    complex<double> cf_value =
        (gamma_term / kappa)
        * exp(-0.5 * (gamma_term - kappa) * maturity)
        * ((1.0 - decay_kappa) / (1.0 - decay_gamma));

    // Exponential term involving the endpoint variances V_0 and V_T.
    double variance_scale = (initial_variance + terminal_variance) / (sigma * sigma);

    complex<double> exponential_term =
        kappa * ((1.0 + decay_kappa) / (1.0 - decay_kappa))
        - gamma_term * ((1.0 + decay_gamma) / (1.0 - decay_gamma));

    cf_value *= exp(variance_scale * exponential_term);

    // Order of the modified Bessel function in the CIR bridge formula.
    double bessel_order = 2.0 * theta * kappa / (sigma * sigma) - 1.0;

    // Complex numerator argument and real denominator argument for the Bessel ratio.
    complex<double> numerator_argument =
        sqrt(initial_variance * terminal_variance)
        * (4.0 * gamma_term / (sigma * sigma))
        * exp(-0.5 * gamma_term * maturity)
        / (1.0 - decay_gamma);

    double denominator_argument =
        sqrt(initial_variance * terminal_variance)
        * (4.0 * kappa / (sigma * sigma))
        * exp(-0.5 * kappa * maturity)
        / (1.0 - decay_kappa);

    // Evaluate the numerator Bessel term with branch correction to preserve continuity
    // across successive characteristic-function evaluations.
    CFOutput bessel_numerator =
        evaluate_branch_corrected_bessel(
            numerator_argument,
            previous_z,
            previous_branch,
            bessel_order
        );

    // Evaluate the real-valued denominator Bessel term.
    double bessel_denominator =
        modified_bessel_first_kind_real(bessel_order, denominator_argument);

    // Complete the characteristic function by multiplying by the Bessel ratio.
    cf_value *= bessel_numerator.f / bessel_denominator;

    // Return the characteristic-function value together with the updated complex
    // argument and branch tracker for the next frequency evaluation.
    CFOutput out;
    out.f = cf_value;
    out.z1 = numerator_argument;
    out.branch1 = bessel_numerator.branch1;

    return out;
}


// 13. Function: compute_truncated_cf_grid

// Computes a finite grid of characteristic-function values for Fourier inversion.

// Inputs:
// grid_spacing: Fourier grid spacing h.
// initial_variance: Initial variance V_0.
// terminal_variance: Terminal variance V_T.
// maturity: Time horizon T.
// kappa: Speed of variance mean reversion.
// theta: Long-run mean variance.
// sigma: Volatility of variance.
// tolerance: Truncation tolerance for stopping the Fourier sum.

// Output:
// Vector of complex characteristic-function values evaluated at h, 2h, 3h, ...

// Purpose:
// Supplies the finite Fourier series used to numerically recover the integrated-variance CDF.

vector<complex<double>> compute_truncated_cf_grid(
    double grid_spacing,
    double initial_variance,
    double terminal_variance,
    double maturity,
    double kappa,
    double theta,
    double sigma,
    double tolerance = 1e-5
) {
    // Initialize the branch-tracking inputs for the first characteristic-function evaluation.
    complex<double> previous_z(1.0, 1.0);
    double previous_branch = 0.0;

    // Evaluate the characteristic function at the first grid point h.
    CFOutput out =
        evaluate_integrated_variance_cf(
            grid_spacing,
            previous_z,
            previous_branch,
            initial_variance,
            terminal_variance,
            maturity,
            kappa,
            theta,
            sigma
        );

    complex<double> current_cf = out.f;
    complex<double> current_z = out.z1;
    double current_branch = out.branch1;

    int j = 1;
    vector<complex<double>> cf_values;
    cf_values.push_back(current_cf);

    // Keep adding characteristic-function values at frequencies j*h until the
    // truncation rule implies that the remaining Fourier terms are negligible.
    while (abs(current_cf) / static_cast<double>(j) >= PI * tolerance / 2.0) {
        j += 1;

        // Pass the most recent complex argument and branch to preserve continuity
        // in the Bessel evaluation at the next frequency.
        previous_z = current_z;
        previous_branch = current_branch;

        out =
            evaluate_integrated_variance_cf(
                static_cast<double>(j) * grid_spacing,
                previous_z,
                previous_branch,
                initial_variance,
                terminal_variance,
                maturity,
                kappa,
                theta,
                sigma
            );

        current_cf = out.f;
        current_z = out.z1;
        current_branch = out.branch1;

        // Store the characteristic-function value at frequency j*h.
        cf_values.push_back(current_cf);

        // Hard stop to prevent an infinite loop if numerical decay is too slow.
        if (j > 100000) {
            break;
        }
    }

    // Return the truncated grid of characteristic-function values used in Fourier inversion.
    return cf_values;
}


// 14. Function: sample_integrated_variance

// Samples integrated variance conditional on initial and terminal variance.

// Inputs:
// terminal_variance: Terminal variance V_T.
// initial_variance: Initial variance V_0.
// maturity: Time horizon T.
// kappa: Speed of variance mean reversion.
// theta: Long-run mean variance.
// sigma: Volatility of variance.
// rng: Random number generator passed by reference.

// Output:
// Simulated integrated variance, approximating int_0^T V_s ds conditional on V_0 and V_T.

// Purpose:
// Implements the most numerically intensive step of the Broadie-Kaya method:
// Fourier inversion of the conditional integrated-variance distribution, followed
// by inverse transform sampling.

double sample_integrated_variance(
    double terminal_variance,
    double initial_variance,
    double maturity,
    double kappa,
    double theta,
    double sigma,
    mt19937_64& rng
) {
    // For extremely short maturities, avoid the expensive inversion routine.
    // In this regime, the time-integrated variance is well approximated by
    // the trapezoidal average of the initial and terminal variance over [0,T].
    if (maturity < 1e-4) {
        return maturity * (initial_variance + terminal_variance) / 2.0;
    }

    // Compute the first two conditional moments of the integrated variance:
    //   mu1 = E[∫ V_s ds | V_0, V_T]
    //   mu2 = E[(∫ V_s ds)^2 | V_0, V_T]
    // These moments are used to build a normal approximation that initializes
    // the inverse-CDF root search.
    MomentIV moments =
        compute_integrated_variance_moments(
            terminal_variance,
            initial_variance,
            maturity,
            kappa,
            theta,
            sigma
        );

    double mean_iv = moments.mu1;
    double second_moment_iv = moments.mu2;
    double variance_iv = second_moment_iv - mean_iv * mean_iv;

    // If the moment calculation fails numerically, or implies a non-positive
    // variance, fall back to the same simple trapezoidal approximation.
    if (!isfinite(mean_iv) || !isfinite(second_moment_iv) || variance_iv <= 0.0) {
        return maturity * (initial_variance + terminal_variance) / 2.0;
    }

    // Standard deviation implied by the conditional moments.
    double sd_iv = sqrt(variance_iv);

    // Upper scale used when constructing the Fourier inversion grid.
    // Mean + 5 SD provides a right-tail search range.
    double upper_scale = mean_iv + 5.0 * sd_iv;

    // Draw U ~ Uniform(0,1). We will invert the conditional CDF at this value
    // to obtain one sample from the integrated variance distribution.
    uniform_real_distribution<double> uniform01(0.0, 1.0);
    double target_probability = uniform01(rng);

    // Use a moment-matched normal approximation to initialize the search point.
    double x0 = mean_iv + sd_iv * inverse_standard_normal_cdf(target_probability);
    double x0_original = x0;

    // Integrated variance must be nonnegative; if the normal guess is negative,
    // replace it by a small positive value.
    x0 = (x0 > 0.0) ? x0 : 0.01 * mean_iv;

    bool newton_converged = false;
    int newton_iterations = 0;
    double x = x0;

    // Attempt inverse-CDF solving with second-order Newton iteration.
    // The CDF is evaluated numerically using the truncated Fourier series.
    while (true) {
        newton_iterations += 1;

        // Choose the Fourier grid spacing h = 2π / (x + upper bound).
        // This follows the inversion setup used to approximate the CDF.
        double grid_spacing = 2.0 * PI / (x0 + upper_scale);

        // Evaluate the characteristic function on the truncated Fourier grid.
        // These values drive the numerical CDF, derivative, and curvature.
        vector<complex<double>> cf_values =
            compute_truncated_cf_grid(
                grid_spacing,
                initial_variance,
                terminal_variance,
                maturity,
                kappa,
                theta,
                sigma,
                1e-5
            );

        // Build:
        //  cdf_error= F(x0) - U
        //  cdf_derivative= F'(x0)
        //  cdf_second_derivative= F''(x0)
        //  using the Fourier inversion formula and its derivatives.
        double cdf_error = grid_spacing * x0 / PI - target_probability;
        double cdf_derivative = grid_spacing / PI;
        double cdf_second_derivative = 0.0;

        for (size_t idx = 0; idx < cf_values.size(); ++idx) {
            int j = static_cast<int>(idx) + 1;
            double hj = grid_spacing * static_cast<double>(j);
            double cf_real_part = real(cf_values[idx]);

            cdf_error += (2.0 / PI)
                * (sin(hj * x0) / static_cast<double>(j))
                * cf_real_part;

            cdf_derivative += (2.0 / PI)
                * cos(hj * x0)
                * grid_spacing
                * cf_real_part;

            cdf_second_derivative += -(2.0 / PI)
                * sin(hj * x0)
                * hj
                * grid_spacing
                * cf_real_part;
        }

        // If the slope or curvature is too small, Newton updates become unstable.
        // In that case, abandon Newton and switch to bisection later.
        if (abs(cdf_derivative) < 1e-14 || abs(cdf_second_derivative) < 1e-14) {
            newton_converged = false;
            x = x0;
            break;
        }

        // Second-order Newton update:
        // x_{n+1} = x_n - (F'/F'') [1 - sqrt(1 - 2(F-U)F''/(F')^2)]
        double newton_discriminant =
            1.0 - 2.0 * cdf_error * cdf_second_derivative
            / (cdf_derivative * cdf_derivative);

        if (newton_discriminant >= 0.0) {
            x = x0
                - (cdf_derivative / cdf_second_derivative)
                * (1.0 - sqrt(newton_discriminant));

            // If the update leaves the admissible region or becomes non-finite,
            // stop Newton and fall back to a safer bracketing method.
            if (x < 0.0 || !isfinite(x)) {
                newton_converged = false;
                break;
            }
        } else {
            // A negative discriminant indicates numerical instability in the
            // higher-order update, so revert to bisection.
            newton_converged = false;
            x = x0;
            break;
        }

        // Stop if successive iterates are close enough, or if Newton has
        // already been tried sufficiently many times.
        if (abs(x - x0) < 1e-5 || newton_iterations > 10) {
            newton_converged = true;
            break;
        }

        // Continue Newton iteration from the updated point.
        x0 = x;
    }

    // If Newton fails, use bisection as a robust fallback.
    if (!newton_converged) {
        double lower_bound = 0.0;
        double upper_bound = mean_iv + 5.0 * sd_iv;

        double left, right;

        // Choose an initial bracket based on where U lies in the distribution.
        // Lower-tail probabilities search closer to zero; upper-tail probabilities
        // search further to the right; central probabilities search around the
        // original normal-based initialization.
        if (target_probability < 0.1) {
            left = lower_bound;
            right = mean_iv - sd_iv;

            if (right < 0.0) {
                double n_sd = mean_iv / sd_iv;
                right = mean_iv - (n_sd / 2.0) * sd_iv;
            }
        } else if (target_probability > 0.9) {
            left = mean_iv + 2.0 * sd_iv;
            right = upper_bound;
        } else {
            left = max(x0_original - 2.0 * sd_iv, lower_bound);
            right = min(x0_original + 2.0 * sd_iv, upper_bound);
        }

        // Enforce a valid nonnegative bracket with positive width.
        left = max(left, 0.0);
        right = max(right, left + 1e-10);

        x = 0.5 * (left + right);

        int bisection_iterations = 0;

        // Standard bisection loop: repeatedly halve the bracket until the
        // implied CDF value matches the target probability closely enough.
        while ((right - left) > 1e-5 && bisection_iterations < 100) {
            bisection_iterations += 1;
            x = 0.5 * (left + right);

            // Rebuild the Fourier inversion grid at the current midpoint.
            double grid_spacing = 2.0 * PI / (x + upper_scale);

            vector<complex<double>> cf_values =
                compute_truncated_cf_grid(
                    grid_spacing,
                    initial_variance,
                    terminal_variance,
                    maturity,
                    kappa,
                    theta,
                    sigma,
                    1e-5
                );

            // Evaluate the conditional CDF F(x) at the current midpoint.
            double cdf_value = grid_spacing * x / PI;

            for (size_t idx = 0; idx < cf_values.size(); ++idx) {
                int j = static_cast<int>(idx) + 1;
                double hj = grid_spacing * static_cast<double>(j);
                double cf_real_part = real(cf_values[idx]);

                cdf_value += (2.0 / PI)
                    * (sin(hj * x) / static_cast<double>(j))
                    * cf_real_part;
            }

            // Narrow the bracket depending on whether F(x) lies below or above U.
            if (abs(cdf_value - target_probability) < 1e-5) {
                break;
            } else if (cdf_value < target_probability) {
                left = x;
            } else {
                right = x;
            }
        }
    }
    return x;
}


// 15. Function: sample_terminal_stock_price_exact

// Samples the terminal stock price S_T using the exact Heston simulation structure.

// Inputs:
// initial_stock_price: Initial stock price S_0.
// initial_variance: Initial variance V_0.
// maturity: Time horizon T.
// risk_free_rate: Risk-free rate r.
// kappa: Speed of variance mean reversion.
// theta: Long-run mean variance.
// sigma: Volatility of variance.
// rho: Correlation between stock and variance shocks.
// rng: Random number generator passed by reference.

// Output:
// Simulated terminal stock price S_T.

// Purpose:
// Implements the Broadie-Kaya conditional simulation logic:
// sample V_T, sample integrated variance, recover the variance Brownian integral,
// and then sample S_T from its conditional lognormal distribution.

double sample_terminal_stock_price_exact(
    double initial_stock_price,
    double initial_variance,
    double maturity,
    double risk_free_rate,
    double kappa,
    double theta,
    double sigma,
    double rho,
    mt19937_64& rng
) {
    normal_distribution<double> standard_normal(0.0, 1.0);

    // Step (1) Sample the terminal Variance
    double terminal_variance =
        sample_terminal_variance(
            initial_variance,
            maturity,
            kappa,
            theta,
            sigma,
            rng
        );
    
    // Step (2) Sample the integrated variancew
    double integrated_variance =
        sample_integrated_variance(
            terminal_variance,
            initial_variance,
            maturity,
            kappa,
            theta,
            sigma,
            rng
        );

    // Step (3) Recover the brownian increment
    double variance_brownian_integral =
        (terminal_variance
        - initial_variance
        - kappa * theta * maturity
        + kappa * integrated_variance) / sigma;

    // Step (4) Recover moments of the terminal asset price, and compute the terminal value
    double conditional_log_mean =
        log(initial_stock_price)
        + risk_free_rate * maturity
        - 0.5 * integrated_variance
        + rho * variance_brownian_integral;

    double conditional_log_variance =
        (1.0 - rho * rho) * integrated_variance;

    if (conditional_log_variance < 0.0) {
        conditional_log_variance = 0.0;
    }

    double normal_draw = standard_normal(rng);

    return exp(conditional_log_mean + sqrt(conditional_log_variance) * normal_draw);
}


// 16. Function: estimate_call_price_exact

// Estimates a European call option price using the exact Heston simulation method.

// Inputs:
// number_of_paths: Number of Monte Carlo simulation paths.
// initial_stock_price: Initial stock price S_0.
// strike_price: Strike price K.
// initial_variance: Initial variance V_0.
// kappa: Speed of variance mean reversion.
// theta: Long-run mean variance.
// sigma: Volatility of variance.
// rho: Correlation between stock and variance shocks.
// risk_free_rate: Risk-free rate r.
// maturity: Option maturity T.
// benchmark_price: Benchmark true price, included for comparison.
// seed: Random seed.

// Output:
// MCResult object containing number of paths, estimated price, standard error,
// RMS error, and computation time placeholder.

// Purpose:
// Repeats the exact terminal-stock-price sampler over many paths, discounts the
// average payoff, and reports Monte Carlo uncertainty.

MCResult estimate_call_price_exact(
    long long number_of_paths,
    double initial_stock_price,
    double strike_price,
    double initial_variance,
    double kappa,
    double theta,
    double sigma,
    double rho,
    double risk_free_rate,
    double maturity,
    double benchmark_price,
    unsigned long long seed
) {
    // Initialize the random number generator for reproducible Monte Carlo simulation.
    mt19937_64 rng(seed);

    // Risk-neutral discount factor used to convert expected payoff to present value.
    double discount_factor = exp(-risk_free_rate * maturity);

    double payoff_sum = 0.0;
    double payoff_squared_sum = 0.0;

    // Simulate terminal stock prices under the exact Heston scheme and accumulate
    // discounted European call payoffs.
    for (long long i = 0; i < number_of_paths; ++i) {
        double terminal_stock_price =
            sample_terminal_stock_price_exact(
                initial_stock_price,
                initial_variance,
                maturity,
                risk_free_rate,
                kappa,
                theta,
                sigma,
                rho,
                rng
            );

        // European call payoff: max(S_T - K, 0).
        double payoff = max(terminal_stock_price - strike_price, 0.0);

        payoff_sum += payoff;
        payoff_squared_sum += payoff * payoff;
    }

    // Monte Carlo estimate of the undiscounted expected payoff.
    double mean_payoff = payoff_sum / static_cast<double>(number_of_paths);

    // Present-value option price estimate.
    double price = discount_factor * mean_payoff;

    // Unbiased sample variance of the simulated payoff values.
    double sample_variance_payoff =
        (payoff_squared_sum
        - static_cast<double>(number_of_paths) * mean_payoff * mean_payoff)
        / static_cast<double>(number_of_paths - 1);

    // Guard against tiny negative values caused by floating-point roundoff.
    if (sample_variance_payoff < 0.0) {
        sample_variance_payoff = 0.0;
    }

    // Standard error of the discounted Monte Carlo estimator.
    double standard_error =
        discount_factor * sqrt(sample_variance_payoff / static_cast<double>(number_of_paths));

    // Under exact simulation, discretization bias is removed, so the reported RMS error
    // is taken to be the Monte Carlo standard error.
    double rms_error = standard_error;

    MCResult out;
    out.M = number_of_paths;
    out.price = price;
    out.std_error = standard_error;
    out.rms_error = rms_error;
    out.seconds = 0.0;

    return out;
}


// 17. Function: write_exact_results_to_csv

// Writes exact-method simulation results to a CSV file.

// Inputs:
// results: Vector of MCResult objects.
// output_path: Full file path for the output CSV.

// Output:
// None directly; writes a CSV file to disk.

// Side effects:
// Creates or overwrites the specified CSV file and prints a confirmation message.

void write_exact_results_to_csv(
    const vector<MCResult>& results,
    const string& output_path
) {
    ofstream file(output_path);

    if (!file.is_open()) {
        cerr << "Could not open output file: " << output_path << endl;
        return;
    }

    // The exact-method table reports only the number of simulation trials,
    // RMS error, and computation time.
    file << "Number of Trials,RMS Error (SE),Time (Sec)\n";
    file << fixed << setprecision(10);

    for (const auto& row : results) {
        file
            << row.M << ","
            << row.rms_error << ","
            << row.seconds << "\n";
    }

    file.close();

    cout << "Table written to: " << output_path << endl;
}


// 18. Function: create_exact_simulation_table

// Constructs the table of results as they are presented in Broadie & Kaya.

// Inputs:
// initial_stock_price: Initial stock price S_0.
// strike_price: Strike price K.
// initial_variance: Initial variance V_0.
// kappa: Speed of variance mean reversion.
// theta: Long-run mean variance.
// sigma: Volatility of variance.
// rho: Correlation between stock and variance shocks.
// risk_free_rate: Risk-free rate r.
// maturity: Option maturity T.
// benchmark_price: Benchmark true option price.
// seed: Base random seed.
// small_test: If true, uses a small grid {100, 500, 1000}; otherwise uses the full table grid (Used in development)

// Output:
// Vector of MCResult objects, one per simulation size.

// Purpose:
// Runs exact Monte Carlo estimation for each specified sample size and records
// price, standard error, RMS error, and computation time.

vector<MCResult> create_exact_simulation_table(
    double initial_stock_price,
    double strike_price,
    double initial_variance,
    double kappa,
    double theta,
    double sigma,
    double rho,
    double risk_free_rate,
    double maturity,
    double benchmark_price,
    unsigned long long seed,
    bool small_test = false,
    vector<long long> simulation_sizes = {}
) {
    // If the user does not provide simulation sizes, use either a small diagnostic
    // grid or the full Broadie–Kaya replication grid.
    if (simulation_sizes.empty()) {
        if (small_test) {
            simulation_sizes = {100, 500, 1000};
        } else {
            simulation_sizes = {10000, 40000, 160000, 640000, 2560000, 10240000};
        }
    }

    // Store one MCResult row for each Monte Carlo path count.
    vector<MCResult> results;

    for (size_t i = 0; i < simulation_sizes.size(); ++i) {
        long long number_of_paths = simulation_sizes[i];

        // Report progress before starting the current simulation size.
        cout << "Running exact method: M = " << number_of_paths << "..." << endl;

        auto start = chrono::high_resolution_clock::now();

        // Estimate the exact-simulation call price and associated Monte Carlo error.
        MCResult row =
            estimate_call_price_exact(
                number_of_paths,
                initial_stock_price,
                strike_price,
                initial_variance,
                kappa,
                theta,
                sigma,
                rho,
                risk_free_rate,
                maturity,
                benchmark_price,
                seed + static_cast<unsigned long long>(i)
            );

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;

        // Record the wall-clock runtime for this simulation size.
        row.seconds = elapsed.count();

        results.push_back(row);

        // Print a summary line for the completed run.
        cout << fixed << setprecision(6);
        cout << "Done: Price = " << row.price
             << ", SE = " << row.std_error
             << ", RMS = " << row.rms_error
             << ", Time = " << row.seconds << " sec"
             << endl << endl;
    }

    // Return the full table of exact-simulation results.
    return results;
}


// 19. Function: main

// Program entry point.

// Purpose:
// Reads optional command-line parameters, sets default Broadie-Kaya Table 1 exact-method parameters if none are supplied, constructs the exact simulation table, and writes the table to CSV.

// Command-line usage (i.e. to implement the full procedure of creating the table):

// To implement the full procedure of creating the table, the user must supply
// custom parameter values with a user-designed experiment grid in the form:

// ./heston_exact S K V0 kappa theta sigma rho r T benchmark_price output_filename seed M_vals

// The experiment-grid argument must be a comma-separated list with no spaces.
//      (i) M_vals gives the Monte Carlo path counts.

// Unlike the Euler code, the exact-method code does not require time_steps_vals because the Broadie-Kaya exact method does not discretize the Heston process over an Euler time grid.

// The code assumes that the i-th value of M_vals corresponds to the i-th row of the exact Monte Carlo table.


// This makes the code adaptable beyond the fixed Broadie-Kaya tables because the user can choose the Monte Carlo effort without editing the source code.

// Required command-line inputs:
// S: Initial stock price.
// K: Strike price.
// V0: Initial variance.
// kappa: Speed of variance mean reversion.
// theta: Long-run mean variance.
// sigma: Volatility of variance.
// rho: Correlation between stock and variance shocks.
// r: Risk-free rate.
// T: Maturity in years.
// benchmark_price: Benchmark true call option price.
// Optional command-line inputs:
// output_filename: Name of the output CSV file.
// seed: Random seed.
// M_vals: Comma-separated list of Monte Carlo path counts.

// Output:
// Prints parameter values and simulation progress, then writes the final table to CSV.

int main(int argc, char* argv[]) {
    // Create the output directory if it does not already exist.
    string output_dir = "Tables";
    filesystem::create_directories(output_dir);

    // Default Heston / option parameters corresponding to the baseline experiment 
    // (If none are supplied by the user).
    double initial_stock_price = 100.0;
    double strike_price = 100.0;
    double initial_variance = 0.010201;
    double kappa = 6.21;
    double theta = 0.019;
    double sigma = 0.61;
    double rho = -0.70;
    double risk_free_rate = 0.0319;
    double maturity = 1.0;
    double benchmark_price = 6.8061;

    // Default output and simulation controls.
    string output_filename = "broadie_kaya_exact_table.csv";
    unsigned long long seed = 12345;
    bool small_test = false;

    // Default Broadie–Kaya-style Monte Carlo grid.
    vector<long long> simulation_sizes = {10000, 40000, 160000, 640000, 2560000, 10240000};

    // No command-line arguments: use all defaults.
    if (argc == 1) {
        cout << "Using default parameter values." << endl;
        cout << "Using default Broadie-Kaya-style exact simulation grid." << endl;

    // Full parameter input, with optional final argument for either a small test
    // run or a user-specified Monte Carlo grid (Overwrites the previous parameters
    // in the event that they are given).
    } else if (argc == 13 || argc == 14) {
        initial_stock_price = stod(argv[1]);
        strike_price = stod(argv[2]);
        initial_variance = stod(argv[3]);
        kappa = stod(argv[4]);
        theta = stod(argv[5]);
        sigma = stod(argv[6]);
        rho = stod(argv[7]);
        risk_free_rate = stod(argv[8]);
        maturity = stod(argv[9]);
        benchmark_price = stod(argv[10]);
        output_filename = argv[11];
        seed = stoull(argv[12]);

        if (argc == 14) {
            string final_argument = argv[13];

            // Small diagnostic grid for quick checking.
            if (final_argument == "test") {
                small_test = true;
                simulation_sizes = {100, 500, 1000};
                cout << "Using command-line parameter values." << endl;
                cout << "Using small test grid." << endl;
            } else {
                // Otherwise, interpret the final argument as a comma-separated
                // list of Monte Carlo path counts.
                simulation_sizes.clear();

                stringstream M_stream(final_argument);
                string token;

                while (getline(M_stream, token, ',')) {
                    simulation_sizes.push_back(stoll(token));
                }

                // Reject an empty custom grid.
                if (simulation_sizes.empty()) {
                    cerr << "Invalid experiment grid." << endl;
                    cerr << "M_vals cannot be empty." << endl;
                    return 1;
                }

                cout << "Using command-line parameter values." << endl;
                cout << "Using user-specified exact simulation grid." << endl;
            }
        } else {
            // Full parameter input provided, but no custom grid: use the default grid.
            cout << "Using command-line parameter values." << endl;
            cout << "Using default Broadie-Kaya-style exact simulation grid." << endl;
        }
    } else {
        // Any other argument count is invalid; print usage instructions.
        cerr << "Invalid arguments." << endl;
        cerr << endl;
        cerr << "Usage:" << endl;
        cerr << "  ./heston_exact" << endl;
        cerr << endl;
        cerr << "or:" << endl;
        cerr << "  ./heston_exact S K V0 kappa theta sigma rho r T benchmark_price output_filename seed" << endl;
        cerr << endl;
        cerr << "or:" << endl;
        cerr << "  ./heston_exact S K V0 kappa theta sigma rho r T benchmark_price output_filename seed test" << endl;
        cerr << endl;
        cerr << "or:" << endl;
        cerr << "  ./heston_exact S K V0 kappa theta sigma rho r T benchmark_price output_filename seed M_vals" << endl;
        cerr << endl;
        cerr << "Experiment grid format:" << endl;
        cerr << "  M_vals must be a comma-separated list with no spaces." << endl;
        cerr << "  Example: 10000,50000,100000" << endl;
        cerr << endl;
        cerr << "Example with custom exact simulation grid:" << endl;
        cerr << "  ./heston_exact 100 100 0.010201 6.21 0.019 0.61 -0.70 0.0319 1.0 6.8061 exact_custom.csv 12345 10000,50000,100000" << endl;
        return 1;
    }

    // Full output path for the CSV file.
    string output_path = output_dir + "/" + output_filename;

    // Echo the experiment settings before running the simulation.
    cout << fixed << setprecision(6);
    cout << "Parameters:" << endl;
    cout << "S = " << initial_stock_price << endl;
    cout << "K = " << strike_price << endl;
    cout << "V0 = " << initial_variance << endl;
    cout << "kappa = " << kappa << endl;
    cout << "theta = " << theta << endl;
    cout << "sigma = " << sigma << endl;
    cout << "rho = " << rho << endl;
    cout << "r = " << risk_free_rate << endl;
    cout << "T = " << maturity << endl;
    cout << "benchmark_price = " << benchmark_price << endl;
    cout << "seed = " << seed << endl;
    cout << "output = " << output_path << endl;
    cout << "small_test = " << (small_test ? "true" : "false") << endl;
    cout << "grid = " << endl;
    for (size_t i = 0; i < simulation_sizes.size(); ++i) {
        cout << "       row " << i + 1 << ": M = " << simulation_sizes[i] << endl;
    }
    cout << endl;

    // Run the exact-simulation experiments across the requested grid.
    vector<MCResult> table =
        create_exact_simulation_table(
            initial_stock_price,
            strike_price,
            initial_variance,
            kappa,
            theta,
            sigma,
            rho,
            risk_free_rate,
            maturity,
            benchmark_price,
            seed,
            small_test,
            simulation_sizes
        );

    // Write the completed results table to CSV.
    write_exact_results_to_csv(table, output_path);

    return 0;
}