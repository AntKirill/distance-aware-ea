#include <algorithm>
#include <array>
#include <cstdint>
#include <climits>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include "coco.h"

//#define EPRINTF

#ifdef EPRINTF
#define eprintf(...) fprintf(stderr, __VA_ARGS__);
#define DEBUG(...) {__VA_ARGS__}
#else
#define eprintf(...)
#define DEBUG(...)
#endif

using namespace std;

struct RandomEngine {
    RandomEngine() { srand(time(nullptr)); }
    RandomEngine(int seed) { srand(seed); }
    int sampleSimBernoulli() const { return rand() % 2; }
    int sampleBernoulli(double p) const { return sample01Uniform() < p; }
    int sampleIntUniform(int fromIncl, int toExcl) const { return fromIncl + rand() % (toExcl - fromIncl); }
    double sample01Uniform() const { return ((double)rand() / (double)RAND_MAX); }
    int sampleBinomial(int n, double p) const {
        int success = 0;
        for (int i = 0; i < n; ++i) if (sample01Uniform() < p) ++success;
        return success;
    }
    int sampleDiscreteDistribution(const vector<double> &p) const {
        int n = p.size();
        vector<double> cdf(n + 1, 0.);
        for (int i = 1; i < cdf.size(); ++i) cdf[i] = cdf[i - 1] + p[i - 1];
        double r = sample01Uniform();
        return upper_bound(cdf.begin(), cdf.end(), r) - cdf.begin() - 1;
    }
    vector<int> samplePermutationUniform(int n) {
        vector<int> permutation(n, 1);
        for (int i = 1; i < n; ++i) {
            permutation[i] = i + 1;
            int pos = sampleIntUniform(0, i + 1);
            swap(permutation[pos], permutation[i]);
        }
        return permutation;
    }
    vector<int> sampleCombinationNKUniform(int n, int k) {
        vector<int> permutation = samplePermutationUniform(n);
        return vector<int>(permutation.begin(), permutation.begin() + k);
    }
} rnd;

vector<int> bernoulliScheme1Successes(int n, double p) {
    vector<int> successes;
    for (int i = 0; i < n; ++i) if (rnd.sampleBernoulli(p)) successes.push_back(i);
    if (successes.size() == 0) successes.push_back(rnd.sampleIntUniform(0, n));
    return successes;
}

struct TruncatedExponentialDistribution {
    double lambda0, lambda1;
    void build(double m, double eps) {
        double lb = -1. / m;
        double ub = -eps;
        auto leftPart = [](double lambda) { return exp(lambda) * (lambda - 1) / (exp(lambda) - 1) + 1 / (exp(lambda) - 1); };
        //eprintf("lb = %.5f, left(lb) = %.5f, right(lb) = %.5f\n", lb, leftPart(lb), m * lb);
        //eprintf("ub = %.5f, left(ub) = %.5f, right(ub) = %.5f\n", ub, leftPart(ub), m * ub);
        while (ub - lb >= eps) {
            double lambda = (lb + ub) / 2.;
            double left = exp(lambda) * (lambda - 1) / (exp(lambda) - 1) + 1 / (exp(lambda) - 1);
            double right = m * lambda;
            if (left >= right) lb = lambda;
            else ub = lambda;
        }
        lambda1 = lb;
        lambda0 = log(lambda1 / (exp(lambda1) - 1));
    }
    double pDF(double x) const { return exp(lambda0 + lambda1 * x); }
    double inverseCDF(double p) const { return (log(lambda1 * p + exp(lambda0)) - lambda0) / lambda1; }
};

using BinarySolution = vector<double>;
using RealSolution = vector<double>;
using FunctionRnToR = function<double(const RealSolution &)>;
using TerminationCriterion = function<bool (int iterationNumber, int spentBudget, double fitnessValue)>;
using Distance = function<double (const BinarySolution &, const BinarySolution &)>;
using Crossover = function<RealSolution (const RealSolution &, const RealSolution &)>;
using Mutation = function<RealSolution (const RealSolution &)>;

set<string> getAvailableBenchmarkProblemNames() {
    set<string> problems = {"OneMax", "Ruggedness"};
    for (int i = 1; i < 25; ++i) {
        string p = "BBOB-MIXINT-F" + to_string(i);
        problems.insert(p);
        p = "T-BBOB-MIXINT-F" + to_string(i);
        problems.insert(p);
    }
    return problems;
}

struct Config {
    const set<string> problems = getAvailableBenchmarkProblemNames();
    const set<string> algorithms = {"opl", "ddOpl", "op(l,l)", "ddOp(l,l)", "binUMDA", "intUMDA", "muLambdaGA", "ddMuLambdaGA", "rlsAB", "ddRlsAB", "onePlusLambdaAB"};
    const set<string> intMutations = {"uniform", "harmonic", "local"};
    string problem = "OneMax";
    string algorithm = "opl";
    string intMutation = "uniform";
    int N = 100, V = 2, lambda = 1, budget = 5e6, instance = 1, independentRuns = 1, t = 1;
    int mu = lambda;
    bool useDdCrossover = false, useComma = false;
    vector<int> cardinalities = {2, 4, 8, 16, 0};
    void printUsedConfigs() const {
        stringstream st;
        st << "Problem: " << problem << "\n";
        st << "Instance: " << instance << " (applicable to BBOB-MIXINT)\n";
        st << "Cardinalities: [" << cardinalities[0];
        for (int i = 1; i < cardinalities.size(); ++i) st << "," << cardinalities[i];
        st << "] (applicable to BBOB-MIXINT and intUMDA)\n";
        st << "Problem size: " << N << "\n";
        st << "Ruggedness parameter: " << V << " (applicable to Ruggedness)" << "\n";
        st << "T-BBOB-MIXINT parameter: " << t << " (applicable to T-BBOB-MIXINT)" << "\n";
        st << "Algorithm: " << algorithm << "\n";
        st << "IntMutation: " << intMutation << " (applicable to muLambdaGA)" << "\n";
        st << "Lambda: " << lambda << "\n";
        st << "Mu: " << mu << " (applicable to UMDA and muLambdaGA)" << "\n";
        st << "useDdCrossover: " << useDdCrossover << " (applicable to ddOp(l,l) and muLambdaGA)\n";
        st << "useComma: " << useComma << " (applicable to muLambdaGA)\n";
        st << "Budget: " << budget << "\n";
        st << "Independent runs: " << independentRuns << "\n";
        cout << st.str() << "\n";
        cerr << st.str() << "\n";
    }
} config;

int getCompressedIndexByAbsoluteIndex(int index, int period) {
    int m = config.N / period;
    int r = config.N % period;
    int j = index / (m + 1);
    if (j > r) j = r + (index - r * (m + 1)) / m;
    return j;
}

int getCardinality(int component, const vector<int> compressedCardinalities) {
    int j = getCompressedIndexByAbsoluteIndex(component, compressedCardinalities.size());
    return compressedCardinalities[j];
}

struct SharedStateBetweenExperiments {
    vector<vector<int>> ps;
    vector<vector<int>> invPs;
    int experimentNumber = 0;
    bool needGenerateNewPermutation() { return experimentNumber % 5 == 0; }
} sharedStateBetweenExperiments;

int oneMax(const BinarySolution &a) {
    int cnt = 0;
    for (int i = 0; i < a.size(); ++i) if (a[i]) ++cnt;
    return cnt;
}

// N = 100, V = 5
// 95 -> 99
// 96 -> 99, 96 + 5 - 2 = 98
// 97 -> 98, 97 + 5 - 4 = 97
// 98 -> 97, 98 + 5 - 6 = 96
// 99 -> 96, 99 + 5 - 8 = 95
// N = 100, V = 3
// 0 -> 2
// 1 -> 1
// 2 -> 0
int ruggedOneMax(const BinarySolution &a) {
    int oneMaxFitness = oneMax(a);
    if (a.size() == oneMaxFitness) return oneMaxFitness;
    int chunk = a.size() / config.V;
    if (oneMaxFitness / config.V == chunk) {
        int v = a.size() % config.V;
        int rShift = chunk * config.V % v;
        int r = oneMaxFitness % v - rShift;
        return oneMaxFitness + v - 2 * r - 1;
    }
    int r = oneMaxFitness % config.V;
    return oneMaxFitness + config.V - 2 * r - 1;
}

int permuteBBOBMIXINTSearchSpace(const vector<vector<int>> &invPs, const RealSolution &x, coco_problem_t *problem, const int bestValue) {
    double value = 0;
    RealSolution y(x.size());
    for (int i = 0; i < y.size(); ++i) {
        int ind = getCompressedIndexByAbsoluteIndex(i, config.cardinalities.size());
        y[i] = invPs[ind][(int)x[i]];
    }
    coco_evaluate_function(problem, y.data(), &value);
    return -bestValue - value;
}

vector<int> generatePermutation(int n, int t) {
    if (config.cardinalities.size() == 1 and config.cardinalities[0] == 100 and config.N == 40 and config.t == 5)
        return {1, 38, 43, 71, 92, 14, 21, 46, 61, 91, 11, 37, 51, 65, 87, 12, 25, 47, 76, 94, 7, 23, 40, 68, 95, 5, 32, 44, 79, 99, 16, 26, 50, 66, 97, 13, 30, 49, 63, 90, 18, 27, 52, 67, 98, 2, 33, 41, 72, 81, 10, 24, 55, 60, 85, 4, 31, 45, 73, 86, 8, 29, 53, 77, 80, 0, 20, 42, 70, 84, 3, 39, 48, 64, 88, 15, 36, 58, 75, 96, 17, 34, 59, 78, 82, 6, 35, 56, 62, 93, 19, 22, 57, 69, 89, 9, 28, 54, 74, 83};

    vector<int> p(n, 0);
    int value = 0;
    for (int r = 0; r < t; ++r) {
        int cnt = 0;
        for (int i = r; i < n; i += t) {
            p[i] = value++;
            int prvCnt = rnd.sampleIntUniform(0, cnt + 1);
            int prvI = i - t * prvCnt;
            swap(p[i], p[prvI]);
            ++cnt;
        }
    }
    return p;
}

vector<vector<int>> generatePermutations(const vector<int> &cardinalities, int t) {
    vector<vector<int>> permutations;
    for (int i = 0; i < cardinalities.size(); ++i) {
        auto p = generatePermutation(cardinalities[i], t);
        permutations.push_back(p);
    }
    return permutations;
}

vector<int> inversePermutation(const vector<int> &permutation) {
    vector<int> inverse(permutation.size());
    for (int i = 0; i < permutation.size(); ++i) inverse[permutation[i]] = i;
    return inverse;
}

vector<vector<int>> inversePermutations(vector<vector<int>> permutations) {
    for (auto &p : permutations) p = inversePermutation(p);
    return permutations;
}

struct Logger {
    const bool isLogEverything;
    double prvFitness = -DBL_MAX;
    Logger(bool _isLogEverything) : isLogEverything(_isLogEverything) {}
    template <typename ... Args> void logCSV(Args ... args) {
        if (isLogEverything) _logCSV(args...);
        else _logCSVFitness(args...);
    }
    template <typename ... Args> void logAdditional(Args ... args) const { _logCSV(args...); }
private:
    void _logCSV(double v) const { printf("%.5f\n", v); }
    void _logCSV(int v) const { printf("%d\n", v); }
    void _logCSV(const string &s) const { printf("%s\n", s.c_str()); }
    template <typename... Args> void _logCSV(double v, Args ... args) const { printf("%.5f,", v); _logCSV(args...); }
    template <typename... Args> void _logCSV(int v, Args ... args) const { printf("%d,", v); _logCSV(args...); }
    template <typename... Args> void _logCSV(const string &s, Args ... args) const { printf("%s,", s.c_str()); _logCSV(args...); }
    template <typename... Args> void _logCSVFitness(const string &s, Args ... args) const { printf("%s,", s.c_str()); _logCSV(args...); }
    template <typename... Args> void _logCSVFitness(double fitness, Args ... args) {
        if (prvFitness < fitness) {
            _logCSV(fitness, args...);
            prvFitness = fitness;
        }
    }
};

static inline void flip(double &b) {
    if (b < 0.5) b = 1;
    else b = 0;
}

void binaryUMDA(int mu, int lambda, BinarySolution &parent, FunctionRnToR fitnessFunction, TerminationCriterion isTerminate, Logger *loggerPtr = nullptr) {
    int spentBudget = 0;
    int n = parent.size();
    vector<double> p(n, 0.5);
    double lb = 1. / n;
    double ub = 1. - lb;
    double bestFitness = fitnessFunction(parent);
    ++spentBudget;
    if (loggerPtr) loggerPtr->logCSV("BestSoFarFitness", "Iteration", "SpentBudget");
    for (int iteration = 0; ; ++iteration) {
        if (loggerPtr) loggerPtr->logCSV(bestFitness, iteration, spentBudget);
        if (isTerminate(iteration, spentBudget, bestFitness)) break;
        vector<pair<double, BinarySolution>> pop;
        for (int i = 0 ; i < lambda; ++i) {
            BinarySolution x(n, 0);
            for (int j = 0; j < n; ++j) x[j] = rnd.sampleBernoulli(p[j]);
            double fitnessValue = fitnessFunction(x);
            ++spentBudget;
            pop.emplace_back(fitnessValue, x);
        }
        sort(pop.rbegin(), pop.rend());
        if (pop[0].first > bestFitness) {
            bestFitness = pop[0].first;
            parent = pop[0].second;
        }
        for (int i = 0; i < n; ++i) {
            int cnt1 = 0;
            for (int j = 0; j < mu; ++j) cnt1 += (pop[j].second[i] > 0.5);
            p[i] = (double)cnt1 / mu;
            p[i] = max(p[i], lb);
            p[i] = min(p[i], ub);
        }
    }
}

void integerUMDA(int mu, int lambda, const vector<int> &cardinalities, RealSolution &parent, const FunctionRnToR &fitnessFunction, const TerminationCriterion &isTerminate, Logger *loggerPtr = nullptr) {
    int spentBudget = 0;
    int n = parent.size();
    vector<vector<double>> p(n, vector<double>());
    for (int i = 0; i < n; ++i) p[i].resize(getCardinality(i, cardinalities), 1. / getCardinality(i, cardinalities));
    vector<double> lbs(n, 0.);
    vector<double> ubs(n, 0.);
    for (int i = 0; i < n; ++i) {
        lbs[i] = 1. / (getCardinality(i, cardinalities) - 1) / n;
        ubs[i] = 1. - lbs[i];
    }
    //double bestFitness = fitnessFunction(parent);
    double bestFitness = -DBL_MAX;
    ++spentBudget;
    if (loggerPtr) loggerPtr->logCSV("BestSoFarFitness", "Iteration", "SpentBudget");
    for (int iteration = 0; ; ++iteration) {
        if (loggerPtr) loggerPtr->logCSV(bestFitness, iteration, spentBudget);
        if (isTerminate(iteration, spentBudget, bestFitness)) break;
        vector<pair<double, BinarySolution>> pop;
        for (int i = 0 ; i < lambda; ++i) {
            RealSolution x(n, 0);
            for (int j = 0; j < n; ++j) x[j] = rnd.sampleDiscreteDistribution(p[j]);
            double fitnessValue = fitnessFunction(x);
            ++spentBudget;
            pop.emplace_back(fitnessValue, x);
        }
        sort(pop.rbegin(), pop.rend());
        if (pop[0].first > bestFitness) {
            bestFitness = pop[0].first;
            parent = pop[0].second;
        }
        for (int i = 0; i < n; ++i) {
            vector<int> cnt(getCardinality(i, cardinalities), 0);
            for (int j = 0; j < mu; ++j) cnt[pop[j].second[i]]++;
            for (int j = 0; j < getCardinality(i, cardinalities); ++j) {
                p[i][j] = (double)cnt[j] / mu;
                p[i][j] = max(p[i][j], lbs[i]);
                p[i][j] = min(p[i][j], ubs[i]);
            }
        }
    }
}

void onePlusLambdaEA(int lambda, BinarySolution &parent, FunctionRnToR fitnessFunction, TerminationCriterion isTerminate, Logger *loggerPtr = nullptr) {
    int spentBudget = 0;
    double parentFitness = fitnessFunction(parent);
    ++spentBudget;
    double p = 1. / parent.size();
    if (loggerPtr) loggerPtr->logCSV("BestSoFarFitness", "Iteration", "SpentBudget");
    for (int i = 0;; ++i) {
        if (loggerPtr) loggerPtr->logCSV(parentFitness, i, spentBudget);
        if (isTerminate(i, spentBudget, parentFitness)) break;
        BinarySolution bestIndividual;
        double bestIndividualFitness = -DBL_MAX;
        for (int k = 0; k < lambda; ++k) {
            auto individual = parent;
            int cntFlipped = 0;
            for (int j = 0; j < individual.size(); ++j) {
                double r = rnd.sample01Uniform();
                if (r < p) flip(individual[j]), ++cntFlipped;
            }
            if (cntFlipped == 0) {
                int pos = rnd.sampleIntUniform(0, individual.size());
                flip(individual[pos]);
            }
            double indFitness = fitnessFunction(individual);
            ++spentBudget;
            if (bestIndividualFitness < indFitness) {
                bestIndividualFitness = indFitness;
                swap(individual, bestIndividual);
            }
        }
        if (bestIndividualFitness >= parentFitness) swap(parent, bestIndividual), parentFitness = bestIndividualFitness;
    }
}

RealSolution generateDistantSolution(const RealSolution &x, const vector<int> cardinalities, const Distance &d, double distanceValue, const int budget, const int mu, const int lambda, const Logger *loggerPtr = nullptr) {
    //const int budget = 100000, mu = 500, lambda = 1000;
    auto fitnessFunction = [x, &d, distanceValue](const BinarySolution & candidate) { return -abs(d(candidate, x) - distanceValue); };
    auto isTerminate = [budget](int iterationNumber, int spentBudget, double fitness) { return spentBudget >= budget or fitness == 0; };
    RealSolution y = x;
    integerUMDA(mu, lambda, cardinalities, y, fitnessFunction, isTerminate);
    eprintf("d(x, y) = %.5f, expected %.5f\n", d(x, y), distanceValue);
    if (loggerPtr) loggerPtr->logAdditional("ObtainedMutDist", d(x, y), "TargetMutDist", distanceValue);
    return y;
}

BinarySolution ddMutation(const BinarySolution &initial, Distance d, int distanceValue) {
    auto fitnessFunction = [initial, d, distanceValue](const BinarySolution & candidate) { return -abs(d(candidate, initial) - distanceValue); };
    auto isTerminate = [](int iterationNumber, int spentBudget, double fitness) { return iterationNumber == 1000 or spentBudget == 1000 or fitness == 0; };
    auto parent = initial;
    //onePlusLambdaEA(1, parent, fitnessFunction, isTerminate);
    const vector<int> cardinalities(parent.size(), 2);
    integerUMDA(50, 100, cardinalities, parent, fitnessFunction, isTerminate);
    eprintf("d(parent, s) = %.3f, expected %d\n", d(initial, parent), distanceValue);
    return parent;
}

RealSolution ddMutation1(const RealSolution &x, const Distance &d, const TruncatedExponentialDistribution &distribution, const vector<int> &cardinalities, const int budget, const int mu, const int lambda, const Logger *loggerPtr = nullptr) {
    double uniform01 = rnd.sample01Uniform();
    double targetDistance = distribution.inverseCDF(uniform01);
    return generateDistantSolution(x, cardinalities, d, targetDistance, budget, mu, lambda, loggerPtr);
}

RealSolution ddCrossover(const RealSolution &parent1, const RealSolution &parent2, double c, Distance d, const vector<int> &cardinalities) {
    double dParents = d(parent1, parent2);
    double d1 = rnd.sampleBinomial(dParents, c);
    double d2 = dParents - d1;
    auto fitnessFunction = [parent1, parent2, d1, d2, d](const RealSolution & s) { return -abs(d(parent1, s) - d1) - abs(d(parent2, s) - d2); };
    auto isTerminate = [](int iterationNumber, int spentBudget, double fitness) { return spentBudget >= 1000 or fitness == 0.; };
    auto child = parent1;
    //onePlusLambdaEA(1, child, fitnessFunction, isTerminate);
    integerUMDA(50, 100, cardinalities, child, fitnessFunction, isTerminate);
    eprintf("d(p1, s) = %.3f, expected %.3f; d(p2, s) = %.3f, expected %.3f; d(p1, p2) = %.3f\n", d(parent1, child), d1, d(parent2, child), d2, d(parent1, parent2));
    return child;
}


void ddOnePlusLambdaEA(int lambda, BinarySolution &parent, FunctionRnToR fitnessFunction, TerminationCriterion isTerminate, Distance d, Logger *loggerPtr = nullptr) {
    int spentBudget = 0;
    double parentFitness = fitnessFunction(parent);
    ++spentBudget;
    double p = 1. / parent.size();
    if (loggerPtr) loggerPtr->logCSV("BestSoFarFitness", "Iteration", "SpentBudget");
    for (int i = 0;; ++i) {
        if (loggerPtr) loggerPtr->logCSV(parentFitness, i, spentBudget);
        if (isTerminate(i, spentBudget, parentFitness)) break;
        BinarySolution bestIndividual;
        double bestIndividualFitness = -DBL_MAX;
        for (int k = 0; k < lambda; ++k) {
            int distanceValue = rnd.sampleBinomial(parent.size(), p);
            if (distanceValue == 0) ++distanceValue;
            auto individual = ddMutation(parent, d, distanceValue);
            double indFitness = fitnessFunction(individual);
            ++spentBudget;
            if (bestIndividualFitness < indFitness) {
                bestIndividualFitness = indFitness;
                swap(individual, bestIndividual);
            }
        }
        if (bestIndividualFitness >= parentFitness) swap(parent, bestIndividual), parentFitness = bestIndividualFitness;
    }
}

void onePlusLambdaLambdaEA(int lambda, BinarySolution &parent, FunctionRnToR fitnessFunction, TerminationCriterion isTerminate, Logger *loggerPtr = nullptr) {
    int spentBudget = 0;
    double parentFitness = fitnessFunction(parent);
    ++spentBudget;
    int n = parent.size();
    double p = (double) lambda / n;
    double c = 1. / lambda;
    if (loggerPtr) loggerPtr->logCSV("BestSoFarFitness", "Iteration", "SpentBudget");
    for (int i = 0; ; ++i) {
        if (loggerPtr) loggerPtr->logCSV(parentFitness, i, spentBudget);
        if (isTerminate(i, spentBudget, parentFitness)) break;
        vector<pair<double, BinarySolution>> population;
        for (int k = 0; k < lambda; ++k) {
            auto individual = parent;
            int cntFlipped = 0;
            for (int j = 0; j < individual.size(); ++j) {
                double r = rnd.sample01Uniform();
                if (r < p) flip(individual[j]), ++cntFlipped;
            }
            if (cntFlipped == 0) {
                int pos = rnd.sampleIntUniform(0, individual.size());
                flip(individual[pos]);
            }
            double indFitness = fitnessFunction(individual);
            ++spentBudget;
            population.emplace_back(indFitness, individual);
        }
        sort(population.begin(), population.end(), [](const auto & a, const auto & b) {return a.first > b.first;});
        int cntBestInPop = 0;
        double bestFitnessInPop = population.begin()->first;
        while (cntBestInPop < n and bestFitnessInPop == population[cntBestInPop].first) ++cntBestInPop;
        int pos = rnd.sampleIntUniform(0, cntBestInPop);
        double xPrimeFitness = population[pos].first;
        auto xPrime = population[pos].second;
        for (int k = 0; k < lambda; ++k) {
            population[k] = {parentFitness, parent};
            for (int j = 0; j < n; ++j) {
                double r = rnd.sample01Uniform();
                if (r < c) population[k].second[j] = xPrime[j];
            }
            population[k].first = fitnessFunction(population[k].second);
            ++spentBudget;
        }
        sort(population.begin(), population.end(), [](const auto & a, const auto & b) {return a.first > b.first;});
        cntBestInPop = 0;
        bestFitnessInPop = population.begin()->first;
        while (cntBestInPop < n and bestFitnessInPop == population[cntBestInPop].first) ++cntBestInPop;
        pos = rnd.sampleIntUniform(0, cntBestInPop);
        if (population[pos].first >= parentFitness) swap(parent, population[pos].second), parentFitness = population[pos].first;
    }
}

void ddOnePlusLambdaLambdaEA(int lambda, BinarySolution &parent, FunctionRnToR fitnessFunction, TerminationCriterion isTerminate, Distance d, Logger *loggerPtr = nullptr, bool useDdCrossover = false) {
    int spentBudget = 0;
    double parentFitness = fitnessFunction(parent);
    ++spentBudget;
    int n = parent.size();
    double p = (double) lambda / n;
    //double p = 1./n;
    double c = 1. / lambda;
    const vector<int> cardinalities(n, 2);
    if (loggerPtr) loggerPtr->logCSV("BestSoFarFitness", "Iteration", "SpentBudget");
    for (int i = 0; ; ++i) {
        if (loggerPtr) loggerPtr->logCSV(parentFitness, i, spentBudget);
        if (isTerminate(i, spentBudget, parentFitness)) break;
        vector<pair<double, BinarySolution>> population;
        for (int k = 0; k < lambda; ++k) {
            int cntFlipped = rnd.sampleBinomial(parent.size(), p);
            if (cntFlipped == 0) ++cntFlipped;
            eprintf("Run Mutation %d: ", k);
            auto individual = ddMutation(parent, d, cntFlipped);
            double indFitness = fitnessFunction(individual);
            ++spentBudget;
            population.emplace_back(indFitness, individual);
        }
        sort(population.rbegin(), population.rend());
        int cntBestInPop = 0;
        double bestFitnessInPop = population.begin()->first;
        while (cntBestInPop < n and bestFitnessInPop == population[cntBestInPop].first) ++cntBestInPop;
        int pos = rnd.sampleIntUniform(0, cntBestInPop);
        double xPrimeFitness = population[pos].first;
        auto xPrime = population[pos].second;
        if (!useDdCrossover) {
            for (int k = 0; k < lambda; ++k) {
                population[k] = {parentFitness, parent};
                for (int j = 0; j < n; ++j) {
                    double r = rnd.sample01Uniform();
                    if (r < c) population[k].second[j] = xPrime[j];
                }
                population[k].first = fitnessFunction(population[k].second);
                ++spentBudget;
            }
        } else {
            for (int k = 0; k < lambda; ++k) {
                eprintf("Run Crossover %d: ", k);
                population[k] = {parentFitness, parent};
                population[k].second = ddCrossover(population[k].second, xPrime, c, d, cardinalities);
                population[k].first = fitnessFunction(population[k].second);
                ++spentBudget;
            }
        }
        sort(population.rbegin(), population.rend());
        cntBestInPop = 0;
        bestFitnessInPop = population.begin()->first;
        while (cntBestInPop < n and bestFitnessInPop == population[cntBestInPop].first) ++cntBestInPop;
        pos = rnd.sampleIntUniform(0, cntBestInPop);
        if (population[pos].first >= parentFitness) swap(parent, population[pos].second), parentFitness = population[pos].first;
    }
}

void muLambdaGA(int mu, int lambda, bool isElitist, const vector<int> &cardinalities, RealSolution &bestSolution, FunctionRnToR fitnessFunction, TerminationCriterion isTerminate, Crossover crossover, Mutation mutation, Logger *loggerPtr = nullptr) {
    int n = bestSolution.size();
    vector<pair<double, RealSolution>> population;
    int spentBudget = 0;
    double bestFitness = -DBL_MAX;
    if (loggerPtr) loggerPtr->logCSV("BestSoFarFitness", "Iteration", "SpentBudget");
    for (int i = 0; i < mu; ++i) {
        RealSolution individual(n);
        for (int j = 0; j < n; ++j) individual[j] = rnd.sampleIntUniform(0, getCardinality(j, cardinalities));
        double fValue = fitnessFunction(individual);
        ++spentBudget;
        population.emplace_back(fValue, individual);
        if (fValue > bestFitness) bestFitness = fValue, bestSolution = individual;
    }
    for (int iteration = 0; ; ++iteration) {
        if (loggerPtr) loggerPtr->logCSV(bestFitness, iteration, spentBudget);
        if (isTerminate(iteration, spentBudget, bestFitness)) break;
        vector<pair<double, RealSolution>> newPopulation;
        if (isElitist) newPopulation = population;
        for (int i = 0; i < lambda; ++i) {
            const auto &p1 = population[rnd.sampleIntUniform(0, population.size())];
            const auto &p2 = population[rnd.sampleIntUniform(0, population.size())];
            auto child = crossover(p1.second, p2.second);
            auto mutant = mutation(child);
            double fValue = fitnessFunction(mutant);
            ++spentBudget;
            if (fValue > bestFitness) bestFitness = fValue, bestSolution = mutant;
            newPopulation.emplace_back(fValue, mutant);
        }
        sort(newPopulation.rbegin(), newPopulation.rend());
        population = vector<pair<double, RealSolution>>(newPopulation.begin(), newPopulation.begin() + mu);
    }
}

void rlsAB(double a, double b, const vector<int> &cardinalities, RealSolution &x, FunctionRnToR fitnessFunction, TerminationCriterion isTerminate, Logger *loggerPtr = nullptr) {
    double bestFitness = fitnessFunction(x);
    int spentBudget = 0;
    int n = x.size();
    vector<double> v(n);
    double r = cardinalities[0];
    generate(v.begin(), v.end(), [r]() {return rnd.sample01Uniform() * (floor(r / 4) - 1) + 1;});
    if (loggerPtr) loggerPtr->logCSV("BestSoFarFitness", "Iteration", "SpentBudget", "MeanDist");
    for (int iter = 0; ; ++iter) {
        if (loggerPtr) {
            double s = 0;
            for (int vi : v) s += vi;
            s /= n;
            loggerPtr->logCSV(bestFitness, iter, spentBudget, s);
        }
        if (isTerminate(iter, spentBudget, bestFitness)) break;
        int ind = rnd.sampleIntUniform(0, n);
        int yInd = x[ind];
        if (rnd.sampleSimBernoulli()) yInd = x[ind] - floor(v[ind]);
        else yInd = x[ind] + floor(v[ind]);
        auto y = x;
        y[ind] = yInd;
        double fy = fitnessFunction(y);
        ++spentBudget;
        if (fy > bestFitness) {
            v[ind] = min(a * v[ind], floor(r / 4));
            bestFitness = fy;
            x = y;
        } else v[ind] = max(1., b * v[ind]);
    }
}

void onePlusLambdaAB(double a, double b, const vector<int> &cardinalities, RealSolution &x, FunctionRnToR fitnessFunction, TerminationCriterion isTerminate, Logger *loggerPtr = nullptr) {
    double bestFitness = fitnessFunction(x);
    int spentBudget = 0;
    int n = x.size();
    vector<double> v(n);
    double r = cardinalities[0];
    generate(v.begin(), v.end(), [r]() {return rnd.sample01Uniform() * (floor(r / 4) - 1) + 1;});
    if (loggerPtr) loggerPtr->logCSV("BestSoFarFitness", "Iteration", "SpentBudget", "MeanDist");
    for (int iter = 0; ; ++iter) {
        if (loggerPtr) {
            double s = 0;
            for (int vi : v) s += vi;
            s /= n;
            loggerPtr->logCSV(bestFitness, iter, spentBudget, s);
        }
        if (isTerminate(iter, spentBudget, bestFitness)) break;
        vector<int> inds = bernoulliScheme1Successes(n, 1. / n);
        auto y = x;
        for (int ind : inds) {
            int yInd = x[ind];
            if (rnd.sampleSimBernoulli()) yInd = x[ind] - floor(v[ind]);
            else yInd = x[ind] + floor(v[ind]);
            y[ind] = yInd;
        }
        double fy = fitnessFunction(y);
        ++spentBudget;
        if (fy > bestFitness) {
            for (int ind: inds) v[ind] = min(a * v[ind], floor(r / 4));
            bestFitness = fy;
            x = y;
        } else for (int ind: inds) v[ind] = max(1., b * v[ind]);
    }
}

RealSolution sampleRnUniformly(int n, const vector<int> &cardinalities) {
    RealSolution x(n, 0);
    for (int i = 0; i < n; ++i) x[i] = rnd.sampleIntUniform(0, getCardinality(i, cardinalities));
    return x;
}

double rbfBasedDistTransform(double gamma, double d) {
    const double c = gamma * d;
    return 1. - exp(-c * c);
}

double findSmallestDist(const RealSolution &x, const vector<int> &cardinalities, const Distance &d, int budget, const double threshold = 0.) {
    const double t = 2.;
    RealSolution z = sampleRnUniformly(x.size(), cardinalities);
    RealSolution z1 = z;
    int spentBudget = 0;
    double dist = d(x, z);
    while (dist > threshold and spentBudget < budget) {
        z = z1;
        double targetDist = dist / t;
        auto fitnessFunction = [x, d, targetDist](const BinarySolution & candidate) { return targetDist - d(candidate, x); };
        auto isTerminate = [targetDist](int iterationNumber, int spentBudget, double fitness) { return spentBudget >= 500000 or (fitness >= 0 and fitness < targetDist); };
        z1 = z;
        integerUMDA(500, 10000, cardinalities, z1, fitnessFunction, isTerminate);
        eprintf("d(parent, s) = %.3f, expected %.3f\n", d(x, z1), targetDist);
        double distZ1 = d(x, z1);
        dist = min(dist, distZ1);
        ++spentBudget;
    }
    DEBUG(
        for (int i = 0; i < x.size(); ++i) printf("%.1f,", x[i]);
        printf("\n");
        for (int i = 0; i < x.size(); ++i) printf("%.1f,", z[i]);
            printf("\n");
        );
    return d(x, z);
}

double findLargestDist(const RealSolution &x, const vector<int> &cardinalities, const Distance &d, int budget, const double threshold = 1e5) {
    const double t = 1.2;
    RealSolution z = sampleRnUniformly(x.size(), cardinalities);
    RealSolution z1 = z;
    int spentBudget = 0;
    double dist = d(x, z);
    while (dist < threshold and spentBudget < budget) {
        z = z1;
        double targetDist = dist * t;
        auto fitnessFunction = [x, d, targetDist](const BinarySolution & candidate) { return d(candidate, x) - targetDist; };
        auto isTerminate = [targetDist](int iterationNumber, int spentBudget, double fitness) { return spentBudget >= 500000 or fitness >= 0; };
        z1 = z;
        integerUMDA(500, 10000, cardinalities, z1, fitnessFunction, isTerminate);
        eprintf("d(parent, s) = %.3f, expected %.3f\n", d(x, z1), targetDist);
        double distZ1 = d(x, z1);
        dist = max(dist, distZ1);
        ++spentBudget;
    }
    DEBUG(
        for (int i = 0; i < x.size(); ++i) printf("%.1f,", x[i]);
        printf("\n");
        for (int i = 0; i < x.size(); ++i) printf("%.1f,", z[i]);
            printf("\n");
        );
    return d(x, z);
}

double findDistGamma(double dMin, double dMax, const Logger *logger = nullptr) {
    const double t = 1.1;
    const double delta = 1e-4;
    const double dd = pow(dMax / dMin, 2.);
    double eps1 = delta;
    double eps2 = 0.;
    double dEpsMin = DBL_MAX;
    while (eps1 <= 0.01) {
        double eps2LB = pow((1. - eps1), dd);
        double dEps = eps2LB - eps1;
        if (dEps < dEpsMin) {
            dEpsMin = dEps;
            eps2 = (eps1 + eps2LB) / 2.;
        }
        if (dEpsMin <= 0.) break;
        //eps1 *= t;
        eps1 += delta;
    }
    double lbGamma = sqrt(-log(eps2)) / dMax;
    double ubGamma = sqrt(-log(1. - eps1)) / dMin;
    eprintf("eps1 = %.5f, eps2 = %.5f\n", eps1, eps2);
    logger->logAdditional("eps1", eps1, "eps2", eps2);
    return (lbGamma + ubGamma) / 2.;
}

double exploreSearchSpace(const Distance &distance) {
    if (config.cardinalities.size() == 1 and config.cardinalities[0] == 2000 and config.N == 40) return 0.00611;
    if (config.cardinalities.size() == 1 and config.cardinalities[0] == 100 and config.N == 40) return 0.00611;
    RealSolution x = sampleRnUniformly(config.N, config.cardinalities);
    double dSmallest = findSmallestDist(x, config.cardinalities, distance, 10);
    double dLargest = findLargestDist(x, config.cardinalities, distance, 10);
    //double dSmallest = 15, dLargest = 9700;
    double gamma = findDistGamma(dSmallest, dLargest);
    return gamma;
}


void ddRlsAB(double a, double b, const vector<int> &cardinalities, RealSolution &x, FunctionRnToR fitnessFunction, TerminationCriterion isTerminate, const Distance &d, Logger *loggerPtr = nullptr) {
    double dSmallest = findSmallestDist(x, cardinalities, d, 10);
    double dLargest = findLargestDist(x, cardinalities, d, 10);
    double gamma = findDistGamma(dSmallest, dLargest, loggerPtr);
    if (loggerPtr) loggerPtr->logAdditional("dSmallest", dSmallest, "dLargest", dLargest, "gamma", gamma);
    eprintf("gamma = %.5f\n", gamma);
    auto transformedDist = [gamma, &d](const RealSolution & x, const RealSolution & y) { return rbfBasedDistTransform(gamma, d(x, y));};
    int spentBudget = 0;
    double bestFitness = fitnessFunction(x);
    ++spentBudget;
    int n = x.size();
    double r = cardinalities[0];
    TruncatedExponentialDistribution distribution;
    double m = 0.02;
    distribution.build(m, 0.00001);
    if (loggerPtr) loggerPtr->logCSV("BestSoFarFitness", "Iteration", "SpentBudget", "MeanDist", "NewM", "lambda0", "lambda1");
    double c1 = 1.001, c2 = 0.999;
    for (int iter = 0; ; ++iter) {
        auto y = ddMutation1(x, transformedDist, distribution, cardinalities, 40000, 100, 1000, loggerPtr);
        double fy = fitnessFunction(y);
        ++spentBudget;
        double meanDist = sqrt(-log(1 - m)) / gamma;
        double d1;
        //if (fy > bestFitness) d1 = meanDist + (a * a - 1.);
        //else d1 = meanDist - (1. - b * b);
        //if (fy >= bestFitness) bestFitness = fy, x = y;
        //m = rbfBasedDistTransform(gamma, d1);
        if (fy > bestFitness) m *= c1;
        else m *= c2;
        if (fy >= bestFitness) bestFitness = fy, x = y;
        m = max(0.0003, m);
        m = min(0.4, m);
        distribution.build(m, 0.00001);
        eprintf("%d: fBest = %.5f, meanDist = %.5f, newMeanDist = %.5f, m = %.5f\n", spentBudget, bestFitness, meanDist, d1, m);
        if (loggerPtr) loggerPtr->logCSV(bestFitness, iter, spentBudget, meanDist, m, distribution.lambda0, distribution.lambda1);
        if (isTerminate(iter, spentBudget, bestFitness)) break;
    }
}

double fitnessBasedDistance(FunctionRnToR f, const BinarySolution &a, const BinarySolution &b) {
    return abs(f(a) - f(b));
}

RealSolution intUniformCrossover(const RealSolution &x, const RealSolution &y) {
    auto z = x;
    for (int i = 0 ; i < x.size(); ++i) if (rnd.sampleSimBernoulli()) z[i] = y[i];
    return z;
}

RealSolution intUniformMutation(const vector<int> &cardinalities, const RealSolution &x) {
    double p = 1. / x.size();
    vector<int> indexes = bernoulliScheme1Successes(x.size(), p);
    auto y = x;
    for (int i : indexes) {
        int newValue = rnd.sampleIntUniform(0, getCardinality(i, cardinalities) - 1);
        newValue = newValue >= y[i] ? newValue + 1 : newValue;
        y[i] = newValue;
    }
    return y;
}

RealSolution intHarmonicMutation(const vector<vector<double>> &harmonicDistribution, const vector<int> &cardinalities, const RealSolution &x) {
    double p = 1. / x.size();
    vector<int> indexes = bernoulliScheme1Successes(x.size(), p);
    auto y = x;
    for (int i : indexes) {
        int component = getCompressedIndexByAbsoluteIndex(i, harmonicDistribution.size());
        int stepSize = rnd.sampleDiscreteDistribution(harmonicDistribution[component]);
        int direction = rnd.sampleSimBernoulli() ? 1 : -1;
        int newValue = y[i] + direction * stepSize;
        int c = cardinalities[component];
        y[i] = (c + newValue % c) % c;
    }
    return y;
}

RealSolution intLocalMutation(const vector<int> &cardinalities, const RealSolution &x) {
    vector<int> indexes = bernoulliScheme1Successes(x.size(), 1. / x.size());
    auto y = x;
    for (int i : indexes) {
        int component = getCompressedIndexByAbsoluteIndex(i, cardinalities.size());
        int c = cardinalities[component];
        int step = rnd.sampleSimBernoulli() ? 1 : -1;
        int newValue = y[i] + step;
        y[i] = (c + newValue % c) % c;
    }
    return y;
}

vector<vector<double>> precomputeHarmonicDistribution(const vector<int> &cardinalities) {
    vector<vector<double>> d;
    for (int j = 0; j < cardinalities.size(); ++j) {
        int n = getCardinality(j, cardinalities) - 1;
        double s = 0;
        for (int i = 1; i <= n; ++i) s += 1. / i;
        double c = 1. / s;
        vector<double> distribution(n, 0.);
        for (int i = 1; i <= n; ++i) distribution[i - 1] = c / i;
        d.push_back(distribution);
    }
    return d;
}



double l2Norm(const RealSolution &x, const RealSolution &y) {
    double l2Norm = 0.;
    for (int i = 0; i < x.size(); ++i) {
        double d = x[i] - y[i];
        l2Norm += d * d;
    }
    return sqrt(l2Norm);
}

double untangledL2Norm(const vector<vector<int>> &invPerms, const RealSolution &x, const RealSolution &y) {
    double l2Norm = 0.;
    for (int i = 0; i < y.size(); ++i) {
        int xi = round(x[i]);
        int yi = round(y[i]);
        int ind = getCompressedIndexByAbsoluteIndex(i, invPerms.size());
        int x1 = invPerms[ind][xi];
        int y1 = invPerms[ind][yi];
        double d = x1 - y1;
        l2Norm += d * d;
    }
    double ans = sqrt((double)l2Norm);
    return ans;
}

void doOneExperiment() {
    BinarySolution initSolution = vector<double>(config.N);
    generate(initSolution.begin(), initSolution.end(), []() {return rnd.sampleSimBernoulli();});
    TerminationCriterion isTerminate = [](int iteration, int spentBudget, double fitness) {return iteration >= config.budget or spentBudget >= config.budget or fitness == config.N;};
    Logger logger(true);
    FunctionRnToR fitness;
    Distance dist;
    if (config.problem == "OneMax") {
        fitness = oneMax;
    } else if (config.problem == "Ruggedness") {
        fitness = ruggedOneMax;
        dist = [](const BinarySolution & a, const BinarySolution & b) { return fitnessBasedDistance(ruggedOneMax, a, b); };
    } else if (config.problem.find("BBOB-MIXINT") != string::npos) {
        generate(initSolution.begin(), initSolution.end(), []() {return rnd.sampleIntUniform(0, config.cardinalities[0]);});
        int p = 0, num = 0;
        while (p < config.problem.size() and !isdigit(config.problem[p])) ++p;
        while (p < config.problem.size() and isdigit(config.problem[p])) num = num * 10 + config.problem[p] - '0', ++p;
        coco_problem_t *problem = naco_get_problem_bbob_mixint(num, config.N, config.instance, config.cardinalities.size(), config.cardinalities.data());
        double bestValue = -naco_problem_get_best_value(problem);
        if (config.problem.find("T-") == 0) {
            if (sharedStateBetweenExperiments.needGenerateNewPermutation()) {
                sharedStateBetweenExperiments.ps = generatePermutations(config.cardinalities, config.t);
                eprintf("\n");
                for (int i = 0; i < sharedStateBetweenExperiments.ps[0].size(); ++i) {
                    eprintf("%d,", sharedStateBetweenExperiments.ps[0][i]);
                }
                eprintf("\n");
                sharedStateBetweenExperiments.invPs = inversePermutations(sharedStateBetweenExperiments.ps);
            }
            dist = [](const RealSolution & y1, const RealSolution & y2) { return untangledL2Norm(sharedStateBetweenExperiments.invPs, y1, y2); };
            fitness = [bestValue, problem](const RealSolution & x) {
                return permuteBBOBMIXINTSearchSpace(sharedStateBetweenExperiments.invPs, x, problem, bestValue);
            };
        } else {
            fitness = [bestValue, problem](const RealSolution & a) {
                double value = 0;
                coco_evaluate_function(problem, a.data(), &value);
                return -bestValue - value;
            };
        }
        isTerminate = [](int iteration, int spentBudget, double fitness) {return iteration >= config.budget or spentBudget >= config.budget or fitness == 0.;};
        //printf("%s-i%d-d%d best possible value: %.7f\n", config.problem.c_str(), config.instance, config.N, bestValue);
    }
    const set<string> binaryAlgorithms = {"opl", "ddOpl", "op(l,l)"};
    if (binaryAlgorithms.count(config.algorithm) and config.problem.find("BBOB") == 0) {
        for (int i = 0; i < config.cardinalities.size(); ++i) if(config.cardinalities[i] > 2) {
                cerr << "Algorithm " << config.algorithm << " is binary, but the cardinality " << i + 1 << " is " << config.cardinalities[i] << "\n";
                return;
            }
    }
    //Crossover crossover = [](const RealSolution & x, const RealSolution & y) { return intUniformCrossover(x, y); };
    Crossover crossover = [](const RealSolution & x, const RealSolution & y) { return x; };
    Mutation mutation;
    if (config.intMutation == "uniform") mutation = [](const RealSolution & x) { return intUniformMutation(config.cardinalities, x);};
    else if (config.intMutation == "local") mutation = [](const RealSolution & x) { return intLocalMutation(config.cardinalities, x); };
    else if (config.intMutation == "harmonic") {
        auto distribution = precomputeHarmonicDistribution(config.cardinalities);
        mutation = [distribution](const RealSolution & x) { return intHarmonicMutation(distribution, config.cardinalities, x); };
    }
    if (config.algorithm == "opl") {
        onePlusLambdaEA(config.lambda, initSolution, fitness, isTerminate, &logger);
    } else if (config.algorithm == "ddOpl") {
        ddOnePlusLambdaEA(config.lambda, initSolution, fitness, isTerminate, dist, &logger);
    } else if (config.algorithm == "op(l,l)") {
        onePlusLambdaLambdaEA(config.lambda, initSolution, fitness, isTerminate, &logger);
    } else if (config.algorithm == "ddOp(l,l)") {
        ddOnePlusLambdaLambdaEA(config.lambda, initSolution, fitness, isTerminate, dist, &logger, config.useDdCrossover);
    } else if (config.algorithm == "binUMDA") {
        binaryUMDA(config.mu, config.lambda, initSolution, fitness, isTerminate, &logger);
    } else if (config.algorithm == "intUMDA") {
        integerUMDA(config.mu, config.lambda, config.cardinalities, initSolution, fitness, isTerminate, &logger);
    } else if (config.algorithm == "muLambdaGA") {
        muLambdaGA(config.mu, config.lambda, !config.useComma, config.cardinalities, initSolution, fitness, isTerminate, crossover, mutation, &logger);
    } else if (config.algorithm == "ddMuLambdaGA") {
        RealSolution x = sampleRnUniformly(config.N, config.cardinalities);
        auto tmpDist = [](const RealSolution & x, const RealSolution & y) { return untangledL2Norm(sharedStateBetweenExperiments.invPs, x, y); };
        double dSmallest = findSmallestDist(x, config.cardinalities, tmpDist, 10);
        double dLargest = findLargestDist(x, config.cardinalities, tmpDist, 10);
        //double dSmallest = 15, dLargest = 9700;
        double gamma = findDistGamma(dSmallest, dLargest);
        TruncatedExponentialDistribution distribution;
        distribution.build(0.1, 0.0001);
        eprintf("dSmallest = %.5f, dLargest = %.5f, gamma = %.5f, lambda0 = %.5f, lambda1 = %.5f\n", dSmallest, dLargest, gamma, distribution.lambda0, distribution.lambda1);
        dist = [gamma](const RealSolution & y1, const RealSolution & y2) { return rbfBasedDistTransform(gamma, untangledL2Norm(sharedStateBetweenExperiments.invPs, y1, y2)); };
        if (config.useDdCrossover) crossover = [dist](const RealSolution & x, const RealSolution & y) { return ddCrossover(x, y, 0.5, dist, config.cardinalities); };
        mutation = [dist, distribution](const RealSolution & x) { return ddMutation1(x, dist, distribution, config.cardinalities, 10000, 100, 1000); };
        muLambdaGA(config.mu, config.lambda, !config.useComma, config.cardinalities, initSolution, fitness, isTerminate, crossover, mutation, &logger);
    } else if (config.algorithm == "rlsAB") {
        rlsAB(1.7, 0.9, config.cardinalities, initSolution, fitness, isTerminate, &logger);
    } else if (config.algorithm == "ddRlsAB") {
        ddRlsAB(1.7, 0.9, config.cardinalities, initSolution, fitness, isTerminate, dist, &logger);
    } else if (config.algorithm == "onePlusLambdaAB") {
        onePlusLambdaAB(1.7, 0.9, config.cardinalities, initSolution, fitness, isTerminate, &logger);
    }
}

void experiment() {
    for (int i = 0; i < config.independentRuns; ++i) {
        cerr << "Run #" << i + 1 << " ...";
        sharedStateBetweenExperiments.experimentNumber = i;
        rnd = RandomEngine(i);
        doOneExperiment();
        cerr << "Done\n";
    }
}
