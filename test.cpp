#include <cassert>
#include <chrono>
#include <gsl/gsl_integration.h>
#include "impl.cpp"

int test() {
    BinarySolution initSolution = vector<double>(config.N);
    generate(initSolution.begin(), initSolution.end(), []() {return 1;});
    //initSolution[0] = 0, initSolution[1] = 0, initSolution[2] = 0, initSolution[3] = 0;
    initSolution[0] = 1;
    int f = ruggedOneMax(initSolution);
    printf("%d -> %d\n", oneMax(initSolution), ruggedOneMax(initSolution));
    return 0;
}

//auto create_bbob_problem_ioh(int fId, int instanceId, int dim) {
//const auto &problem_factory = ioh::problem::ProblemRegistry<ioh::problem::RealSingleObjective>::instance();
//return problem_factory.create(fId, instanceId, dim);
//}

//void test1() {
//auto problem_ptr = create_bbob_problem_ioh(1, 1, 2);
//cout << std::setprecision(16) << (*problem_ptr)({0., 0.}) << "\n";
//vector<double> x = problem_ptr->optimum().x;
//double y =  problem_ptr->optimum().y;
//cout << std::setprecision(16) << x[0] << " " << x[1] << "\n";
//cout << std::setprecision(16) << y << "\n";
//}

int test2() {
    //coco_problem_t *problem = coco_get_largescale_problem(1, 2, 1);
    coco_suite_t *suite = coco_suite("bbob", "", "");
    //size_t index =  coco_suite_encode_problem_index(suite, 1, 2, 1);
    //coco_problem_t *problem = coco_suite_get_problem(suite, index);
    coco_problem_t *problem = coco_suite_get_problem_by_function_dimension_instance(suite, 1, 2, 1);
    //double *x = coco_allocate_vector(2);
    //double *value = coco_allocate_vector(1);
    //x[0] = 0., x[1] = 0.;
    vector<double> x = {0., 0.};
    double value = 0;
    coco_evaluate_function(problem, x.data(), &value);
    printf("%.16f\n", value);
    //cout << std::setprecision(9) << problem->best_parameter[0] << " " << problem->best_parameter[1] << "\n";
    bbob_problem_best_parameter_print(problem);
    coco_suite_free(suite);
    return 0;
}

int test3() {
    vector<int> cards = {2};
    int N = 100;
    coco_problem_t *problem = naco_get_problem_bbob_mixint(1, N, 1, cards.size(), cards.data());
    vector<double> a(N, 0);
    double value = 0;
    coco_evaluate_function(problem, a.data(), &value);
    printf("%.16f\n", value);
    double *x = naco_problem_get_best_parameter(problem);
    double y = naco_problem_get_best_value(problem);
    for (int i = 0; i < N; ++i) printf("%.1f,", x[i]);
    printf("\n");
    printf("%.16f\n", y);
    coco_evaluate_function(problem, x, &value);
    printf("%.16f\n", value);
    return 0;
}

int test4() {
    int r = (-5) % 2;
    printf("%d\n", r);
    r = (-5) % 3;
    printf("%d\n", r);
    return 0;
}


int test5() {
    const int n = 50;
    const vector<int> cardinalities = {100};
    RealSolution x = sampleRnUniformly(n, cardinalities);
    double dSmallest = findSmallestDist(x, cardinalities, l2Norm, 10);
    double dLargest = findLargestDist(x, cardinalities, l2Norm, 10);
    printf("Smallest distance: %.5f, Largest distance: %.5f\n", dSmallest, dLargest);
    return 0;
}

int test6() {
    const int n = 50;
    const vector<int> cardinalities = {100};
    double dSmallest = 10., dLargest = 552.;
    double gamma = findDistGamma(dSmallest, dLargest);
    double dSmallestT = rbfBasedDistTransform(gamma, dSmallest);
    double dLargestT = rbfBasedDistTransform(gamma, dLargest);
    printf("Gamma = %.5f, SmallestDT = %.5f, LargestDT = %.5f\n", gamma, dSmallestT, dLargestT);
    for (int i = 0; i < 10; ++i) {
        RealSolution x = sampleRnUniformly(50, cardinalities);
        RealSolution y = sampleRnUniformly(50, cardinalities);
        double d = l2Norm(x, y);
        double dT = rbfBasedDistTransform(gamma, d);
        printf("%.5f -> %.5f\n", d, dT);
    }
    return 0;
}

int test7() {
    const int n = 50;
    const vector<int> cardinalities = {100};
    TruncatedExponentialDistribution distribution;
    distribution.build(0.01, 0.0001);
    printf("lambda0 = %.16f, lambda1 = %.16f\n", distribution.lambda0, distribution.lambda1);
    RealSolution x = sampleRnUniformly(n, cardinalities);
    double dSmallest = 10., dLargest = 552.;
    double gamma = findDistGamma(dSmallest, dLargest);
    auto myDist = [gamma](const RealSolution & x, const RealSolution & y) {
        return rbfBasedDistTransform(gamma, l2Norm(x, y));
    };
    for (int i = 0; i < 1; ++i) {
        double r = rnd.sample01Uniform();
        double dist = distribution.inverseCDF(r);
        RealSolution y = generateDistantSolution(x, cardinalities, myDist, dist, 100000, 500, 1000);
        printf("d(x, y) = %.5f, expected %.5f\n", myDist(x, y), dist);
    }
    return 0;
}

int test8() {
    const int n = 50;
    const vector<int> cardinalities = {100};
    TruncatedExponentialDistribution distribution;
    distribution.build(0.01, 0.0001);
    RealSolution x = sampleRnUniformly(n, cardinalities);
    double dSmallest = 10., dLargest = 552.;
    double gamma = findDistGamma(dSmallest, dLargest);
    auto permutations = generatePermutations(cardinalities, 10);
    auto invPerms = inversePermutations(permutations);
    auto myDist = [invPerms, gamma](const RealSolution & x, const RealSolution & y) {
        return rbfBasedDistTransform(gamma, untangledL2Norm(invPerms, x, y));
    };
    for (int i = 0; i < 10; ++i) {
        double r = rnd.sample01Uniform();
        double dist = distribution.inverseCDF(r);
        RealSolution yRandom = sampleRnUniformly(n, cardinalities);
        printf("d(x, yUAR) = %.5f\n", myDist(x, yRandom));
        RealSolution y = generateDistantSolution(x, cardinalities, myDist, dist, 1e5, 500, 1000);
        printf("d(x, y) = %.5f, expected %.5f\n", myDist(x, y), dist);
    }
    return 0;
}

double f (double x, void *params) {
    double alpha = *(double *) params;
    double f = log(alpha * x) / sqrt(x);
    return f;
}

int test9() {
    gsl_integration_workspace *w = gsl_integration_workspace_alloc (1000);

    double result, error;
    double expected = -4.0;
    double alpha = 1.0;

    gsl_function F;
    F.function = &f;
    F.params = &alpha;

    gsl_integration_qags (&F, 0, 1, 0, 1e-7, 1000,
                          w, &result, &error);

    printf ("result          = % .18f\n", result);
    printf ("exact result    = % .18f\n", expected);
    printf ("estimated error = % .18f\n", error);
    printf ("actual error    = % .18f\n", result - expected);
    printf ("intervals       = %zu\n", w->size);

    gsl_integration_workspace_free (w);

    return 0;
}

double f1(double x, void *params) {
    double gamma = ((double *)params)[0], lambda0 = ((double *)params)[1], lambda1 = ((double *)params)[2];
    double fValue = -1. / gamma / gamma * exp(lambda0 + lambda1 * x) * log(1 - x);
    return fValue;
}

double f2(double x, void *params) {
    double gamma = ((double *)params)[0], lambda0 = ((double *)params)[1], lambda1 = ((double *)params)[2];
    double fValue = 1. / gamma * exp(lambda0 + lambda1 * x) * sqrt(-log(1 - x));
    return fValue;
}

double numerically_integrate(double (*f)(double, void *), vector<double> params) {
    gsl_integration_workspace *w = gsl_integration_workspace_alloc (1000);
    double result, error;

    gsl_function F;
    F.function = f;
    F.params = params.data();
    double eps1 = 0.0003, eps2 = 0.00015;

    gsl_integration_qags (&F, eps1, 1. - eps2, 0, 1e-7, 1000, w, &result, &error);

    printf ("result          = % .18f\n", result);
    printf ("estimated error = % .18f\n", error);
    gsl_integration_workspace_free (w);
    return result;
}

int test10() {
    vector<double> alpha = {0.00660, 2.06256, -7.86309};
    numerically_integrate(f1, alpha);
    numerically_integrate(f2, alpha);
    return 0;

}

int main(int argc, char **argv) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    using fPtr = int (*)();
    vector<pair<fPtr, string>> tests = {
        {test, "0"},
        {test2, "2"},
        {test3, "3"},
        {test4, "4"},
        //{test5, "5"},
        {test6, "6"},
        {test7, "7"},
        //{test8, "8"},
        {test9, "Numerical integration"},
        {test10, "Numerical integration smooth"},
    };
    for (auto test : tests) {
        printf("--- Running test %s\n", test.second.c_str());
        auto t1 = chrono::high_resolution_clock::now();
        int ret = test.first();
        auto t2 = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> t = t2 - t1;
        printf("--- Done in %.5f ms\n\n", t.count());
        fflush(stdout);
        if (ret != 0) return ret;
    }
    return 0;
}

