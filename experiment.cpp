#include <getopt.h>
#include "impl.cpp"

int main(int argc, char **argv) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    while (true) {
        static struct option long_options[] = {
            {"problem", required_argument, 0, 'p'},
            {"size", required_argument, 0, 'n'},
            {"v", required_argument, 0, 'v'},
            {"t", required_argument, 0, 'T'},
            {"algorithm", required_argument, 0, 'a'},
            {"mu", required_argument, 0, 'm'},
            {"lambda", required_argument, 0, 'l'},
            {"useDdCrossover", no_argument, 0, 'u'},
            {"useComma", no_argument, 0, 'e'},
            {"mutation", required_argument, 0, 't'},
            {"budget", required_argument, 0, 'b'},
            {"instance", required_argument, 0, 'i'},
            {"cardinalities", required_argument, 0, 'c'},
            {"runs", required_argument, 0, 'r'},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = getopt_long (argc, argv, "", long_options, &option_index);
        int p, num;
        string cs;
        if (c == -1) break;
        switch (c) {
        case 'p':
            config.problem = string(optarg);
            if (!config.problems.count(config.problem)) {
                cerr << "No such problem " << config.problem << "\n";
                return 1;
            }
            break;
        case 'a':
            config.algorithm = string(optarg);
            if (!config.algorithms.count(config.algorithm)) {
                cerr << "No such algorithm " << config.algorithm << "\n";
                return 1;
            }
            break;
        case 't':
            config.intMutation = string(optarg);
            if (!config.intMutations.count(config.intMutation)) {
                cerr << "No such mutation " << config.intMutation << "\n";
                return 1;
            }
            break;
        case 'l':
            config.lambda = atoi(optarg);
            break;
        case 'm':
            config.mu = atoi(optarg);
            break;
        case 'n':
            config.N = atoi(optarg);
            break;
        case 'v':
            config.V = atoi(optarg);
            break;
        case 'T':
            config.t = atoi(optarg);
            break;
        case 'b':
            config.budget = atoi(optarg);
            break;
        case 'i':
            config.instance = atoi(optarg);
            break;
        case 'c':
            p = 0;
            cs = string(optarg);
            config.cardinalities.clear();
            while (p < cs.size()) {
                num = 0;
                for (; p < cs.size() and isdigit(cs[p]); ++p) num = num * 10 + cs[p] - '0';
                config.cardinalities.push_back(num);
                ++p;
            }
            break;
        case 'r':
            config.independentRuns = atoi(optarg);
            break;
        case 'u':
            config.useDdCrossover = true;
            break;
        case 'e':
            config.useComma = true;
            break;
        default:
            abort();
        }
    }
    config.printUsedConfigs();
    experiment();
    fflush(stdout);
    return 0;
}

