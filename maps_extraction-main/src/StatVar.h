#ifndef STAT_VAR_H
#define STAT_VAR_H

#include <stdio.h>

struct StatVar {
    double mean;
    double var;
    double std;
    double sumweight;
    unsigned int n;
    StatVar() {
        reset();
    }

    void reset() {
        mean = var = std = 0;
        sumweight = 0;
        n = 0;
    }

    void finalize() {
        if (sumweight > 0) {
            mean /= sumweight;
            var = var/sumweight - mean*mean;
            std = sqrt(var);
        }
        sumweight = -1;
    }
    void add(double x,double weight=1) {
        mean += weight*x;
        var += weight*x*x;
        sumweight += weight;
        n += 1;
    }
    void print(const std::string & prefix) {
        printf("%s: %d values, mean %f, std %f (var %f)\n",prefix.c_str(),n,mean,std,var);
    }

    bool empty() const {
        return n == 0;
    }
    bool ready() const {
        // check if this var has been finalized
        return sumweight < 0;
    }
};

struct StatVarMedian : public StatVar {
    std::vector<double> values;
    double median;
    void reset() {
        median = 0.0;
        StatVar::reset();
        values.clear();
    }

    void finalize() {
        StatVar::finalize();
        std::sort(values.begin(),values.end());
        if (values.size()>0) {
            median = values[values.size()/2];
        }
        values.clear();
    }
    void add(double x,double weight=1) {
        StatVar::add(x,weight);
        values.push_back(x);
    }

};

#endif // STAT_VAR_H
