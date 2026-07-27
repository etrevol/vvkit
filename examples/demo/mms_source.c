#include <math.h>

double mms_source(double x, double t) {
    return -exp(-t)*sin(M_PI*x) + 0.10000000000000001*pow(M_PI, 2)*exp(-t)*sin(M_PI*x) + M_PI*exp(-2*t)*sin(M_PI*x)*cos(M_PI*x);
}
