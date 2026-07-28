#include <math.h>

double mms_source(double x, double t) {
    return ((-1 + 0.10000000000000001*pow(M_PI, 2))*exp(t) + M_PI*cos(M_PI*x))*exp(-2*t)*sin(M_PI*x);
}
