#ifndef PARAMS_H
#define PARAMS_H

typedef struct {
    int M;
    int N;

    double dx;
    double dt;
    double c;
    double gamma;

    int i0;
    int j0;
    int intensity;
    int frame_start;

} Params;

int read_params(const char *filename, Params *p);

#endif