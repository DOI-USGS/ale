#ifndef EAL_H
#define EAL_H

#include <vector>
#include <string.h>

using namespace std;

vector<double> getPosition(vector<vector<double>> coords, vector<double> coeffs, char *interp, double time);
vector<double> getVelocity(vector<vector<double>> coords, vector<double> coeffs, char *interp, double time, bool interpolation);

vector<double> getPosition(vector<double> coeffs, char *interp, double time);
vector<double> getVelocity(vector<double> coeffs, char *interp, double time, bool interpolation);

#endif // EAL_H
