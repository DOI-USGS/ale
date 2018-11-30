#ifndef EAL_H
#define EAL_H

#include <vector>
#include <string>

using namespace std;

string constructStateFromIsd(const string positionRotationData);

vector<double> getPosition(vector<vector<double>> coords, vector<double> coeffs, char *interp, double time);
vector<double> getVelocity(vector<vector<double>> coords, vector<double> coeffs, char *interp, double time, bool interpolation);

vector<double> getPosition(vector<double> coeffs, char *interp, double time);
vector<double> getVelocity(vector<double> coeffs, char *interp, double time, bool interpolation);

vector<double> getRotation(char *from, char *to, vector<vector<double>> rotations, vector<double> times, char *interp, double time);
vector<double> getAngularVelocity(char *from, char *to, vector<vector<double>> rotations, vector<double> times, char *interp, double time);

vector<double> getRotation(char *from, char *to, vector<double> coefficients, char *interp, double time);
vector<double> getAngularVelocity(char *from, char *to, vector<double> coefficients, char *interp, double time);

#endif // EAL_H
