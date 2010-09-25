#include <boost/python.hpp>
#include <boost/python/object.hpp>
#include "iterators.hpp"
#include <algorithm>
#include <cmath>

using namespace std;

namespace bp = boost::python;

namespace sdopencv
{

StepFunc::StepFunc(vector<double> const &thresholds, vector<double> const &values)
{
    if(thresholds.size()+1 != values.size())
    {
        PyErr_SetString(PyExc_ValueError, "The number of thresholds must be the number of values minus 1 in initialising class 'StepFunc'.");
        throw bp::error_already_set();
    }
    
    int i = thresholds.size()-1;
    while(--i >= 0)
    {
        if(thresholds[i] > thresholds[i+1])
        {
            PyErr_SetString(PyExc_ValueError, "The thresholds must be in ascending order in initialising class 'StepFunc'.");
            throw bp::error_already_set();
        }
    }
    
    this->thresholds = thresholds;
    this->values = values;
}

double StepFunc::operator()(double input)
{
    return values[upper_bound(thresholds.begin(), thresholds.end(), input)-thresholds.begin()];
}


LUTFunc::LUTFunc(double low, double high, double output_low, double output_high, vector<double> const &output)
    : low(low), high(high), output_low(output_low), output_high(output_high), output(output)
{
    if(low >= high)
    {
        PyErr_SetString(PyExc_ValueError, "The value of 'low' is not less than the value of 'high' in initialising class 'LUTFunc'.");
        throw bp::error_already_set();
    }
    interval = (high-low) / output.size();
}

double LUTFunc::operator()(double input)
{
    if(input < low) return output_low;
    if(input >= high) return output_high;
    return output[(int)floor((input-low)/interval)];
}

LUTFunc::operator StepFunc() const
{
    vector<double> thresholds, values;
    int i, N = output.size();
    
    thresholds.resize(N+1); 
    for(i = 0; i <= N; ++i) thresholds[i] = low + i*interval;
    values.resize(N+2); 
    values[0] = output_low; 
    values[N+1] = output_high;
    for(i = 0; i < N; ++i) values[i+1] = output[i];
    
    return StepFunc(thresholds, values);
}


StumpFunc::operator StepFunc() const
{
    vector<double> thresholds, values;
    thresholds.resize(1); thresholds[0] = threshold;
    values.resize(2); values[0] = neg_val; values[1] = pos_val;
    return StepFunc(thresholds, values);
}


}
