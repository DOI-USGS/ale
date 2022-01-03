#include "ale/Load.h"

#include <nlohmann/json.hpp>

#include <Python.h>

#include <string>
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;
using namespace std;

namespace ale {
  std::string getPyTraceback() {
    PyObject* err = PyErr_Occurred();
    if (err != NULL) {
        PyObject *ptype, *pvalue, *ptraceback;
        PyObject *pystr, *module_name, *pyth_module, *pyth_func;
        char *str;
        char *full_backtrace;
        char *error_description;

        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        pystr = PyObject_Str(pvalue);
        str = PyBytes_AS_STRING(PyUnicode_AsUTF8String(pystr));
        error_description = strdup(str);

        // See if we can get a full traceback
        module_name = PyUnicode_FromString("traceback");
        pyth_module = PyImport_Import(module_name);
        Py_DECREF(module_name);

        if (pyth_module == NULL) {
            throw runtime_error("getPyTraceback - Failed to import Python traceback Library");
        }

        pyth_func = PyObject_GetAttrString(pyth_module, "format_exception");
        PyObject *pyth_val;
        pyth_val = PyObject_CallFunctionObjArgs(pyth_func, ptype, pvalue, ptraceback, NULL);

        pystr = PyObject_Str(pyth_val);
        str = PyBytes_AS_STRING(PyUnicode_AsUTF8String(pystr));
        full_backtrace = strdup(str);
        Py_DECREF(pyth_val);

        std::string join_cmd = "trace = ''.join(list(" + std::string(full_backtrace) + "))";
        PyRun_SimpleString(join_cmd.c_str());

        PyObject *evalModule = PyImport_AddModule( (char*)"__main__" );
        PyObject *evalDict = PyModule_GetDict( evalModule );
        PyObject *evalVal = PyDict_GetItemString( evalDict, "trace" );
        full_backtrace = PyBytes_AS_STRING(PyUnicode_AsUTF8String(evalVal));

        return std::string(error_description) + "\n" + std::string(full_backtrace);
    }

    // no traceback to return
    return "";
  }

  std::string loads(std::string filename, std::string props, std::string formatter, int indent, bool verbose) {
    static bool first_run = true;

    if(first_run) {
       // Initialize the Python interpreter but only once.
       first_run = !first_run;
       Py_Initialize();
    }

    // Import the file as a Python module.
    PyObject *pModule = PyImport_Import(PyUnicode_FromString("ale"));
    if(!pModule) {
      throw runtime_error("Failed to import ale. Make sure the ale python library is correctly installed.");
    }
    // Create a dictionary for the contents of the module.
    PyObject *pDict = PyModule_GetDict(pModule);

    // Get the add method from the dictionary.
    PyObject *pFunc = PyDict_GetItemString(pDict, "loads");
    if(!pFunc) {
      // import errors do not set a PyError flag, need to use a custom
      // error message instead.
      throw runtime_error("Failed to import ale.loads function from Python."
                         "This Usually indicates an error in the Ale Python Library."
                         "Check if Installed correctly and the function ale.loads exists.");
    }


    // Create a Python tuple to hold the arguments to the method.
    PyObject *pArgs = PyTuple_New(5);
    if(!pArgs) {
      throw runtime_error(getPyTraceback());
    }

    // Set the Python int as the first and second arguments to the method.
    PyObject *pStringFileName = PyUnicode_FromString(filename.c_str());
    PyTuple_SetItem(pArgs, 0, pStringFileName);
    Py_INCREF(pStringFileName); // take ownership of reference

    PyObject *pStringProps = PyUnicode_FromString(props.c_str());
    PyTuple_SetItem(pArgs, 1, pStringProps);
    Py_INCREF(pStringProps); // take ownership of reference

    PyObject *pStringFormatter = PyUnicode_FromString(formatter.c_str());
    PyTuple_SetItem(pArgs, 2, pStringFormatter);
    Py_INCREF(pStringFormatter); // take ownership of reference

    PyObject *pIntIndent = PyLong_FromLong((long) indent);
    PyTuple_SetItem(pArgs, 3, pIntIndent);
    Py_INCREF(pIntIndent); // take ownership of reference

    PyObject *pBoolVerbose = Py_False;
    if (!verbose) {
      pBoolVerbose = Py_True;
    }
    PyTuple_SetItem(pArgs, 4, pBoolVerbose);
    Py_INCREF(pBoolVerbose); // take ownership of reference


    // Call the function with the arguments.
    PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
    if(!pResult) {
      throw invalid_argument("No Valid instrument found for label.");
    }

    PyObject *pResultStr = PyObject_Str(pResult);
    PyObject *temp_bytes = PyUnicode_AsUTF8String(pResultStr); // Owned reference

    if(!temp_bytes){
      throw invalid_argument(getPyTraceback());
    }

    std::string cResult;

    char *temp_str = PyBytes_AS_STRING(temp_bytes); // Borrowed pointer
    cResult = temp_str; // copy into std::string

    Py_DECREF(pResultStr);

    Py_DECREF(pStringFileName);
    Py_DECREF(pStringProps);
    Py_DECREF(pStringFormatter);
    Py_DECREF(pIntIndent);
    Py_DECREF(pBoolVerbose);

    Py_DECREF(pArgs);

    return cResult;
  }

  json load(std::string filename, std::string props, std::string formatter, bool verbose) {
    std::string jsonstr = loads(filename, props, formatter, verbose);
    return json::parse(jsonstr);
  }
}
