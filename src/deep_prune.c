#define PY_SSIZE_T_CLEAN
#define NOMINMAX
#include <windows.h>
#include <Python.h>

int deep_prune(char *fen, double beta, int depth)
{
    if (beta >= 500 || beta <= -500)
    {
        return 0;
    }
    
    extern PyObject* pFunc;

    PyObject *pArgs, *pResult;
    PyRun_SimpleString("from params import folder_model");
    PyRun_SimpleString("import numpy as np");
    PyRun_SimpleString("par_indi_del = np.load(folder_model + '/par_indi_del.npy')");
    PyErr_Print();

    // Pack arguments
    pArgs = PyTuple_New(2);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromStringAndSize(fen, strlen(fen)));
    PyTuple_SetItem(pArgs, 1, PyFloat_FromDouble(beta/100));

    // Get result from func
    pResult = PyObject_CallObject(pFunc, pArgs);

    if (pResult == NULL)
    {
        PyErr_Print();
        fprintf(stderr, "Failed call function % s \n", "evaluate_nn");
        return 0;
    }
    long result = PyLong_AsLong(pResult);
    
    PyErr_Print();

    return result;
}

void set_python_con()
{
    extern PyObject* pFunc;

    PyObject* pName, * pModule, * pDict;
    //PyObject* pFuncIn;
    // Build the name object
    pName = PyUnicode_FromString((char*)"evaluate_nn");

    // Initialize the python interpreter
    Py_Initialize();
    // Set the python path. Append to it the beta pruning folder
    PyRun_SimpleString("from sys import path");
    PyRun_SimpleString("path.append(r'C:\\Users\\User\\Documents\\tidhar\\barilan\\DeepLearning\\thesis\\chess\\tucano\\Tucano\\bpruning')");
    
    // Import the module
    pModule = PyImport_Import(pName);

    if (pModule == NULL)
    {
        PyErr_Print();
        fprintf(stderr, "Failed to load % s \n", "evaluate_nn");
        return;
    }
    pDict = PyModule_GetDict(pModule);
    // Import function
    pFunc = PyDict_GetItemString(pDict, (char*)"evaluate");

    //Py_DECREF(pName);
    //Py_DECREF(pModule);
    //Py_DECREF(pDict);

    }

void end_python_con()
{
    int val = Py_FinalizeEx();
    int x = Py_IsInitialized();


    if (val != 0) printf("Py_FinalizeEx returns -1");
}