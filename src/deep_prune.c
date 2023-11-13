#define PY_SSIZE_T_CLEAN
#define NOMINMAX
#include <windows.h>

#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>

#include <Python.h>
#include "globals.h"

int deep_prune(char* fen, int beta)
{
	// Used for base reuslts
	//int r = rand() % 2;
	//return r;

	extern PyObject* pFunc;

	PyObject* pArgs, * pResult;
	PyRun_SimpleString("from params import folder_model");
	PyRun_SimpleString("import numpy as np");
	PyErr_Print();

	double d_beta = (double)beta / 200.0;
	// Pack arguments
	pArgs = PyTuple_New(2);
	PyTuple_SetItem(pArgs, 0, PyUnicode_FromStringAndSize(fen, strlen(fen)));
	PyTuple_SetItem(pArgs, 1, PyFloat_FromDouble(d_beta));

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
	srand(time(NULL));
	//PyObject* pFuncIn;
	// Build the name object
	pName = PyUnicode_FromString((char*)"evaluate_nn");

	// Initialize the python interpreter
	Py_Initialize();
	// Set the python path. Append to it the beta pruning folder
	PyRun_SimpleString("from sys import path");
	PyRun_SimpleString("path.append(r'C:\\Users\\User\\Documents\\tidhar\\barilan\\DeepLearning\\thesis\\chess\\tucano\\Tucano\\data\\train_model') ");

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

/*
void load_trace_model()
{
char pt_file[] = "C:\\Users\\User\\Documents\\tidhar\\barilan\\DeepLearning\\thesis\\chess\\tucano\\Tucano\\data\\train_model\\halfka_clear\\trace_model.pt";
torch::jit::script::Module module;
try
{
module = torch::jit::load(pt_file);
}
catch (const c10::Error& e)
{
std::cerr << "error loading trace_model.pt\n";
}

}
*/