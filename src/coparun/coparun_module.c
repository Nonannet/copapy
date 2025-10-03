#define PY_SSIZE_T_CLEAN
#include <Python.h>

// A simple C function exposed to Python
static PyObject* my_function(PyObject* self, PyObject* args) {
    int a, b;
    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
        return NULL;  // Error if arguments aren't two integers
    }
    return PyLong_FromLong(a + b);  // Return sum as Python integer
}

// Method definitions
static PyMethodDef MyMethods[] = {
    {"add", my_function, METH_VARARGS, "Adds two numbers"},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Module definition
static struct PyModuleDef my_module = {
    PyModuleDef_HEAD_INIT,
    "my_module",  // Module name
    NULL,         // Documentation
    -1,           // Size of per-interpreter state (-1 for global)
    MyMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_my_module(void) {
    return PyModule_Create(&my_module);
}