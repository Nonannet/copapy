#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "runmem.h"

static PyObject* coparun(PyObject* self, PyObject* args) {
    const char *buf;
    Py_ssize_t buf_len;
    int result;

    if (!PyArg_ParseTuple(args, "y#", &buf, &buf_len)) {
        return NULL; /* TypeError set by PyArg_ParseTuple */
    }

    /* If parse_commands may run for a long time, release the GIL. */
    Py_BEGIN_ALLOW_THREADS
    result = parse_commands((uint8_t*)buf);
    Py_END_ALLOW_THREADS

    return PyLong_FromLong(result);
}

static PyObject* read_data_mem(PyObject* self, PyObject* args) {
    unsigned long rel_addr;
    unsigned long length;

    // Parse arguments: unsigned long (relative address), Py_ssize_t (length)
    if (!PyArg_ParseTuple(args, "nn", &rel_addr, &length)) {
        return NULL;
    }

    if (length <= 0) {
        PyErr_SetString(PyExc_ValueError, "Length must be positive");
        return NULL;
    }

    const char *ptr = (const char *)(data_memory + rel_addr);

    PyObject *result = PyBytes_FromStringAndSize(ptr, length);
    if (!result) {
        return PyErr_NoMemory();
    }

    return result;
}

static PyMethodDef MyMethods[] = {
    {"coparun", coparun, METH_VARARGS, "Pass raw command data to coparun"},
    {"read_data_mem", read_data_mem, METH_VARARGS, "Read memory and return as bytes"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef coparun_module = {
    PyModuleDef_HEAD_INIT,
    "coparun_module",  // Module name
    NULL,         // Documentation
    -1,           // Size of per-interpreter state (-1 for global)
    MyMethods
};

PyMODINIT_FUNC PyInit_coparun_module(void) {
    return PyModule_Create(&coparun_module);
}