#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "runmem.h"

/*
 * coparun(PyObject *self, PyObject *args)
 * Accepts a Python `bytes` (or objects supporting the buffer protocol).
 * We use the "y#" format in PyArg_ParseTuple which returns a pointer to
 * the internal bytes buffer and its length (Py_ssize_t). For safety and
 * performance we pass that pointer directly to parse_commands which expects
 * a uint8_t* buffer. If parse_commands needs the length, consider
 * extending its API to accept a length parameter.
 */
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

static PyMethodDef MyMethods[] = {
    {"coparun", coparun, METH_VARARGS, "Pass raw command data to coparun"},
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