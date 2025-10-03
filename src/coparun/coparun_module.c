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

static PyObject* read_data_mem(PyObject* self, PyObject* args) {
    unsigned long rel_addr;
    Py_ssize_t length;

    // Parse arguments: unsigned long (relative address), Py_ssize_t (length)
    if (!PyArg_ParseTuple(args, "nk", &rel_addr, &length)) {
        return NULL;
    }

    if (length <= 0) {
        PyErr_SetString(PyExc_ValueError, "Length must be positive");
        return NULL;
    }

    uint8_t *ptr = data_memory + rel_addr;

    PyObject *result = PyBytes_FromStringAndSize((const char *)ptr, length);
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