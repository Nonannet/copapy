#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "runmem.h"
#include <stdlib.h>

static PyObject* coparun(PyObject* self, PyObject* args) {
    PyObject *handle_obj;
    const char *buf;
    Py_ssize_t buf_len;
    int result;

    // Expect: handle, bytes
    if (!PyArg_ParseTuple(args, "Oy#", &handle_obj, &buf, &buf_len)) {
        return NULL; /* TypeError set by PyArg_ParseTuple */
    }

    void *ptr = PyLong_AsVoidPtr(handle_obj);
    if (!ptr) {
        PyErr_SetString(PyExc_ValueError, "Invalid context handle");
        return NULL;
    }
    runmem_t *context = (runmem_t*)ptr;

    /* If parse_commands may run for a long time, release the GIL. */
    Py_BEGIN_ALLOW_THREADS
    result = parse_commands(context, (uint8_t*)buf);
    Py_END_ALLOW_THREADS

    return PyLong_FromLong(result);
}

static PyObject* read_data_mem(PyObject* self, PyObject* args) {
    PyObject *handle_obj;
    unsigned long rel_addr;
    unsigned long length;

    // Expect: handle, rel_addr, length
    if (!PyArg_ParseTuple(args, "Onn", &handle_obj, &rel_addr, &length)) {
        return NULL;
    }

    if (length <= 0) {
        PyErr_SetString(PyExc_ValueError, "Length must be positive");
        return NULL;
    }

    void *ptr = PyLong_AsVoidPtr(handle_obj);
    if (!ptr) {
        PyErr_SetString(PyExc_ValueError, "Invalid context handle");
        return NULL;
    }
    runmem_t *context = (runmem_t*)ptr;

    if (!context->data_memory || rel_addr + length > context->data_memory_len) {
        PyErr_SetString(PyExc_ValueError, "Read out of bounds");
        return NULL;
    }

    const char *data_ptr = (const char *)(context->data_memory + rel_addr);

    PyObject *result = PyBytes_FromStringAndSize(data_ptr, length);
    if (!result) {
        return PyErr_NoMemory();
    }

    return result;
}

static PyObject* create_target(PyObject* self, PyObject* args) {
    runmem_t *context = (runmem_t*)calloc(1, sizeof(runmem_t));
    if (!context) {
        return PyErr_NoMemory();
    }
    // Return the pointer as a Python integer (handle)
    return PyLong_FromVoidPtr((void*)context);
}

static PyObject* clear_target(PyObject* self, PyObject* args) {
    PyObject *handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj)) {
        return NULL;
    }
    void *ptr = PyLong_AsVoidPtr(handle_obj);
    if (!ptr) {
        PyErr_SetString(PyExc_ValueError, "Invalid handle");
        return NULL;
    }
    runmem_t *context = (runmem_t*)ptr;
    free_memory(context);
    free(context);
    Py_RETURN_NONE;
}

static PyMethodDef MyMethods[] = {
    {"coparun", coparun, METH_VARARGS, "Pass raw command data to coparun"},
    {"read_data_mem", read_data_mem, METH_VARARGS, "Read memory and return as bytes"},
    {"create_target", create_target, METH_NOARGS, "Create and return a handle to a zero-initialized runmem_t struct"},
    {"clear_target", clear_target, METH_VARARGS, "Free all memory associated with the given target handle"},
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