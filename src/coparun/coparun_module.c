/*
 * file: coparun_module.c
 * Description: This file defines a Python C extension module that provides an
 * interface to the core functions of the coparun runner, to run copapy programs
 * directly local in Python.
 */ 

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "runmem.h"
#include <stdlib.h>
#include <stddef.h>

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
    parse_commands(context, (uint8_t*)buf, (uint32_t)buf_len);
    result = context->state_flag;
    Py_END_ALLOW_THREADS

    return PyLong_FromLong(result);
}

static PyObject* read_data_mem(PyObject* self, PyObject* args) {
    PyObject *handle_obj;
    Py_ssize_t rel_addr;
    Py_ssize_t length;

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

    if (!context->data_memory || rel_addr < 0 || rel_addr + length > (Py_ssize_t)context->data_memory_len) {
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

// Exporter of the data memory of a context for the buffer protocol. It counts the
// buffers in use (memoryviews, numpy arrays ...): the memory must not be freed
// while exports is not zero.
typedef struct {
    PyObject_HEAD
    char *ptr;
    Py_ssize_t length;
    Py_ssize_t exports;
} DataMemObject;

static int DataMem_getbuffer(PyObject *obj, Py_buffer *view, int flags) {
    DataMemObject *self = (DataMemObject *)obj;
    if (PyBuffer_FillInfo(view, obj, self->ptr, self->length, 0, flags) < 0) {
        return -1;
    }
    self->exports++;
    return 0;
}

static void DataMem_releasebuffer(PyObject *obj, Py_buffer *view) {
    ((DataMemObject *)obj)->exports--;
}

static PyBufferProcs DataMem_as_buffer = {DataMem_getbuffer, DataMem_releasebuffer};

static PyMemberDef DataMem_members[] = {
    {"exports", Py_T_PYSSIZET, offsetof(DataMemObject, exports), Py_READONLY, "Number of buffers in use"},
    {NULL}
};

static PyTypeObject DataMemType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "coparun_module.DataMem",
    .tp_basicsize = sizeof(DataMemObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Data memory of a target for the buffer protocol (no copy)",
    .tp_as_buffer = &DataMem_as_buffer,
    .tp_members = DataMem_members,
};

static PyObject* data_mem(PyObject* self, PyObject* args) {
    PyObject *handle_obj;

    // Expect: handle
    if (!PyArg_ParseTuple(args, "O", &handle_obj)) {
        return NULL;
    }

    void *ptr = PyLong_AsVoidPtr(handle_obj);
    if (!ptr) {
        PyErr_SetString(PyExc_ValueError, "Invalid context handle");
        return NULL;
    }
    runmem_t *context = (runmem_t*)ptr;

    if (!context->data_memory) {
        PyErr_SetString(PyExc_ValueError, "No data memory allocated");
        return NULL;
    }

    DataMemObject *obj = PyObject_New(DataMemObject, &DataMemType);
    if (!obj) {
        return NULL;
    }
    obj->ptr = (char *)context->data_memory;
    obj->length = (Py_ssize_t)context->data_memory_len;
    obj->exports = 0;
    return (PyObject *)obj;
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
    {"data_mem", data_mem, METH_VARARGS, "Data memory of the target as object with the buffer protocol (no copy)"},
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
    if (PyType_Ready(&DataMemType) < 0) {
        return NULL;
    }
    return PyModule_Create(&coparun_module);
}