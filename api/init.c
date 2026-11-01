/*
    Return the file system path representation of the object.

    If the object is str or bytes, then allow it to pass through with
    an incremented refcount. If the object defines __fspath__(), then
    return the result of that method. All other types raise a TypeError.
*/
PyObject *
PyOS_FSPath(PyObject *path)
{
    _Py_IDENTIFIER(__fspath__);
    PyObject *func = NULL;
    PyObject *path_repr = NULL;

    if (PyUnicode_Check(path) || PyBytes_Check(path)) {
        Py_INCREF(path);
        return path;
    }

    func = _PyObject_LookupSpecial(path, &PyId___fspath__);
    if (NULL == func) {
        return PyErr_Format(PyExc_TypeError,
                            "expected str, bytes or os.PathLike object, "
                            "not %S",
                            path->ob_type);
    }

    path_repr = PyObject_CallFunctionObjArgs(func, NULL);
    Py_DECREF(func);
    if (!PyUnicode_Check(path_repr) && !PyBytes_Check(path_repr)) {
        Py_DECREF(path_repr);
        return PyErr_Format(PyExc_TypeError,
                            "expected __fspath__() to return str or bytes, "
                            "not %S",
                            path_repr->ob_type);
    }

    return path_repr;
}
