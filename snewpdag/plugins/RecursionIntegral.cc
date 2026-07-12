#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>

static constexpr double negativeInfinity = -std::numeric_limits<double>::infinity();
static constexpr std::size_t stackWarningBytes = 6 * 1024 * 1024;

static double _logPower(const double x, const unsigned short exponent){
    if (exponent == 0) return 0.0;
    return x > 0.0 ? static_cast<double>(exponent) * std::log(x)
                   : negativeInfinity;
}

static double _logSumExp(const double (&terms)[3]){
    double maximum = negativeInfinity;
    for (const double term : terms){
        if (std::isnan(term)) return std::numeric_limits<double>::quiet_NaN();
        if (term > maximum)   maximum = term;
    }
    if (maximum == negativeInfinity) return negativeInfinity;
    double total = 0.0;
    for (const double term : terms) total += std::exp(term - maximum);
    return maximum + std::log(total);
}

static PyObject *_buildLogJTableRaw(PyObject *, PyObject *args){
    PyObject *nMaxObject;
    PyObject *mMaxObject;
    double a;
    double b;
    double p;
    double q;
    if (!PyArg_ParseTuple(args, "OOdddd:buildLogJTableRaw", &nMaxObject,
                          &mMaxObject, &a, &b, &p, &q)) return nullptr;

    const std::size_t requestedNMax = PyLong_AsSize_t(nMaxObject);
    if (requestedNMax == std::numeric_limits<std::size_t>::max() &&
        PyErr_Occurred()) return nullptr;
    const std::size_t requestedMMax = PyLong_AsSize_t(mMaxObject);
    if (requestedMMax == std::numeric_limits<std::size_t>::max() &&
        PyErr_Occurred()) return nullptr;
    if (requestedNMax > std::numeric_limits<unsigned short>::max() ||
        requestedMMax > std::numeric_limits<unsigned short>::max()){
        PyErr_SetString(PyExc_OverflowError,
                        "table dimensions must not exceed 65535");
        return nullptr;
    }
    const unsigned short nMax = static_cast<unsigned short>(requestedNMax);
    const unsigned short mMax = static_cast<unsigned short>(requestedMMax);

    const std::size_t rows = static_cast<std::size_t>(nMax) + 1;
    const std::size_t columns = static_cast<std::size_t>(mMax) + 1;
    if (rows > std::numeric_limits<std::size_t>::max() / columns){
        PyErr_NoMemory();
        return nullptr;
    }
    const std::size_t cells = rows * columns;
    if (cells > static_cast<std::size_t>(PY_SSIZE_T_MAX) / sizeof(double)){
        PyErr_NoMemory();
        return nullptr;
    }
    const std::size_t tableBytes = cells * sizeof(double);
    if (tableBytes > stackWarningBytes){
        char message[256];
        std::snprintf(
            message, sizeof(message),
            "logJTable (%zu x %zu) would require %.2f MiB on the stack; "
            "the extension allocates it on the heap instead",
            rows, columns, static_cast<double>(tableBytes) / (1024.0 * 1024.0));
        if (PyErr_WarnEx(PyExc_RuntimeWarning, message, 1) < 0) return nullptr;
    }

    PyObject *buffer = PyByteArray_FromStringAndSize(nullptr,
        static_cast<Py_ssize_t>(tableBytes));
    if (buffer == nullptr) return nullptr;
    double *const table = reinterpret_cast<double *>(PyByteArray_AS_STRING(buffer));

    const double s = a + p;
    if (s <= 0.0){
        Py_BEGIN_ALLOW_THREADS
        for (std::size_t index = 0; index < cells; ++index) table[index] = negativeInfinity;
        Py_END_ALLOW_THREADS
        return buffer;
    }

    const double logS = std::log(s);
    const double logA = a > 0.0 ? std::log(a) : negativeInfinity;
    const double logP = p > 0.0 ? std::log(p) : negativeInfinity;

    Py_BEGIN_ALLOW_THREADS
    for (unsigned short n = 0;; ++n){
        for (unsigned short m = 0;; ++m){
            double terms[3] = {negativeInfinity, negativeInfinity, negativeInfinity};
            terms[0] = _logPower(b, n) + _logPower(q, m) - logS;
            if (n > 0 && a > 0.0){
                terms[1] = std::log(static_cast<double>(n)) + logA - logS +
                           table[(static_cast<std::size_t>(n) - 1) * columns + m];
            }
            if (m > 0 && p > 0.0){
                terms[2] = std::log(static_cast<double>(m)) + logP - logS +
                           table[static_cast<std::size_t>(n) * columns + m - 1];
            }
            table[static_cast<std::size_t>(n) * columns + m] = _logSumExp(terms);
            if (m == mMax) break;
        }
        if (n == nMax) break;
    }
    Py_END_ALLOW_THREADS

    return buffer;
}

static PyMethodDef moduleMethods[] = {
    {"buildLogJTableRaw", _buildLogJTableRaw, METH_VARARGS,
     "Return the logJTable as a native-endian float64 byte buffer."},
    {nullptr, nullptr, 0, nullptr},
};

static PyModuleDef moduleDefinition = {
    PyModuleDef_HEAD_INIT,
    "_recursion_integral",
    "Compiled recurrence kernel for PoissonLagLikelihood_v5.",
    -1,
    moduleMethods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

PyMODINIT_FUNC PyInit__recursion_integral(){
    return PyModule_Create(&moduleDefinition);
}
