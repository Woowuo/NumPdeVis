#ifndef QTPYTHON_H
#define QTPYTHON_H
#include <QObject>
#include <Python.h>
#include <QDebug>
#include <map>

enum class RetType
{
    Null,
    Int,
    String,
    Char
};

class QtPython : public QObject
{
    Q_OBJECT
public:
    explicit QtPython(QObject *parent = nullptr);
    PyObject* InitFunc(const char * module_name, const char * func_name);
    PyObject* CallPyFunction1(PyObject * func_name, PyObject * args );
    void CallPyFunction2(const char * module_name, const char * func_name);
    ~QtPython();
signals:
public slots:
    void ReceivedFromWidget(PyObject * func_name, PyObject * args);
private:
    std::map<QString,PyObject*> used_modules;
    std::map<QString,PyObject*> used_functions;
};

#endif // QTPYTHON_H
