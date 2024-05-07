#include "qtpython.h"
#define Debug
QtPython::QtPython(QObject *parent)
    : QObject{parent}
{
    Py_Initialize();
    if( !Py_IsInitialized() )
        qDebug()<<"Initialization fails"<<Qt::endl;
    chdir("./debug/python");//change directory of python files
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./')");
    //QString setSysPath = QString("sys.path.append('%1')").arg(QCoreApplication::applicationDirPath());
    //PyRun_SimpleString(setSysPath.toStdString().c_str());
}
void QtPython::ReceivedFromWidget(PyObject * func_name, PyObject * args )
{
    CallPyFunction1(func_name, args);
}
PyObject * QtPython::InitFunc(const char * module_name, const char * func_name)
{
    //check if the module is already in use
    auto module = used_modules.find(module_name);
    auto func = used_functions.find(func_name);
    PyObject* pModule;
    if(module == used_modules.end()&& used_modules.size() != 0)
    {
        qDebug()<<"Module "<<module_name<<" in use, so it won't be initialized again."<<Qt::endl;
        if(func == used_functions.end() && used_functions.size() != 0)
        {
            qDebug()<<"Function "<<func_name<<" in use, so CallPyFunction will be returned with null pointer"<<Qt::endl;
            return nullptr;
        }
        pModule = module->second;
    }
    // The module is not in use.
    else
    {
        pModule = PyImport_ImportModule(module_name);
        if (!pModule)
        {
            qDebug()<<"Get module pointer fails, CallPyFunction will be returned with null pointer"<<Qt::endl;
            return nullptr;
        }
        used_modules[module_name] = pModule;//Mark module_name as used
    }

    PyObject* pFunc= PyObject_GetAttrString(pModule,func_name);
    if(!pFunc)
    {
        qDebug()<<"Getting function pointer fails, null pointer will be returned"<<Qt::endl;
        return nullptr;
    }
    used_functions[func_name] = pFunc;// Mark function as used
    return pFunc;
}
PyObject* QtPython::CallPyFunction1(PyObject * func_name, PyObject * args)
{
    //check if the module is already in use

    return PyObject_CallObject(func_name,args);

}
void QtPython::CallPyFunction2(const char* module_name, const char* func_name)
{
    PyObject * func;
    auto it = used_functions.find(func_name);
    if(used_functions.size() == 0 || it == used_functions.end())
    {
        func = InitFunc(module_name,func_name); //function is not yet used
    }
    else
        func = it->second; //This function is used already
    PyObject_CallObject(func,nullptr);
    //Py_FinalizeEx();
}
QtPython::~QtPython()
{
    Py_Finalize();
}
