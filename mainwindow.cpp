#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <unistd.h>
#include <QPushButton>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QPushButton * run = new QPushButton("run taichi",this);
    QtPython * module = new QtPython(this);
    // PyObject * func_obj = module->InitFunc("fractl","running");
    connect(run,&QPushButton::clicked,module,[=]()
    {
        module->CallPyFunction2("fractl","running");

    }
    );

}

MainWindow::~MainWindow()
{
    delete ui;
}
