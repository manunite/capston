#include <Python.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <string>
#include <utility>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <sstream>
#include <unistd.h>

using namespace std;


char parameter1[100];
char parameter2[100];
char commandBuffer[100];
string result;
char input[100];
vector<pair<string,string> > vec;

template <class T>
const string& operator+=(string &str,const T &_ty)
{
	ostringstream os;
	os << _ty;
	str += os.str();
	return str;
}

static PyObject* write_log(PyObject *self,PyObject *args)
{
	char *msg;
	
	 if(!PyArg_ParseTuple(args, "s", &msg))
        return NULL;
	
	FILE *fp = fopen("soundtext.txt","r");
	for (int i = 0; i < 72; i++)
	{
		fscanf(fp,"%s %s", parameter1, parameter2);
		string A = parameter1;
		string B = parameter2;
		vec.push_back(make_pair(A, B));
	}
	
	vector<pair<string, string> >::iterator iterator = vec.begin();

	for (iterator = vec.begin(); iterator != vec.end(); iterator++)
	{
		if (iterator->first == msg) {
			result = iterator->second;
			break;
		}
	}
	
	//if(result == 0) return Py_BuildValue("i",0);
	
	string Str;
	Str += result;
	
	cout << Str <<endl;
	
	sprintf(commandBuffer,"/usr/bin/play -q %s",Str.c_str());
	system(commandBuffer);
	
	return Py_BuildValue("i",0);
}

static PyMethodDef methods[] =
{
    {"wlog", write_log, METH_VARARGS},
    {NULL, NULL}
};


extern "C" void initmylib()
{
	(void)Py_InitModule("mylib", methods);
}
