---
layout:     post
title:      "C++ 读取文件与字符串类型装换"
subtitle:   " \"C++\""
date:       2016-09-21
author:     "Leiym"
header-img: "img/post-bg-2015.jpg"
tags:
    - C++
---

> 简单记录 C++ 读取文件以及字符串转换的一些操作


- [读取文件](#读取文件)
- [字符串类型转换](#字符串类型转换)
	- [字符串数字之间的转换](#字符串数字之间的转换)
		- [string --> char *](#string-char-)
		- [char * -->string](#char-string)
		- [char * -->CString](#char-cstring)
		- [CString --> char *](#cstring-char-)
		- [string --> CString](#string-cstring)
		- [CString --> string](#cstring-string)
		- [double/float->CString](#doublefloat-cstring)
		- [CString->double](#cstring-double)
		- [string->double](#string-double)
	- [数字转字符串：使用sprintf()函数](#数字转字符串使用sprintf函数)
	- [字符串转数字：使用sscanf()函数](#字符串转数字使用sscanf函数)
	- [使用stringstream类](#使用stringstream类)
		- [ostringstream](#ostringstream)
		- [istringstream](#istringstream)


### 读取文件

#### 文件操作头文件以及类

```
#include <fstream>  
ofstream         //文件写操作 内存写入存储设备   
ifstream         //文件读操作，存储设备读区到内存中  
fstream          //读写操作，对打开的文件可进行读写操作
```
#### open

`void open ( const char * filename, ios_base::openmode mode = ios_base::in | ios_base::out );`

* *filename* ：操作文件名。
* *mode* ：打开文件方式，有如下几种。

|选项|意义|
|:---:|:---:|
|ios::in|为输入(读)而打开文件|
|ios::out|为输出(写)而打开文件|
|ios::ate|初始位置：文件尾|
|ios::app|所有输出附加在文件末尾|
|ios::trunc|如果文件已存在则先删除该文件|
|ios::binary|二进制方式|

这些方式是能够以“或”运算（“|”）的方式进行组合使用的。

####

### 字符串类型转换

**转载，原文：http://www.cnblogs.com/luxiaoxun/**

#### 字符串数字之间的转换

##### string --> char *

```
string str("OK");
char * p = str.c_str();
```

##### char * -->string

```
char *p = "OK";
string str(p);
```
##### char * -->CString

```
char *p ="OK";
CString m_Str(p);

//OR

CString m_Str;
m_Str.Format("%s",p);
```

##### CString --> char *

```
CString str("OK");
char * p = str.GetBuffer(0);
...
str.ReleaseBuffer();
```

##### string --> CString

```
CString.Format("%s", string.c_str());  
```

##### CString --> string

```
string s(CString.GetBuffer(0));  
//GetBuffer()后一定要ReleaseBuffer()，否则就没有释放缓冲区所占的空间，CString对象不能动态增长了。
```

##### double/float->CString

```
double data;
CString.Format("%.2f",data); //保留2位小数
```

##### CString->double

```
CString s="123.12";
double   d=atof(s);   
```

##### string->double

```
double d=atof(s.c_str());
```

#### 数字转字符串：使用sprintf()函数

```
char str[10];
int a=1234321;
sprintf(str,"%d",a);

//--------------------

char str[10];
double a=123.321;
sprintf(str,"%.3lf",a);

//--------------------

char str[10];
int a=175;
sprintf(str,"%x",a);//10进制转换成16进制，如果输出大写的字母是sprintf(str,"%X",a)

//--------------------

char *itoa(int value, char* string, int radix);
//同样也可以将数字转字符串，不过itoa()这个函数是平台相关的（不是标准里的），故在这里不推荐使用这个函数。
```

#### 字符串转数字：使用sscanf()函数

```
char str[]="1234321";
int a;
sscanf(str,"%d",&a);

//----------------

char str[]="123.321";
double a;
sscanf(str,"%lf",&a);

//----------------

char str[]="AF";
int a;
sscanf(str,"%x",&a); //16进制转换成10进制

//另外也可以使用atoi(),atol(),atof().
```

#### 使用stringstream类

##### ostringstream

```
//用ostringstream对象写一个字符串，类似于sprintf()
ostringstream s1;
int i = 22;
s1 << "Hello " << i << endl;
string s2 = s1.str();
cout << s2;
```

##### istringstream

```
//用istringstream对象读一个字符串，类似于sscanf()
istringstream stream1;
string string1 = "25";
stream1.str(string1);
int i;
stream1 >> i;
cout << i << endl;  // displays 25
```
