---
layout:     post
title:      "Linux下 使用C++按名称顺序读取文件夹中所有文件名的方法"
subtitle:   " \"C++\""
date:       2016-09-20
author:     "Leiym"
header-img: "img/post-bg-2015.jpg"
tags:
    - C++
---

> 简单记录一下常用的 C++ 按顺序读取文件夹中所有文件名的方法

- [代码](#代码)
- [解释](#解释)
  - [dirent.h](#direnth)
  - [DIR](#dir)
  - [struct dirent](#struct-dirent)
  - [opendir](#opendir)
  - [closedir](#closedir)
  - [readdir](#readdir)
  - [rewinddir](#rewinddir)
  - [seekdir](#seekdir)
  - [telldir](#telldir)


### 代码

```
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <vector>

using namespace std;

int readFileList(char *basePath, vector<string>& files)
{
  DIR *dir;
  struct dirent *dirptr;

  if((dir = opendir(basePath)) == NULL)
    {
      perror("Open dir error!!!");
      return 1;
    }

  int count = 0;
  while((dirptr = readdir(dir)) != NULL)
    {
      if(strcmp(dirptr->d_name, ".") == 0 || strcmp(dirptr->d_name, "..") == 0)
        continue;

      else if(dirptr->d_type == 8)       //file
        {
          files.push_back(dirptr->d_name);
          count++;
        }

      //else if(dirptr->d_type == 10)    //link file
      //else if(dirptr->d_type == 4)     //dir
    }

  //排序，按从小到大排序
  sort(files.begin(), files.end());

  closedir(dir);

  return count;
}
```

### 解释

#### <dirent.h>

POSIX.1标准定义的unix类目录操作的头文件，包含了许多UNIX系统服务的函数原型，例如opendir函数、readdir函数、opendir函数。

#### DIR

属于 <dirent.h> 中的一个结构体，表示目录流，保存目录的有关信息。

```
struct __dirstream
{
  void *__fd;                     /* `struct hurd_fd' pointer for descriptor. */
  char *__data;                   /* Directory block. */
  int __entry_data;               /* Entry number `__data' corresponds to. */
  char *__ptr;                    /* Current pointer into the block. */
  int __entry_ptr;                /* Entry number `__ptr' corresponds to. */
  size_t __allocation;            /* Space allocated for the block. */
  size_t __size;                  /* Total valid data in the block. */
  __libc_lock_define (, __lock)   /* Mutex lock for this structure. */
};
typedef struct __dirstream DIR;
```

#### struct dirent

```
struct dirent
{
  long d_ino;                 /* inode number 索引节点号 */
  off_t d_off;                /* offset to this dirent 在目录文件中的偏移 */
  unsigned short d_reclen;    /* length of this d_name 文件名长 */
  unsigned char d_type;       /* the type of d_name 文件类型 */
  char d_name [NAME_MAX+1];   /* file name (null-terminated) 文件名，最长255字符 */
}
```

#### opendir

`DIR  *opendir(const char *);`

* 作用：打开目录句柄。

* 参数：指定的目录。

* 返回值：如果函数成功运行，将返回一组目录流，否则返回错误，可以再函数最前面加上 “@” 来隐藏错误。

#### closedir

`int  closedir(DIR *);`

* 作用：关闭参数所指的目录流。

* 参数：要关闭的目录流的指针。

* 返回值：成功返回 0 ，失败返回 -1 。

#### readdir

`struct dirent *readdir(DIR *);`

* 作用：读取指定目录中一个文件信息

* 参数：要读取的目录流的指针

* 返回值：结构体 dirent 指针。

**注意**：当读取完成时，会自动将 DIR 指针的读取位置加一，所以多次使用调用该函数去读取指定目录中文件时，会 **按某一顺序** （笔者猜测为在磁盘上的索引顺序） 读取文件，并不一定是按照名称或者时间顺序。

#### rewinddir

`void rewinddir(DIR *);`

* 作用：重置 DIR 目录目前的读取位置为初始读取位置。

* 参数：需要重置读取位置的 DIR 的指针。

#### seekdir

`void seekdir(DIR *, long );`

* 作用：设置参数 DIR 目录流当前的读取位置。

* 参数：DIR* ，需要设置读取位置的目录流的指针； long ，表示距离目录文件初始读取位置的偏移量。

#### telldir

`long int telldir(DIR *);`

* 作用：返回目录流当前读取文件的位置。

* 参数：DIR，目录流。

* 返回值：当前读取位置距离目录文件开头的偏移量，有错误发生时返回-1。
