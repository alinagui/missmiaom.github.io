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
