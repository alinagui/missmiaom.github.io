---
layout:     post
title:      "PCL学习笔记——持续更新"
subtitle:   " \"PCL\""
date:       2016-08-31
author:     "Leiym"
header-img: "img/post-bg-2015.jpg"
tags:
    - PCL vslam
---

### 开发环境

#### CMakeLists.txt

`find_package(PCL 1.7 REQUIRED COMPONENTS io commom)`

查找PCL库的两个组件io和common，版本要求1.7。

如果找到安装PCL，则以下变量自动被设置，否则不设置：
* PCL_FOUND：1
* PCL_INCLUDE_DIRS：头文件目录
* PCL_LIBRARIES：库文件名
* PCL_LIBRARY_DIRS：库文件目录
* PCL_VERSION：所找到的PCL的版本
* PCL_COMPONENTS：所有可用组件
* PCL_DEFINITIONS：所需要的预处理器定义和编译器标志

`include_directories(${PCL_INCLUDE_DIRS})`

`link_directories(${PCL_LIBRARY_DIRS})`

`add_definitions(${PCL_DEFINITIONS})`

添加PCL库的头文件路径，库文件路径，预处理。

### I/O

#### PCD (Point Cloud Data) 文件格式

##### PCD版本
在正1.0版本之前，用PCD_Vx来对PCD文件版本编号，代表PCD的0.x版本号，正式发布的是0.7版本（PCD_V7）。

##### 文件头

* VERSION ： 指定PCD版本
* FIELDS  ： 指定一个点可以有的每一个维度和字段的名字。
    FIELDS x y z                                # XYZ data
    FIELDS x y z rgb                            # XYZ + colors
    FIELDS x y z normal_x normal_y normal_z     # XYZ + surface normals
    FIELDS j1 j2 j3                             # moment invariants
    ...
* SIZE    ： 用字节数指定每一个维度的大小。
    unsigned char/char has 1 byte
    unsigned short/short has 2 bytes
    unsigned int/int/float has 4 bytes
    double has 8 bytes
* TYPE    ： 用一个字符指定每一个维度的类型。
    I - represents signed types int8 (char), int16 (short), and int32 (int)
    U - represents unsigned types uint8 (unsigned char), uint16 (unsigned short), uint32 (unsigned int)
    F - represents float types
* COUNT   ： 指定每一个维度包含的元素数目。
* WIDTH   ： 用点的数量表示点云数据集的宽度。
    1. 它能确定无序数据集的点云中点的个数。
    2. 它能确定有序点云数据集的宽度。
* HEIGHT  ： 用点的数目表示点云数据集的高度。
    1. 它表示有序点云数据集的高度。
    2. 对于无序数据集被设置为1 （用来检查一个数据集是否有序）。
* VIEWPOINT ： 指定数据集中点云的获取视点。
* POINTS  ： 指定点云中点的总数。
* DATA    ： 指定存储点云数据的数据类型。

**PCD文件的文件头部分必须以上面的顺序精确指定，并用换行隔开**
字符串“nan”表示NaN，该点不存在或非法。

#### 从PCD文件中读取点云数据

`#include <pcl/io/pcd_io.h>`  pcd读写类相关的头文件

`#include <pcl/point_types.h>`  PCl中支持的点类型头文件

`pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>;`
创建PointCloud的共享指针并实例化

`pcl::io::loadPCDFile<pcl::PointXYZ> ("test_pcd.pcd", *cloud);`
从test.pcd中读取点云并存入cloud中

`pcl::io::savePCDFileASCII ("test_pcd.pcd", cloud);`
将点云cloud存入test_pcd.pcd点云中
