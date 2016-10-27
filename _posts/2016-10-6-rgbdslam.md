---
layout:     post
title:      "RGBDSLAM 源码剖析"
subtitle:   " \"SLAM\""
date:       2016-10-06
author:     "Leiym"
header-img: "img/post-bg-2015.jpg"
tags:
    - vslam
---

> 一步一步讲解 RGBDSLAM 源码，深度剖析理论与算法。

### 源码剖析

#### OpenNIListener

**OpenNIListener** 类的主要功能是接收和同步 *kinect* 的rgb图像和深度图像，并将其转换为opencv所使用的图像格式进行处理。然后构造 **Node**，进行特征点检测匹配，以及计算相对运动。

其构造函数初始化了一些成员变量，并根据 *ParameterServer* 中的 *feature_detector_type* 、*feature_extractor_type* 创建检测子和提取子。然后调用了成员函数 *setupSubscribers*。

在 *setupSubscribers* 函数中，其使用 *message_filters* 同步rgb图像话题、深度图像话题、点云话题三个图像话题，并注册回调函数 *kinectCallback* ，保证三个话题的同步。

成员回调函数 *kinectCallback* 中，先后将rgb图像消息和深度图像消息转换为opencv图像格式（Mat），将点云消息转换为pcl点云格式。然后调用 *cameraCallback* 成员函数。

在 *cameraCallback* 中，做了最重要的工作——构造 *Node* ,计算相对运动。构造完 *Node* 后，分别调用了 *retrieveTransformations* 和 *callProcessing* 。

在 *retrieveTransformations* 中，**TODO**

在 *callProcessing* 中，处理一些数据，并根据参数决定是否多线程地执行 *processNode*。

在 *processNode* 中，利用 *GraphManager* 类型的成员变量进行里程计的计算。

#### Node
