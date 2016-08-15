---
layout:     post
title:      "使用一根网线搭建ubuntu虚拟机与ubuntu物理机之间的局域网"
subtitle:   " \"ubuntu\""
date:       2016-08-12
author:     "Leiym"
header-img: "img/post-bg-2015.jpg"
tags:
    - ROS
---

> 这篇博文记录了如何使用一根网线建立ubuntu虚拟机与ubuntu物理机之间的局域网，这里我使用的时VMware的虚拟机和妙算进行局域网通信。

### 虚拟机配置步骤

#### 第一步，新建以太网

* 在网络连接中点击 *Add* 。

<img src="http://leiym.com/img/in-post/post-ros/network1.png"/>

* 新建 *Ethernet* ，点击 *Create* 。

<img src="http://leiym.com/img/in-post/post-ros/network2.png"/>

* 输入新建以太网名称，选择 *IPv4 Settings* 标签页， *Method* 选择 *Manual*，添加 *Address* ,这里填的是固定的 IP 地址，稍后通信时会用到，我设置为 *192.168.1.10* ， 子网掩码填 *255.255.255.0* 即可 。

<img src="http://leiym.com/img/in-post/post-ros/network3.png"/>

#### 第二步，配置虚拟机网络连接为桥接模式

具体配置步骤见另一篇博文：[桥接模式配置](http://leiym.com/2016/08/09/Q&A/#qvmware)

**注意：** 上述博文中，我桥接的是无线网卡，而这里要用到网线，所以 **要桥接到有线网卡！** 如下图：

<img src="http://leiym.com/img/in-post/post-ros/network_bridge.png"/>

### 物理机配置步骤

我使用妙算（nvidia tk1），除了固定ip设置为 *192.168.1.2* 以外，其他配置步骤和虚拟机的第一步一样，省略。

### 互 ping 测试

<img src="http://leiym.com/img/in-post/post-ros/network_ping.png"/>

这里只截取一个图，妙算 ping 虚拟机就省略了。

---

### 后记

好记性不如烂笔头...
