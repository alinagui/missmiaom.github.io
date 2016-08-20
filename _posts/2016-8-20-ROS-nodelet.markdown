---
layout:     post
title:      "nodelet的使用方法以及传输时间测试"
subtitle:   " \"the nodelet of ROS\""
date:       2016-08-20
author:     "Leiym"
header-img: "img/post-bg-2015.jpg"
tags:
    - ROS
---

> 这篇博文介绍了有关 ROS 通信的包——nodelet的使用以及相关测试

### 概述

ROS，从软件构架的角度说，是一种基于消息传递通信的分布式多进程框架。ROS 进程之间的通信其实是基于标准的 TCP/IP 协议的，所以在传递 *Messages* 或者 *Services* 时，必须先经过一个打包的过程，而接收时也需要一个解包的过程，这些都会导致在传输过程中损耗不少时间。在数据量小，频率低的情况下，传输耗费的时间可以忽略不计。但当传输图像流，点云等数据量较大的消息，或者执行有一定的实时性要求的任务时，因为传输而耗费的时间就不得不考虑。

ROS 中需要有一个工具能够解决这个问题，**nodelet** 便诞生了。*nodelet* 的设计目的是提供一种方法，可以让多个算法程序在一个进程中用 *shared_ptr* 实现零复制通信（zero copy transport），以降低因为传输大数据而损耗的时间。

*nodelet* 设计目标：

* 使用现有的 C++ ROS 接口
* 允许 *nodelet* 之间进行零复制的传输
* 像插件一样不论时间先后动态的加载
* location transparent except for performance improvements
* 最小化 *nodelet* 的编程差异

### 运行机制

使用 *nodelet* 时，需要有一个 *nodelet_manager* 的进程来加载其他 *nodelet* 插件，而加载在一个 *nodelet_manager* 之中的 *nodelet* 之间就拥有零复制通信的条件。

### 编写方法

**nodelet 是基于 pluginlib 插件机制的，对于不太熟悉 pluginlib 的读者，我推荐先浏览一遍我之前的博客： [pluginlib详解](http://leiym.com/2016/08/10/ROS-pluginlib/)**

由于 *nodelet* 使用了插件机制，所以其使用方法和 *pluginlib* 的使用方法类似。具体使用方法有以下三步。

1. 继承 *nodelet::Nodelet* 基类，并实现 *onInit* 纯虚函数，以用于初始化。
2. 加入 *PLUGINLIB_EXPORT_CLASS* 宏将子类声明为插件类，并编译为动态库。
3. 使 ROS 工具链可以找到相应的 *nodelet* 插件。

这篇博文中，我使用两个 *nodelet* 插件分别来发布和接收有五千万元素的 *Int64MultiArray* ，并记录了传输和接收的时间，在文章末尾进行了一个粗略的对比。

#### 第一步：编写子类

##### *pub_with_nodelet.cpp* :

```
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ecl/threads/thread.hpp>
#include <std_msgs/Int64MultiArray.h>

namespace nodelet_test
{
  class Pub_nodelet : public nodelet::Nodelet
  {
  public:
    Pub_nodelet()
    {
    }
    ~Pub_nodelet()
    {
      pub_thread_.join();
    }
  private:
    virtual void onInit()
    {
      ros::NodeHandle& nh = getNodeHandle();

      pub = nh.advertise<std_msgs::Int64MultiArray>("nodelet",1);
      //ros::Subscriber sub = nh.subscribe("nodelet", 1, &Sub_nodelet::Callback, this);

      pub_thread_.start(&Pub_nodelet::publish, *this);
    }

    void publish()
    {
      std_msgs::Int64MultiArrayPtr array(new std_msgs::Int64MultiArray);
      array->data.resize(50000000);
      for(double i = 0; i < 50000000; i++)
        array -> data.push_back(std::rand());

      int num = 1;
      ros::Rate r(1);
      while(ros::ok())
      {
        ROS_INFO(" %d ,publish!", num++);
        pub.publish(array);
        ros::spinOnce();
        r.sleep();
      }
    }

    ros::Publisher pub;
    ecl::Thread pub_thread_;
  };
}

PLUGINLIB_EXPORT_CLASS(nodelet_test::Pub_nodelet, nodelet::Nodelet)
```

以下是分段解释：

```
  class Pub_nodelet : public nodelet::Nodelet
```

实现继承于 *nodelet::Nodelet* 基类的 *Pub_nodelet* 子类。

```
virtual void onInit()
{
  ros::NodeHandle& nh = getNodeHandle();

  pub = nh.advertise<std_msgs::Int64MultiArray>("nodelet",1);
  //ros::Subscriber sub = nh.subscribe("nodelet", 1, &Sub_nodelet::Callback, this);

  pub_thread_.start(&Pub_nodelet::publish, *this);
}
```

当 *nodelet* 插件类被 *nodelet_manager* 加载时，*nodelet* 插件类的 *onInit* 方法就会被调用，用于初始化插件类。 **onInit 方法不能被阻塞，只用于初始化，如果 nodelet 插件需要执行循环任务，要将其放入线程中执行。**

*getNodeHandle* 方法可以获得 *nodelet_manager* 的 *nodehandle* 。相应的 *getPrivateNodeHandle* 可以获得 *private nodehandle* 。

```
    void publish()
    {
      std_msgs::Int64MultiArrayPtr array(new std_msgs::Int64MultiArray);
      array->data.resize(50000000);
      for(double i = 0; i < 50000000; i++)
        array -> data.push_back(std::rand());

     ...
    }
```

*publish* 线程中，我声明了 *Int64MultiArrayPtr* 类型的消息 *array* ，为了方便之后比较用与不用 *nodelet* 的差别，我压入了五千万个随机数。至于消息类型的后缀 *Ptr*，是消息类型的 *shared_ptr*，即共享指针，它是 *roscpp* 来对进程内实现零复制的一个优化工具。 [shared_ptr 详情点击这里]( roscpp/Overview/Publishers and Subscribers#Intraprocess_Publishing)。

```
ROS_INFO(" %d ,publish!", num++);
```

每发布一条消息，使用 *ROS_INFO* 宏来打印时间和次序。


##### *sub_with_nodelet.cpp* ：

```
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <std_msgs/Int64MultiArray.h>


namespace nodelet_test
{

  class Sub_nodelet : public nodelet::Nodelet
  {
  public:
    Sub_nodelet() :
      num(1)
    {}

  private:
    virtual void onInit()
    {
      ros::NodeHandle& nh = getNodeHandle();
      //pub = nh.advertise<std_msgs::Int64MultiArray>("test",1);
      sub = nh.subscribe("nodelet", 1, &Sub_nodelet::Callback, this);

      //ros::spin();
    }

    void Callback(const std_msgs::Int64MultiArray::ConstPtr& msg)
    {
      ROS_INFO(" %d, subsrcibe!", num++);
    }

  private:
    //ros::Publisher pub;
    ros::Subscriber sub;
    int num;
  };
}

PLUGINLIB_EXPORT_CLASS(nodelet_test::Sub_nodelet, nodelet::Nodelet)
```

接收消息的 *nodelet* 插件，每接收到一条消息，就使用 *ROS_INFO* 宏来打印时间和次序。

#### 第二步：编译为动态库

```
PLUGINLIB_EXPORT_CLASS(nodelet_test::Pub_nodelet, nodelet::Nodelet)
```

上述程序中的最后一行便是使用 *PLUGINLIB_EXPORT_CLASS* 宏来将所写子类声明为插件类的。第一个参数为子类命名空间：：子类名，第二个参数为父类命名空间：：父类名。

然后，在 *CMakeLists.txt* 中添加 *add_library* 和 *target_link_libraries* 编译选项，然后编译，在工作空间文件夹下的 *lib* 文件夹下会生成相应的动态库。

#### 第三步：使 ROS 工具链可以找到相应的 *nodelet* 插件

这一步其实就是 *pluginlib* 的使用方法，不再赘述，不太明白的读者可以详读上面提到的博文。

##### *pub_nodelet.xml* :

```
<library path="lib/libpub_nodelet">
  <class name="nodelet_test/Pub_nodelet" type="nodelet_test::Pub_nodelet" base_class_type="nodelet::Nodelet">
    <description>
      publisher nodelet.
    </description>
  </class>
</library>
```
##### *sub_nodelet.xml* ：

```
<library path="lib/libsub_nodelet">
  <class name="nodelet_test/Sub_nodelet" type="nodelet_test::Sub_nodelet" base_class_type="nodelet::Nodelet">
    <description>
      subscriber nodelet.
    </description>
  </class>
</library>
```

##### *packgae.xml* :

```
...
<export>
  <nodelet plugin="${prefix}/pub_nodelet.xml" />
  <nodelet plugin="${prefix}/sub_nodelet.xml" />
</export>
...
```

**注意**，这里的标签一定要是 *nodelet* 表明是用在 *nodelet* 包的插件！

### 使用方法以及效果对比

先开启名为 *nodelet manager* 的 *nodelet_manager* :

`$ rosrun nodelet nodelet manager __name:=nodelet_manager`

然后分别开启 *sub_with_nodelet* 和 *pub_with_nodelet* 。

`$ rosrun nodelet nodelet load nodelet_test/Sub_nodelet nodelet_manager _output:=screen`

`$ rosrun nodelet nodelet load nodelet_test/Pub_nodelet nodelet_manager _output:=screen`

<img src="http://leiym.com/img/in-post/post-ros/nodelet_r7.png"/>

可以看到，接收到的时间和发布的时间几乎相差无几，那么如果不使用 nodelet 进行传递五千万元素的数组呢，对比结果如下：（注：运行环境均一样，数据也是随机生成的五千万元素的数组）

#### 在不同进程使用 *shared_ptr* 进行发布和接收：

五次的发布时间：

<img src="http://leiym.com/img/in-post/post-ros/nodelet_r1.png"/>

五次的接收时间：

<img src="http://leiym.com/img/in-post/post-ros/nodelet_r2.png"/>

可以看到，在不同进程中就算使用了 *shared_ptr* 也是会进行拷贝传输的，时间损耗大概在 **1.3s** 左右。

```
[ INFO] [1471502198.478018266]: publish!
[ INFO] [1471502199.776652182]: subsrcibe!

[ INFO] [1471502232.558265498]: publish!
[ INFO] [1471502233.881644185]: subsrcibe!

[ INFO] [1471502266.854639873]: publish!
[ INFO] [1471502268.175051612]: subsrcibe!

[ INFO] [1471502300.566516803]: publish!
[ INFO] [1471502301.859423722]: subsrcibe!

[ INFO] [1471502337.054734337]: publish!
[ INFO] [1471502338.323174927]: subsrcibe!
```

#### 在同一进程中使用 *shared_ptr* 进行发布和接收：

五次的发布和接收的时间：

<img src="http://leiym.com/img/in-post/post-ros/nodelet_r6.png"/>

可以看到，在同一进程中使用 *shared_ptr* 进行的是零拷贝复制，非常迅速，平均时间和使用 *nodelet* 相近。

#### 在同一进程中不使用 *shared_ptr* 进行发布和接收：

五次的发布和接收的时间：

<img src="http://leiym.com/img/in-post/post-ros/nodelet_r5.png"/>

可以看到，在同一进程中不使用 *shared_ptr* 进行的是拷贝复制，时间损耗也大概在 **1.3s** 左右，和在不同进程中分别发布接收的效果相同。

所以，运行时间的对比也证实了博文开头讲解的 *nodelet* 机制：其使用 *pluginlib* 动态插件机制，将不同的算法程序加载到同一个 *nodelet_manager* 进程中，使用 *shared_ptr* 来实现零复制通信，以降低因为传输大数据而损耗的时间。

---

### 后记

本篇博文讲解了 *nodelet* 的含义以及使用方法，希望能够帮到大家，如有错误请联系我！谢谢～
