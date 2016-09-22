---
layout:     post
title:      "C++ Boost 之 线程操作"
subtitle:   " \"C++\""
date:       2016-09-22
author:     "Leiym"
header-img: "img/post-bg-2015.jpg"
tags:
    - C++
    - Boost
---

> 本文记录C++ Boost库的线程操作

### 使用boost库

```
#include <boost/thread.hpp>
using namespace boost;
```

在 Linux/UNIX 下链接 thread 库需要使用 -lpthread 选项来链接 POSIX 线程库。

### 时间功能

thread 库直接利用 date_time 库提供了对时间的支持，可以使用millisec/milliseconds、microsec/microseconds 等时间长度表示超过的时间，或者用 ptime 表示某个确定的时间点。

```
this_thread::sleep(posix_time::seconds(2));   //睡眠2秒钟
```

thread 库提供了一个自由函数 get_system_time()，它调用 microsec_clock 类方便地获得当前的UTC时间值。

### 互斥量

thread提供了七种互斥量类型（实际是五种），分别是：

* 独占式互斥量

  * mutex: 独占式的互斥量，是最简单最常用的一种互斥量类型

  * try_mutex: 它是mutex的同义词，为了与兼容以前的版本而提供

  * timed_mutex: 它也是独占式的互斥量，但提供超时锁定功能

* 递归式互斥量

  * recursive_mutex: 递归式互斥量，可以多次锁定，相应地也要多次解锁

  * recursive_try_mutex: 它是recursive_mutex 的同义词，为了与兼容以前的版本而提供

  * recursive_timed_mutex: 它也是递归式互斥量，基本功能同recursive_mutex, 但提供超时锁定功能

* 共享式互斥量：

  * shared_mutex: multiple-reader/single-writer 型的共享互斥量(又称读写锁)。

这些互斥量除了互斥功能不同外基本接口都很接近。以下是相同的接口函数：

#### lock

`void lock();`   

线程阻塞等待直至获得互斥量的所有权（即锁定）。

#### try_lock

`bool try_lock();`

尝试锁定互斥量，如果成功返回 true ，否则 false ，非阻塞。

#### unlock

`void unlock();`

解除对互斥量的锁定。

#### timed_lock

```
bool timed_lock(system_time const & abs_time);

template<typename TimeDuration>
bool timed_lock(TimeDuration const & relative_time);
```

只属于 timed_mutex 和 recursive_timed_mutex，阻塞等待一定时间试图锁定互斥量，如果时间到还未锁定则返回false。

#### 互斥量用法

```
mutex mu;
try
{
  mu.lock();           //锁定

  ···

  mu.unlock();         //解锁             
}
catch (···)
{
  mu.unlock();         //防止意外退出而导致死锁
}
```

#### lock_guard

thread 库提供了一系列的 RAII 型的 lock_guard 类，其在构造时锁定互斥量，在析构时自动解锁，从而保证了互斥量的正确操作，以避免遗忘解锁。

mutex 子空间定义了 scoped_lock 和 scoped_try_lock 两种类：

```
{
mutex mu;
mutex::scoped_lock lock(mu);

···

}
```

### 线程

thread 类是 thread 库的核心类，负责启动和管理线程对象，在概念和操作上都与 POSIX 线程很相似。

#### 创建并启动线程

```
void printing(atom_int& x, const string& str);

int main()
{
  atom_int x;

  thread t1(printing, ref(x), "hello");
  thread t2(printing, ref(x), "boost");

  this_thread::sleep(posix_time::seconds(2));
}
```

* 在创建了一个 thread 对象后，线程就立刻开始执行。

* 在传递参数时，thread 使用的是参数的拷贝，如果希望传递引用值，则需要使用 ref 库进行包装，而且必须保证对象在线程执行期间一直存在，否则会引发未定义行为。

* 在线程启动后，主线程必须等待其他程结束，否则会因为主线程结束，而导致其他线程一并结束。

#### 等待线程结束

##### join & timed_join

```
t1.timed_join(posix_time::seconds(1));   //最多等待1s
t2.join();              //等待t2线程结束再返回
```
* joinable() ：判断是否标识了一个可执行的线程体。

* join() ：一直阻塞等待，直到线程结束。

* timed_join() ：阻塞等待线程结束，超过一段时间后，不管线程结束与否都将返回。

#### 使用 bind 和 function 绑定函数

```
thread t3(bind(printing, ref(x), "thread"));   //启动线程
function<void()> f = bind(printing, 5, "mutex");
thread(f);
```

#### 操作线程

通常情况下一个非空的thread对象唯一地标识了一个可执行的线程体，是 joinable() 的，成员函数 get_id() 可以返回线程 id 对象。

thread 类还提供了三个很有用的 **静态成员函数** ：

* yield() ：指示当前线程放弃时间片，允许其他的线程运行。

* sleep() ：让线程睡眠等待一小段时间，参数时system_time UTC时间，而不是时间长度。

* hardware_concurrency() ：可以获得硬件系统可并行的线程数量，即CPU数量或者CPU内核数量，如果无法获取信息则返回 0。

为了方便，thread 库还在 this_thread 子空间中提供了三个自由函数 ：get_id()、yield()、sleep()，他们与 thread 类的上述三个静态函数功能相同。


#### 中断线程

thread 的成员函数 interrupt() 允许正在执行的线程被中断，被中断的线程会抛出一个 thread_interrupted 异常，它是一个空类。thread_interrupted 异常应该在线程执行函数里捕获并处理，如果线程不处理这个异常，那么默认的动作是中止线程。

##### 线程中断点

线程不是在任意时刻都可以被中断的，thread 库预定义了若干个线程的中断点，只有当线程执行到中断点的时候才能被中断，一个线程可以拥有任意多个中断点。

thread 库预定义了共 9 个中断点，他们都是函数，如下：

```
boost::thread::join()
boost::thread::timed_join()
boost::condition_variable::wait()
boost::condition_variable::timed_wait()
boost::condition_variable_any::wait()
boost::condition_variable_any::timed_wait()
boost::thread::sleep()
boost::this_thread::sleep()
boost::this_thread::interruption_point()
```

前八个中断点都是以某种形式的等待函数，表明线程在阻塞等待的时候可以被中断。而最后一个位于子命名空间 this_thread 的 interruption_point() 则是一个特殊的中断点函数，它并不等待，只是起到一个标签的作用，表示线程执行到这个函数所在的语句就可以被中断。

##### 启用/禁用线程中断

thread 库在子命名空间 this_thread 提供了一组函数和类来共同完成线程的中断启用和禁用：

* interruption_enable() ：检测当前线程是否允许中断

* interrupt_requested() ：检测当前线程是否被要求中断

* class disable_interruption ：一个 RAII 类型对象，它在构造时关闭线程的中断，析构时自动恢复线程的中断状态，在其生命期类线程始终不可中断，除非使用 restore_interruption 对象。

* class restore_interruption ：只能在 disable_interruption 的作用域内使用，它在构造时临时打开线程的中断状态，在析构时又关闭中断状态。
