---
layout:     post
title:      "Effective C++ 读书笔记"
subtitle:   " \"bat\""
date:       2016-12-19
author:     "Leiym"
header-img: "img/post-bg-2015.jpg"
tags:
    - C++
    - 读书笔记
---

### 让自己习惯C++

#### 尽量以 const，enum，inline 替换 #define

使用 `#define` 定义常量时（`#define PI 3.14`），编译器在预处理阶段即将常量名（PI）替换为常量值（3.14），而不是将常量名（PI）放入符号表中。当运用此常量如果获得一个编译错误，错误信息中提到的是常量值（3.14）而不是常量名（PI），这会浪费因为追踪它而花费的时间。而使用 `const` 定义常量时则不会出现上述情况。

当需要创建一个 class 专属常量时，可以使用私有枚举类型来实现，因为枚举类型的数值可以充当 `int` 而被使用。

```
class GamePlayer {
private:
  enum { NumTurns = 5 };   //"the enum hack" —— 令 NumTurns 成为 5 的一个记号名称
  int scores[NumTurns];    //合法
};
```

函数宏因为是直接替换参数，所以很容易发生不安全的行为：

```
#define CALL_WITH_MAX(a, b) f( (a) > (b) ? (a) : (b) )   //以a和b的较大值调用f

int a = 5, b = 0;
CALL_WITH_MAX(++a, b);             //a被累加两次
CALL_WITH_MAX(++a, b+10);          //a被累加一次，a的递增次数不确定！
```

谨记：

* 对于单纯常量，最好以 `const` 对象或 `enum`s 替换 `#define`s 。

* 对于形似函数的宏（macros），最好改用 `inline` 函数替换 `#define` 。

#### 尽可能使用 const

```
char* p = "Hello";              //non-const pointer, non-const data
const char* p = "Hello"         //non-const pointer, const data
char* const p = "Hello"         //const pointer, non-const data
const char* const p = "Hello"   //const pointer, const data
```

#### 确定对象被使用前已先被初始化

谨记：

* 为内置型对象进行手工初始化，因为C++不保证初始化它们。

* 构造函数最好使用成员初值列，而不要在构造函数本体内使用赋值操作。初值列列出的成员变量，其排列次序应该和它们在class中的声明次序相同。

* 为免除“跨编译单元之初始化次序”问题，请以 local static 对象替换 non-local static对象。

### 构造、析构、赋值运算

#### C++隐式声明的函数

当一个类没有显式地声明 *构造函数*，*copy构造函数*，*copy assignment操作符* 和 *析构函数* 任意一个或几个时，编译器会隐式地为其生成相应 **公有的** **非虚** 函数。

那么，编译器所隐式声明的函数做了什么呢？

 *default构造函数* 和 *析构函数* 依次调用 base class 和 non-static 成员变量的构造函数和析构函数。而 *copy构造函数* 和 *copy assignment操作符* 只是单纯地将对象的每一个 non-static 成员变量拷贝到目标对象。

 **例外**：当类中含有引用成员时，即使没有显式地声明 *copy assignment操作符*，编译器也不会为其隐式地声明该函数。因为 *copy assignment操作符* 会将一个引用值赋值给一个已经初始化的引用，这是非法的！（C++禁止引用改指向不同对象）。

 另外，当一个类的 *copy assignment操作符* 声明为 private 时，编译器不会为其所有子类隐式声明 *copy assignment操作符* ，因为编译器无权调用基类的 *copy assignment操作符*。

#### 拒绝C++隐式声明的拷贝构造函数和赋值操作符

当一个类的设计初衷就是禁止拷贝的话，C++隐式声明的 *copy构造函数*，*copy assignment操作符* 函数将会违背初衷。

解决办法：

1. 显式地只声明 *copy构造函数*，*copy assignment操作符* 为 private ，而不做任何实现。

    ```
    class NonCopy{
    public:
      ...
    private:
      ...
      NonCopy(const NonCopy&);                 //只有声明
      NonCopy& operator=(const NonCopy&);      //外部无法调用，内部函数调用时，会因没有实现而在编译期报错
    }
    ```

2. 私有继承满足第一点的基类。

    ```
    class Example：private NonCopy{
      ...                                       //无需做任何动作
    }
    ```

#### 为多态基类声明 virtual 析构函数

C++ 允许使用基类指针指向派生类，并通过基类指针访问派生类中的函数，典型的例子便是factory（工厂）模式。如果指针由 `new` 在 heap 上动态生成的话，必须使用 `delete` 去释放动态生成的空间，否则很可能造成内存泄漏。

派生类被释放时，总是先调用最下层派生类的析构函数，然后从下往上依次调用各个基类的析构函数。使用指向派生类的 **基类指针** ，当其被释放时，首先调用的是基类的析构函数。所以，如果基类的析构函数是 **非虚** （non-virtual）的话，**子类的析构函数将不会被调用，即子类的成员将不会被释放从而导致内存泄漏。**

谨记：

* 带有多态性质的基类应该声明一个 virtual 析构函数。如果一个类带有任何 virtual 函数，它就应该拥有一个 virtual 析构函数。

* 相反，如果一个类的设计目的不是作为基类，那么，它就不应该声明 virtual 析构函数。

####
