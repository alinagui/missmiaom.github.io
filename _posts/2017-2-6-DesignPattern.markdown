---
layout:     post
title:      "Design Pattern 读书笔记"
subtitle:   " \"Design Pattern\""
date:       2017-2-6
author:     "Leiym"
header-img: "img/post-bg-2015.jpg"
tags:
    - Design Pattern
    - 读书笔记
---

### 创建型模式

#### Singleton Pattern

**单例模式(Singleton Pattern)** ：单例模式确保某一个类只有一个实例，而且自行实例化并向整个系统提供这个实例，这个类称为单例类，它提供全局访问的方法

模式要点：

1. 某个类只能有一个实例。

2. 它必须自行创建这个实例。

3. 它必须自行向整个系统提供这个实例。

实现方法：

1. 私有化构造函数，便于在内部控制创建实例的数目。

2. 定义静态变量（因需在静态方法中使用）存储创建的实例。

3. 定义静态方法（即类方法）来为客户端提供类实例。

4. 提供实例的静态方法中，先判断变量是否为空，如果为空则调用构造函数创建实例并赋值给变量，否则直接使用。

[C++实现代码]()

模式UML图：

<img src="http://leiym.com/img/in-post/post-designpattern/singleton.png"/>

#### Simple Factory Pattern

**简单工厂模式(Simple Factory Pattern)**：又称为静态工厂方法(Static Factory Method)模式。它根据自变量的不同返回不同的类的实例。简单工厂模式专门定义一个类来负责创建其它类的实例，被创建的实例通常都具有共同的父类。

模式参与者：

* Factory：工厂角色，工厂类在客户端的直接控制下（Create方法）创建产品对象。

* Product：抽象产品角色，定义简单工厂创建的对象的父类或它们共同拥有的接口。可以是一个类、抽象类或接口。

* ConcreteProduct：具体产品角色，定义工厂具体加工出的对象。

[C++实现代码]()

模式UML图：

<img src="http://leiym.com/img/in-post/post-designpattern/simplefactory.png"/>

#### Factory Method Pattern

**工厂方法模式(Factory Method Pattern)**：工厂方法模式又简称为工厂模式，也叫虚拟构造器(Virtual Constructor)模式或者多态模式。在工厂方法模式中，父类负责定义创建对象的公共接口，而子类则负责生成具体的对象，这样做的目的是将类的实例化操作延迟到子类中完成，即由子类来决定究竟应该实例化(创建) 哪一个类。

模式参与者：

* Product：抽象产品

* ConcreteProduct：具体产品

* Factory：抽象工厂

* ConcreteFactory：具体工厂

[C++实现代码]()

模式UML图：

<img src="http://leiym.com/img/in-post/post-designpattern/factorymethod.png"/>

#### Prototype Pattern

**原型模式(Prototype Pattern)**：它是一种对象创建型模式，用原型实例指定创建对象的种类，并且通过拷贝这些原型创建新的对象。原型模式允许一个对象再创建另外一个可定制的对象，无需知道任何创建的细节。

模式参与者：

* Prototype：抽象原型类

* ConcretePrototype：具体原型类

* Client：客户

[C++实现代码]()

模式UML图：

<img src="http://leiym.com/img/in-post/post-designpattern/prorotype.png"/>

#### Builder Pattern

**建造者模式(Builder Pattern)**：将一个复杂对象的构建与它的表示分离，使得同样的构建过程可以创建不同的表示。建造者模式是一步一步创建一个复杂的对象，它允许用户只通过指定复杂对象的类型和内容就可以构建它们，用户不需要知道内部的具体构建细节。建造者模式属于对象创建型模式。

模式参与者：

* Builder：抽象建造者

* ConcreteBuilder：具体建造者

* Director：指挥者

* Product：产品角色

模式适用情况：

* 需要生成的产品对象有复杂的内部结构。

* 需要生成的产品对象的属性相互依赖，建造者模式可以强迫生成顺序。

* 在对象创建过程中会使用到系统中的一些其他对象，这些对象在产品对象的创建过程中不易得到。


[C++实现代码]()

模式UML图：

<img src="http://leiym.com/img/in-post/post-designpattern/builder.png"/>

####

**抽象工厂模式(Abstract Factory Pattern)**：提供一个创建一系列相关或相互依赖对象的接口，而无需指定它们具体的类。

模式参与者：

* AbstractFactory：抽象工厂

* ConcreteFactory：具体工厂

* AbstractProduct：抽象产品

* Product：具体产品

模式适用情况：

* 一个系统不应当依赖于产品类实例如何被创建、组合和表达的细节，这对于所有形态的工厂模式都很重要。

* 系统有多于一个的产品族，而客户端只消费其中某一产品族。

* 系统提供一个产品类的库，所有的产品以同样的接口出现，从而使客户端不依赖于实现。


一个系统不应当依赖于产品类实例如何被创建、组合和表达的细节，这对于所有形态的工厂模式都很重要。
系统有多于一个的产品族，而客户端只消费其中某一产品族。
系统提供一个产品类的库，所有的产品以同样的接口出现，从而使客户端不依赖于实现。


[C++实现代码]()

模式UML图：

<img src="http://leiym.com/img/in-post/post-designpattern/abstractfactory.png"/>

### 结构型模式

#### Adapter Pattern

**适配器模式(Adapter Pattern)**： 将一个接口转换成客户希望的另一个接口，适配器模式使接口不兼容的那些类可以一起工作，其别名为包装器(Wrapper)。适配器模式既可以作为类结构型模式，也可以作为对象结构型模式。

模式参与者：

* Target：目标抽象类

* Adapter：适配器类

* Adaptee：适配者类（被适配）

模式适用情况：

* 系统需要使用的类的接口不符合系统的要求。

* 想要建立一个可以重复使用的类，用于与一些彼此之间没有太大关联的一些类，包括一些可能在将来引进的类一起工作。这些源类不一定有很复杂的接口。

* （对象适配器而言）在设计里，需要改变多个已有子类的接口，如果使用类的适配器模式，就要针对每一个子类做一个适配器，而这不太实际。

**注意：** 适配器模式分为类适配方法和对象适配方法。在类适配方法中，是通过类的继承来实现的，同时也具有接口的所有行为，这些就违背了面向对象设计原则中的类的单一职责原则，而对象适配器则是通过对象组合的方式来实现的，则符合面向对象的精神，所以推荐用对象适配的模式。

[C++实现代码]()

模式UML图：

<img src="http://leiym.com/img/in-post/post-designpattern/adapter.png"/>

#### Bridge Pattern

**桥接模式(Bridge Pattern)**：将抽象部分与它的实现部分分离，使它们都可以独立地变化。它是一种对象结构型模式，又称为柄体(Handle and Body)模式或接口(Interface)模式。

模式参与者：

* Abstraction：抽象类

* RefinedAbstraction：扩充抽象类

* Implementor：实现类接口

* ConcreteImplementor：具体实现类

在桥接模式中不仅Implementor具有变化(ConcreateImplementior)，而且Abstraction也可以发生变化(RefinedAbstraction)，这是一个多对多的关系，而且两者的变化是完全独立的。RefinedAbstraction与ConcreateImplementior之间松散耦合，它们仅仅通过Abstraction与Implementor之间的聚合关系联系起来。

模式适用情况：

* 如果一个系统需要在构件的抽象化角色和具体化角色之间增加更多的灵活性， 避免在两个层次之间建立静态的联系。

* 设计要求实现化角色的任何改变不应当影响客户端，或者说实现化角色的改变对客户端是完全透明的。


[C++实现代码]()

模式UML图：

<img src="http://leiym.com/img/in-post/post-designpattern/bridge.png"/>

#### Decorator Pattern

**装饰模式(Decorator Pattern)**：动态地给一个对象增加一些额外的职责 （Responsibility），就增加对象功能来说，装饰模式比生成子类实现 更为灵活。其别名为包装器(Wrapper)。装饰模式是一种对象结构型模式。

模式参与者：

* Component：组件

* ConcreteComponent：具体组件

* Decorator：抽象装饰类

* ConcreteDecorator：具体装饰类

Decorator 和 Component 的聚合关系变现为，Decorator 类中包含一个 Component 类的指针，利用这个指针 Decorator 便可以包含一份 ConcreteComponent 实现装饰。

Decorator 继承 Component 导致了 ConcreteDecorator 可以相互间嵌套包含，实现了非常方便的无限制嵌套装饰。

模式适用情况：

* 在不影响其它对象的情况下，以动态、透明的方式给单个对象添加职责。需要动态地给一个对象增加功能，这些功能可以再动态地被撤销。

* 当不能采用生成子类的方法进行扩充时。一种情况是，可能有大量独立的扩展，每一种组合将产生大量的子类，使得子类数目呈爆炸性增长。另一种情况可能是因为类定义不能继承(final类)，或类不能用于生成子类。


[C++实现代码]()

模式UML图：

<img src="http://leiym.com/img/in-post/post-designpattern/decorator.png"/>

#### Composite Pattern

**组合模式 (Composite Pattern)**：组合多个对象形成树形结构以表示“整体—部分”的结构层次。组合模式对单个对象（即叶子对象）和组合对象（即容器对象）的使用具有一致性。组合模式又可以称为“整体—部分”(Part-Whole)模式，属于对象的结构模式，它将对象组织到树结构中，可以用来描述整体与部分的关系。

模式参与者：

* Component：抽象构件（接口或抽象类）

* Leaf：叶子构件

* Composite：容器构件

模式适用情况：

* 需要表示一个对象整体或部分层次。

* 想让客户能够忽略不同对象层次的变化。

* 对象的结构是动态的并且复杂程度不一样，但客户需要一致地处理它们。

[C++实现代码]()

模式UML图：

<img src="http://leiym.com/img/in-post/post-designpattern/composite.png"/>

#### Flyweight Pattern

**享元模式(Flyweight)**：运用共享技术有效地支持大量细粒度的对象。 系统只使用少量的对象，而这些对象都很近，状态变化很小，对象使用次数增多。享元模式是一种对象结构型模式。

模式参与者：

* Flyweight：抽象享元类

* ConcreteFlyweight：具体享元类

* UnsharedConcreteFlyweight：非共享具体享元类

* FlyweightFactory：享元工厂类

模式适用情况：

* 一个系统有大量的对象，造成耗费大量的内存。

* 这些对象的状态中的大部分都可以外部化。

* 这些对象可以按照内部状态分成很多的组，当把外部对象从对象中剔除时，每一个组都可以用相对较少的共享对象代替。

* 软件系统不依赖于这些对象的身份，换言之，这些对象可以是不可分辨的。

[C++实现代码]()

模式UML图：

<img src="http://leiym.com/img/in-post/post-designpattern/flyweight.png"/>

#### Facade Pattern

**外观模式(Facade)**：外部与一个子系统的通信必须通过一个统一的外观对象进行，为子系统中的一组接口提供一个一致的界面，外观模式定义了一个高层接口，这个接口使得这一子系统更加容易使用。外观模式是对象的结构模式。

模式参与者：

* Facade：外观角色

* SubSystem：子系统角色

模式适用情况：

* 当要为一个复杂子系统提供一个简单接口时。这个接口对大多数用户来说已经足够好；那些需要更多可定制性的用户可以越过Facade层。

* 子系统相对独立——外界只需黑箱操作即可。例如利息计算。

* 预防操作人员带来的风险扩散。

[C++实现代码]()

模式UML图：

<img src="http://leiym.com/img/in-post/post-designpattern/facade.png"/>


#### Proxy Pattern

**代理模式(Proxy Pattern)**：给某一个对象提供一个代理，并由代理对象控制对原对象的引用。代理模式的英文叫做Proxy或Surrogate。代理模式是一种对象结构型模式。

模式参与者：

* Subject：抽象主题角色

* Proxy：代理主题角色

* RealSubject：真实主题角色

[C++实现代码]()

模式UML图：

<img src="http://leiym.com/img/in-post/post-designpattern/proxy.png"/>

### 行为型模式

#### State Pattern

**状态模式(State Pattern)**：允许一个对象在其内部状态改变时改变它的行为，对象看起来似乎修改了它的类。其别名为状态对象(Objects for States)。状态模式是一种对象行为型模式。

模式参与者：

* Context：环境类

* State：抽象状态类

* ConcreteState：具体状态类

模式适用情况：

* 对象的行为依赖于它的状态（属性）并且它必须可以根据它的状态改变而改变它的相关行为。

* 操作的很多部分都带有与对象状态有关的大量条件语句，大量条件语句的出现，会导致代码的可维护性和灵活性变差，不能方便地增加删除状态，使客户类与类库之间的耦合增强。

[C++实现代码]()

模式UML图：

<img src="http://leiym.com/img/in-post/post-designpattern/state.png"/>

#### Observer Pattern

**观察者模式(Observer Pattern)**：定义对象间的一种一对多依赖关系，使得每当一个对象状态发生改变时，其相关依赖对象皆得到通知并被自动更新。观察者模式又叫做发布-订阅（Publish/Subscribe）模式、模型-视图（Model/View）模式、源-监听器（Source/Listener）模式或从属者（Dependents）模式。观察者模式是一种对象行为型模式。

模式参与者：

* Subject：目标（被观察对象）

* ConcreteSubject：具体目标

* Observer：观察者

* ConcreteObserver：具体观察者

模式适用情况：

* 一个抽象模型有两个方面，其中一个方面依赖于另一个方面。将这些方面封装在独立的对象中使它们可以各自独立地改变和复用。

* 一个对象的改变将导致其他一个或多个对象也发生改变，而不知道具体有多少对象将发生改变，可以降低对象之间的耦合度。

[C++实现代码]()

模式UML图：

<img src="http://leiym.com/img/in-post/post-designpattern/observer.png"/>

#### Mediator Pattern

**中介者模式(Mediator Pattern)**：用一个中介对象来封装一系列的对象交互，中介者使各对象不需要显式地相互引用，从而使其耦合松散，而且可以独立地改变它们之间的交互。中介者模式又称为调停者模式，它是一种对象行为型模式。中介者模式是迪米特法则的一个典型应用。

模式参与者：

* Mediator: 抽象中介者

* ConcreteMediator: 具体中介者

* Colleague: 抽象同事类

* ConcreteColleague: 具体同事类

模式适用情况：

* 系统中对象之间存在复杂的引用关系，产生的相互依赖关系结构混乱且难以理解。

* 一个对象由于引用了其他很多对象并且直接和这些对象通信，导致难以复用该对象。

* 想通过一个中间类来封装多个类中的行为，而又不想生成太多的子类。可以通过引入中介者类来实现，在中介者中定义对象交互的公共行为，如果需要改变行为则可以增加新的中介者类。

[C++实现代码]()

模式UML图：

<img src="http://leiym.com/img/in-post/post-designpattern/mediator.png"/>

#### Chain of Responsibility Pattern

**职责链模式(Chain of Responsibility Pattern)**：避免请求发送者与接收者耦合在一起，让多个对象都有可能接收请求，将这些对象连接成一条链，并且沿着这条链传递请求，直到有对象处理它为止。职责链模式又称为责任链模式，它是一种对象行为型模式。

模式参与者：

* Handler: 抽象处理者

* ConcreteHandler: 具体处理者

模式适用情况：

* 有多个对象可以处理同一个请求，具体哪个对象处理该请求由运行时刻自动确定。

* 在不明确指定接收者的情况下，向多个对象中的一个提交一个请求。

* 可动态指定一组对象处理请求
