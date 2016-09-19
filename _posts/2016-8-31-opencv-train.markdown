---
layout:     post
title:      "openCV学习笔记——持续更新"
subtitle:   " \"opencv\""
date:       2016-08-31
author:     "Leiym"
header-img: "img/post-bg-2015.jpg"
tags:
    - opencv
---

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [opencv2 结构](#opencv2-结构)
- [操作图片](#操作图片)
	- [显示图像](#显示图像)
	- [写入图像](#写入图像)
	- [创建图像](#创建图像)
	- [复制图像](#复制图像)
	- [操作像素点](#操作像素点)
	- [设定一幅图像的ROI区域](#设定一幅图像的roi区域)
	- [图像算术运算](#图像算术运算)
	- [颜色空间转换](#颜色空间转换)
	- [计算直方图](#计算直方图)
	- [阈值化](#阈值化)
	- [线性混合](#线性混合)
	- [轨迹条](#轨迹条)
- [形态学基本操作](#形态学基本操作)
	- [腐蚀运算](#腐蚀运算)
	- [膨胀运算](#膨胀运算)
	- [开运算，闭运算，形态学梯度](#开运算闭运算形态学梯度)
- [图像滤波](#图像滤波)
	- [线性滤波](#线性滤波)
		- [均值滤波](#均值滤波)
		- [方框滤波](#方框滤波)
		- [高斯滤波](#高斯滤波)
		- [非线性滤波](#非线性滤波)
		- [中值滤波](#中值滤波)
		- [双边滤波](#双边滤波)
		- [边缘检测](#边缘检测)
- [边缘检测步骤：](#边缘检测步骤)
	- [边缘检测评价标准](#边缘检测评价标准)
	- [canny](#canny)
	- [sobel](#sobel)
	- [Laplace](#laplace)
	- [scharr](#scharr)

<!-- /TOC -->

### opencv2 结构
*core*        :定义了基本数据结构，包括最重要的Mat和一些其他的模块

*imgproc*     :该模块包括了线性和非线性的图像滤波，图像的几何变换，颜色空间转换，直方图处理等等

*video*       :该模块包括运动估计，背景分离，对象跟踪

*calib3d*     :基本的多视角几何算法，单个立体摄像头标定，物体姿态估计，立体相似性算法，3D信息的重建

*features2d*  :显著特征检测，描述，特征匹配

*objdetect*   :物体检测和预定义好的分类器实例（比如人脸，眼睛，面部，人，车辆等等）

*highgui*     :视频捕捉、图像和视频的编码解码、图形交互界面的接口

*gpu*         :利用GPU对OpenCV模块进行加速算法

*ml*          :机器学习模块（SVM，决策树，Boosting等等）

*flann*       :Fast Library for Approximate Nearest Neighbors（FLANN）算法库

*legacy*      :一些已经废弃的代码库，保留下来作为向下兼容

[组件结构](http://blog.csdn.net/poem_qianmo/article/details/19925819)

### 操作图片

#### 显示图像
`void imshow(const string& winname, InputArray mat);`

#### 写入图像
`imwrite("output.jpg",image);`

#### 创建图像
`cv::Mat image(240,320,CV_8U,cv::Scalar(100));`
大小320*240，可以把 CV_8U 换成 CV_8U3来创建一个三通道的彩色图像。或者用 CV_16U 创建无符号16位的。

#### 复制图像
`Mat newImage=Image;`
只复制头，如果对newImage进行修改或操作，则会直接影响Image图像。

`Image.copyTo(newImage);//方法一`
`Mat newImage=image.clone();//方法二`
复制图像，得到一个副本。

`newImage.create(Image.size(),Image.type());`
创建一个跟原图像大小相同的图像。

#### 操作像素点
`image.at<Vec3b>(i,j)[k]`
取出彩色图像中i行j列第k通道的颜色点

`image(i,j)`
重载operator()取图像上的点

`image.ptr<uchar>(i)`
取出图像中第i行数据的指针

`image.isContinuous()`
判断image是否连续，如果连续可将image看成一行

`MatIterator_<Vec3b> it = image.begin<Vec3b>();`
`Mat_<Vec3b>::iterator it = =image.begin<Vec3b>();`
声明迭代器，并让其指向image的首元素

`(*it)[0]`
取出迭代器 it 指向的元素的0通道的值

#### 设定一幅图像的ROI区域
`Mat imageROI=srcImage(Rect(500,250,logo.cols,logo.rows));`
方法一，利用Rect函数构造矩形区域，指定左上角坐标和长宽即可。

`Mat imageROI=srcImage(Range(250,250 + offset),Range(200,200 + offset));`
方法二，利用Range函数指定行或列的范围，Range是指从起始索引到终止索引（不包括终止索引）的一连段连续序列。

#### 图像算术运算
`vector<Mat> planes;`
`split(image,planes);`
将image分为三个通道图像存储在planes中

`merge(planes,result);`
将planes中三幅图像合为一个三通道图像

#### 颜色空间转换
`void cvtColor(InputArray src, OutputArray dst, int code, int dstCn=0 )`
* *src*:输入图像

* *dst*:输出图像

* *code*:颜色转换类型，比如：CV_BGR2Lab,CV_BGR2HSV,CV_HSV2BGR,CV_BGR2RGB

* *dstCn*:输出图像的通道号，如果默认为0，则表示按输入图像的通道数。

#### 计算直方图
`void calcHist(const Mat* images, int nimages, const int* channels, InputArray mask, OutputArray hist, int dims, const int* histSize, const float** ranges, bool uniform=true, bool accumulate=false )`
* _const Mat* images_：为输入图像的指针。

* *int nimages*：要计算直方图的图像的个数。此函数可以为多图像求直方图，通常情况下都只作用于单一图像，所以通常nimages=1。

* _const int* channels_：图像的通道，它是一个数组，如果是灰度图像则channels[1]={0};如果是彩色图像则channels[3]={0,1,2}；如果是只是求彩色图像第2个通道的直方图，则channels[1]={1};

* *IuputArray mask*：一个遮罩图像用于确定哪些点参与计算，默认情况我们都设置为一个空图像，即：Mat()。

* *OutArray hist*：计算得到的直方图

* *int dims*：得到的直方图的维数，灰度图像为1维，彩色图像为3维。

* *const int* histSize*：直方图横坐标的区间数。如果是10，则它会横坐标分为10份，然后统计每个区间的像素点总和。

* _const float** ranges_：一个二维数组，用来指出每个区间的范围。

后面两个参数都有默认值，uniform参数表明直方图是否等距，accumulate参数与多图像下直方图的显示与存储有关。

#### 阈值化
`double threshold(InputArray src, OutputArray dst, double thresh, double maxval, int type)`

* *src*：输入图像。

* *dst*：输出图像。

* *thresh*：阈值。

* *maxval*：当第五个参数阈值类型type取 THRESH_BINARY 或THRESH_BINARY_INV阈值类型时的最大值.

* *itype*：阈值化类型。

```
0: THRESH_BINARY      当前点值大于阈值时，取Maxval，否则设置为0
1: THRESH_BINARY_INV  当前点值大于阈值时，设置为0，否则设置为Maxval
2: THRESH_TRUNC       当前点值大于阈值时，设置为阈值，否则不改变
3: THRESH_TOZERO      当前点值大于阈值时，不改变，否则设置为0
4: THRESH_TOZERO_INV  当前点值大于阈值时，设置为0，否则不改变
```

#### 线性混合
`void addWeighted(InputArray src1, double alpha, InputArray src2, double beta, double gamma, OutputArray dst, int dtype=-1);`
计算两个Mat的加权和。  
* *src1*   表示需要加权的第一个数组。

* *alpha*  表示第一个数组的权重

* *src2*   表示第二个数组，它需要和第一个数组拥有相同的尺寸和通道数。

* *beta*   表示第二个数组的权重。

* *dst*    输出的数组，它和输入的两个数组拥有相同的尺寸和通道数。

* *gamma*  一个加到权重总和上的标量值。

* *dtype*  输出阵列的可选深度，有默认值-1。;当两个输入数组具有相同的深度时，这个参数设置为-1（默认值），即等同于src1.depth（）。

__dst = src1[I]*alpha+ src2[I]*beta + gamma;__  
其中的I，是多维数组元素的索引值。而且，在遇到多通道数组的时候，每个通道都需要独立地进行处理。

#### 轨迹条
`int createTrackbar(conststring& trackbarname, conststring& winname,int* value, int count, TrackbarCallback onChange=0,void* userdata=0); `

* *trackbarname*  ，表示轨迹条的名字，用来代表创建的轨迹条。

* *winname*       ，填窗口的名字，表示这个轨迹条会依附到哪个窗口上，即对应namedWindow（）创建窗口时填的某一个窗口名。

* *value*         ，一个指向整型的指针，表示滑块的位置。并且在创建时，滑块的初始位置就是该变量当前的值。

* *count*         ，表示滑块可以达到的最大位置的值。PS:滑块最小的位置的值始终为0。

* *onChange*      ，首先注意有默认值0。这是一个指向回调函数的指针，每次滑块位置改变时，这个函数都会进行回调。并且这个函数的原型必须为void XXXX(int,void*);其中第一个参数是轨迹条的位置，第二个参数是用户数据（看下面的第六个参数）。如果回调是NULL指针，表示没有回调函数的调用，仅第三个参数value有变化。

* *userdata*      ，也有默认值0。这个参数是用户传给回调函数的数据，用来处理轨迹条事件。如果使用的第三个参数value实参是全局变量的话，完全可以不去管这个userdata参数。


### 形态学基本操作

* *腐蚀运算* 的作用是消除物体的边界点，使目标缩小，腐蚀操作会消除小且无意义的物体，使边界向内部收缩。

* *膨胀运算* 的作用是使目标增大，填充物体内细小的空洞，并且平滑物体的边界，使边界向外部扩张。

* *开运算* 是先腐蚀后膨胀的过程，可以消除图像上细小的噪声，并平滑物体的边界

* *闭运算* 是先膨胀后腐蚀的过程，可以填充物体内细小的空洞，并平滑物体边界

* *形态学梯度* 膨胀图与腐蚀图之差，可以将边缘突出

##### 腐蚀运算
`void erode(InputArray src, OutputArray dst, InputArray kernel, Point anchor=Point(-1,-1),int iterations=1, int borderType=BORDER_CONSTANT,const Scalar& borderValue=morphologyDefaultBorderValue());`
* *src*：输入图像，很多场合下我们使用的是二值图像，当然灰度图像也可以。

* *dst*：输出图像，格式和输入图像一致。

* *kernel*：定义的结构元素。

* *anchor*：结构元素的中心，如果是默认参数(-1,-1)，程序会自动将其设置为结构元素的中心。

* *iterations*：迭代次数，我们可以选择对图像进行多次形态学运算。
后面两个参数是边界类型，由于要处理领域问题，所以图像需要扩充边界。一般情况下使用默认即可。

##### 膨胀运算
`void dilate(InputArray src, OutputArray dst, InputArray kernel, Point anchor=Point(-1,-1),int iterations=1, int borderType=BORDER_CONSTANT,const Scalar& borderValue=morphologyDefaultBorderValue());`

* 参数含义与腐蚀运算完全一致。

##### 开运算，闭运算，形态学梯度
`void morphologyEx(InputArray src, OutputArray dst, int op, InputArray kernel, Point anchor=Point(-1,-1), int iterations=1, int borderType=BORDER_CONSTANT, const Scalar& borderValue=morphologyDefaultBorderValue());`

*op*：
* MORPH_OPEN：开运算（Opening operation）

* MORPH_CLOSE：闭运算（Closing operation）

* MORPH_GRADIENT：形态学梯度（Morphological gradient）

* MORPH_TOPHAT：“顶帽”（“Top hat”）

* MORPH_BLACKHAT：“黑帽”（“Black hat“）

其余参数含义与腐蚀运算完全一致。

`getStructuringElement` 返回指定形状和尺寸的结构元素（内核矩阵）
第一个参数指定形状：

* 矩形: MORPH_RECT

* 交叉形: MORPH_CROSS

* 椭圆形: MORPH_ELLIPSE

第二个第三个参数分别是内核的尺寸以及锚点的位置。锚点位置一般默认为（-1，-1）即中心。

`Mat element = getStructuringElement(MORPH_RECT, Size(x, y), Point(x, y));//获取自定义核`

### 图像滤波
图像的 *频率* 是表征图像中灰度变化剧烈程度的指标，是灰度在平面空间上的梯度。
图像的主要成分是 *低频信息*，它形成了图像的基本灰度等级，对图像结构的决定作用较小；_中频信息_ 决定了图像的基本结构，形成了图像的主要边缘结构；_高频信息_ 形成了图像的边缘和细节，是在中频信息上对图像内容的进一步强化。

#### 线性滤波
* 线性滤波器：线性滤波器经常用于剔除输入信号中不想要的频率或者从许多频率中选择一个想要的频率。

* 邻域算子（局部算子）是利用给定像素周围的像素值的决定此像素的最终输出值的一种算子。

##### 均值滤波
均值滤波是典型的线性滤波算法，主要方法为邻域平均法，即用一片图像区域的各个像素的均值来代替原图像中的各个像素值。

缺陷：它不能很好地保护图像细节，在图像去噪的同时也破坏了图像的细节部分，从而使图像变得模糊，不能很好地去除噪声点。

`void blur(InputArray src, OutputArraydst, Size ksize, Point anchor=Point(-1,-1), int borderType=BORDER_DEFAULT ) `
* *src*，输入图像，即源图像，填Mat类的对象即可。该函数对通道是独立处理的，且可以处理任意通道数的图片，待处理的图片深度应该为CV_8U, CV_16U, CV_16S, CV_32F 以及 CV_64F之一。

* *dst*，即目标图像，需要和源图片有一样的尺寸和类型。比如可以用Mat::Clone，以源图片为模板，来初始化得到如假包换的目标图。

* *ksize*，内核的大小。一般这样写Size( w,h )来表示内核的大小( 其中，w 为像素宽度， h为像素高度)。Size（3,3）就表示3x3的核大小，Size（5,5）就表示5x5的核大小

* *anchor*，表示锚点（即被平滑的那个点），注意他有默认值Point(-1,-1)。如果这个点坐标是负值的话，就表示取核的中心为锚点，所以默认值Point(-1,-1)表示这个锚点在核的中心。

* *borderType*，用于推断图像外部像素的某种边界模式。有默认值BORDER_DEFAULT，我们一般不去管它。

##### 方框滤波
`void boxFilter(InputArray src,OutputArray dst, int ddepth, Size ksize, Point anchor=Point(-1,-1), boolnormalize=true, int borderType=BORDER_DEFAULT )`
* *src*，输入图像，即源图像，填Mat类的对象即可。该函数对通道是独立处理的，且可以处理任意通道数的图片，待处理的图片深度应该为CV_8U, CV_16U, CV_16S, CV_32F 以及 CV_64F之一。

* *dst*，即目标图像，需要和源图片有一样的尺寸和类型。

* *ddepth*，输出图像的深度，-1代表使用原图深度，即src.depth()。

* *ksize*，内核的大小。一般这样写Size( w,h )来表示内核的大小( 其中，w 为像素宽度， h为像素高度)。Size（3,3）就表示3x3的核大小，Size（5,5）就表示5x5的核大小

* *anchor*，表示锚点（即被平滑的那个点），注意他有默认值Point(-1,-1)。如果这个点坐标是负值的话，就表示取核的中心为锚点，所以默认值Point(-1,-1)表示这个锚点在核的中心。

* *normalize*，默认值为true，一个标识符，表示内核是否被其区域归一化（normalized）了。

* *borderType*，用于推断图像外部像素的某种边界模式。有默认值BORDER_DEFAULT，我们一般不去管它。

当normalize=true的时候，方框滤波就变成了我们熟悉的均值滤波。也就是说，均值滤波是方框滤波归一化（normalized）后的特殊情况。而非归一化（Unnormalized）的方框滤波用于计算每个像素邻域内的积分特性，比如密集光流算法（dense optical flow algorithms）中用到的图像倒数的协方差矩阵（covariance matrices of image derivatives）

##### 高斯滤波
高斯滤波是一种线性平滑滤波，适用于消除高斯噪声。
`void GaussianBlur(InputArray src,OutputArray dst, Size ksize, double sigmaX, double sigmaY=0, intborderType=BORDER_DEFAULT )`
* *src*，输入图像，即源图像，填Mat类的对象即可。它可以是单独的任意通道数的图片，但需要注意，图片深度应该为CV_8U,CV_16U, CV_16S, CV_32F 以及 CV_64F之一。

* *dst*，即目标图像，需要和源图片有一样的尺寸和类型。比如可以用Mat::Clone，以源图片为模板，来初始化得到如假包换的目标图。

* *ksize*，其中ksize.width和ksize.height可以不同，但他们都必须为正数和奇数。或者，它们可以是零的，值由sigma计算而来。

* *sigmaX*，表示高斯核函数在X方向的的标准偏差。

* *sigmaY*，表示高斯核函数在Y方向的的标准偏差。若sigmaY为零，就将它设为sigmaX，如果sigmaX和sigmaY都是0，那么就由ksize.width和ksize.height计算出来。为了结果的正确性，最好是把Size，sigmaX和sigmaY全部指定到。

* *borderType*，用于推断图像外部像素的某种边界模式。有默认值BORDER_DEFAULT，我们一般不去管它。

#### 非线性滤波

##### 中值滤波
中值滤波，用像素点邻域灰度值的中值来代替该像素点的灰度值，该方法在去除脉冲噪声、椒盐噪声的同时又能保留图像边缘细节。

与均值滤波的比较：
* 优势：在均值滤波器中，由于噪声成分被放入平均计算中，所以输出受到了噪声的影响，但是在中值滤波器中，由于噪声成分很难选上，所以几乎不会影响到输出。因此同样用3x3区域进行处理，中值滤波消除的噪声能力更胜一筹。中值滤波无论是在消除噪声还是保存边缘方面都是一个不错的方法。

* 劣势：中值滤波花费的时间是均值滤波的5倍以上。

中值滤波在一定条件下，可以克服线性滤波器（如均值滤波等）所带来的图像细节模糊，而且对滤除脉冲干扰即图像扫描噪声最为有效。在实际运算过程中并不需要图像的统计特性，也给计算带来不少方便。但是对一些细节多，特别是线、尖顶等细节多的图像不宜采用中值滤波。

`void medianBlur(InputArray src,OutputArray dst, int ksize)`
* *src*，函数的输入参数，填1、3或者4通道的Mat类型的图像；当ksize为3或者5的时候，图像深度需为CV_8U，CV_16U，或CV_32F其中之一，而对于较大孔径尺寸的图片，它只能是CV_8U。

* *dst*，即目标图像，函数的输出参数，需要和源图片有一样的尺寸和类型。我们可以用Mat::Clone，以源图片为模板，来初始化得到如假包换的目标图。

* *ksize*，孔径的线性尺寸（aperture linear size），注意这个参数必须是大于1的奇数，比如：3，5，7，9 ...

##### 双边滤波
双边滤波，结合图像的空间邻近度和像素值相似度的一种折中处理，同时考虑空域信息和灰度相似性，达到保边去噪的目的。具有简单、非迭代、局部的特点。

双边滤波器的好处是可以做边缘保存（edge preserving），一般过去用的维纳滤波或者高斯滤波去降噪，都会较明显地模糊边缘，对于高频细节的保护效果并不明显。双边滤波比高斯滤波多了一个高斯方差sigma－d，它是基于空间分布的高斯滤波函数，所以在边缘附近，离的较远的像素不会太多影响到边缘上的像素值，这样就保证了边缘附近像素值的保存。但是由于保存了过多的高频信息，对于彩色图像里的高频噪声，双边滤波器不能够干净的滤掉，只能够对于低频信息进行较好的滤波。

`void bilateralFilter(InputArray src, OutputArraydst, int d, double sigmaColor, double sigmaSpace, int borderType=BORDER_DEFAULT)`
* *src*，输入图像，即源图像，需要为8位或者浮点型单通道、三通道的图像。

* *dst*，即目标图像，需要和源图片有一样的尺寸和类型。

* *d*，表示在过滤过程中每个像素邻域的直径。如果这个值我们设其为非正数，那么OpenCV会从第五个参数sigmaSpace来计算出它来。

* *sigmaColor*，颜色空间滤波器的sigma值。这个参数的值越大，就表明该像素邻域内有更宽广的颜色会被混合到一起，产生较大的半相等颜色区域。
sigmaSpace坐标空间中滤波器的sigma值，坐标空间的标注方差。他的数值越大，意味着越远的像素会相互影响，从而使更大的区域足够相似的颜色获取相同的颜色。当d>0，d指定了邻域大小且与sigmaSpace无关。否则，d正比于sigmaSpace。

* *borderType*，用于推断图像外部像素的某种边界模式。注意它有默认值BORDER_DEFAULT。

### 边缘检测

#### 边缘检测步骤：

* 滤波：边缘检测的算法主要是基于图像强度的一阶和二阶导数，但导数通常对噪声很敏感，因此必须采用滤波器来改善与噪声有关的边缘检测器的性能。常见的滤波方法主要有高斯滤波，即采用离散化的高斯函数产生一组归一化的高斯核，然后基于高斯核函数对图像灰度矩阵的每一点进行加权求和。

* 增强：增强边缘的基础是确定图像各点邻域强度的变化值。增强算法可以将图像灰度点邻域强度值有显著变化的点凸显出来。在具体编程实现时，可通过计算梯度幅值来确定。

* 检测：经过增强的图像，往往邻域中有很多点的梯度值比较大，而在特定的应用中，这些点并不是我们要找的边缘点，所以应该采用某种方法来对这些点进行取舍。实际工程中，常用的方法是通过阈值化方法来检测。

#### 边缘检测评价标准

* 低错误率: 标识出尽可能多的实际边缘，同时尽可能的减少噪声产生的误报。

* 高定位性: 标识出的边缘要与图像中的实际边缘尽可能接近。

* 最小响应: 图像中的边缘只能标识一次，并且可能存在的图像噪声不应标识为边缘。

#### canny

1. 消除噪声。一般情况下，使用高斯平滑滤波器卷积降噪。

2. 计算梯度幅值和方向。

3. 非极大值抑制。 这一步排除非边缘像素， 仅仅保留了一些细线条(候选边缘)。

4. 滞后阈值。最后一步，Canny 使用了滞后阈值，滞后阈值需要两个阈值(高阈值和低阈值)：

    * 如果某一像素位置的幅值超过 高 阈值, 该像素被保留为边缘像素。

    * 如果某一像素位置的幅值小于 低 阈值, 该像素被排除。

    * 如果某一像素位置的幅值在两个阈值之间,该像素仅仅在连接到一个高于 高 阈值的像素时被保留。

    tips：对于Canny函数的使用，推荐的高低阈值比在2:1到3:1之间。

`void Canny(InputArray image,OutputArray edges, double threshold1, double threshold2, int apertureSize=3,bool L2gradient=false )`

* *image*，输入图像，即源图像，填Mat类的对象即可，且需为单通道8位图像。

* *edges*，输出的边缘图，需要和源图片有一样的尺寸和类型。

* *threshold1*，第一个滞后性阈值。

* *threshold2*，第二个滞后性阈值。

* *apertureSize*，表示应用Sobel算子的孔径大小，其有默认值3。

* *L2gradient*，一个计算图像梯度幅值的标识，有默认值false。

**注意** ：阈值1和阈值2两者的小者用于边缘连接，而大者用来控制强边缘的初始段，推荐的高低阈值比在2:1到3:1之间。

#### sobel

Sobel 算子是一个主要用作边缘检测的离散微分算子。 它Sobel算子结合了高斯平滑和微分求导，用来计算图像灰度函数的近似梯度。在图像的任何一点使用此算子，将会产生对应的梯度矢量或是其法矢量。

`void Sobel (InputArray src, OutputArray dst, int ddepth, int dx, int dy, int ksize=3, double scale=1, double delta=0, int borderType=BORDER_DEFAULT );`

* *src*，为输入图像，填Mat类型即可。

* *dst*，即目标图像，函数的输出参数，需要和源图片有一样的尺寸和类型。

* *ddepth*，输出图像的深度。

* *dx*，x 方向上的差分阶数。

* *dy*，y方向上的差分阶数。

* *ksize*，有默认值3，表示Sobel核的大小;必须取1，3，5或7。

* *scale*，计算导数值时可选的缩放因子，默认值是1，表示默认情况下是没有应用缩放的。

#### Laplace

Laplacian 算子是n维欧几里德空间中的一个二阶微分算子，定义为梯度grad（）的散度div（）。因此如果f是二阶可微的实函数，则f的拉普拉斯算子定义为：

    1. f的拉普拉斯算子也是笛卡儿坐标系xi中的所有非混合二阶偏导数求和：

    2. 作为一个二阶微分算子，拉普拉斯算子把C函数映射到C函数，对于k ≥ 2。表达式(1)（或(2)）定义了一个算子Δ :C(R) → C(R)，或更一般地，定义了一个算子Δ : C(Ω) → C(Ω)，对于任何开集Ω。

根据图像处理的原理我们知道，二阶导数可以用来进行检测边缘 。 因为图像是 “二维”, 我们需要在两个方向进行求导。使用Laplacian算子将会使求导过程变得简单。

tips：让一幅图像减去它的Laplacian可以增强对比度。

`void Laplacian(InputArray src,OutputArray dst, int ddepth, int ksize=1, double scale=1, double delta=0, intborderType=BORDER_DEFAULT );`

* *image*，输入图像，即源图像，填Mat类的对象即可，且需为单通道8位图像。

* *edges*，输出的边缘图，需要和源图片有一样的尺寸和通道数。

* *ddept*，目标图像的深度。

* *ksize*，用于计算二阶导数的滤波器的孔径尺寸，大小必须为正奇数，且有默认值1。

* *scale*，计算拉普拉斯值的时候可选的比例因子，有默认值1。

* *delta*，表示在结果存入目标图（第二个参数dst）之前可选的delta值，有默认值0。

* *borderType*，边界模式，默认值为BORDER_DEFAULT。这个参数可以在官方文档中borderInterpolate()处得到更详细的信息。

#### scharr

当内核大小为 3 时, Sobel内核可能产生比较明显的误差(毕竟，Sobel算子只是求取了导数的近似值而已)。 为解决这一问题，OpenCV提供了Scharr 函数，但该函数仅作用于大小为3的内核。该函数的运算与Sobel函数一样快，但结果却更加精确。

`void Scharr(InputArray src, OutputArray dst, int ddepth, int dx, int dy, double scale=1, double delta=0, intborderType=BORDER_DEFAULT)`

* *src*，为输入图像，填Mat类型即可。

* *dst*，即目标图像，函数的输出参数，需要和源图片有一样的尺寸和类型。

* *ddepth*，输出图像的深度。

* *dx*，x方向上的差分阶数。

* *dy*，y方向上的差分阶数。

* *scale*，计算导数值时可选的缩放因子，默认值是1，表示默认情况下是没有应用缩放的。我们可以在文档中查阅getDerivKernels的相关介绍，来得到这个参数的更多信息。

* *delta*，表示在结果存入目标图（第二个参数dst）之前可选的delta值，有默认值0。

* *borderType*，我们的老朋友了（万年是最后一个参数），边界模式，默认值为BORDER_DEFAULT。这个参数可以在官方文档中borderInterpolate处得到更详细的信息。

以下两者是等价的：

`Scharr(src, dst, ddepth, dx, dy, scale,delta, borderType);`

`Sobel(src, dst, ddepth, dx, dy, CV_SCHARR,scale, delta, borderType);`
