---
layout:     post
title:      "openCV2之特征点检测匹配以及RANSAC优化"
subtitle:   " \"opencv\""
date:       2016-09-19
author:     "Leiym"
header-img: "img/post-bg-2015.jpg"
tags:
    - opencv
    - vslam
---

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

		- [特征点检测和图像匹配](#特征点检测和图像匹配)
			- [特征检测子（关键点）与描述子](#特征检测子关键点与描述子)
			- [特征检测子公用接口](#特征检测子公用接口)
				- [class KeyPoint](#class-keypoint)
				- [FeatureDetector](#featuredetector)
			- [特征描述子提取公用接口](#特征描述子提取公用接口)
				- [DescriptorExtractor](#descriptorextractor)
			- [特征描述子匹配公用接口](#特征描述子匹配公用接口)
				- [class DMatch](#class-dmatch)
				- [DescriptorMatcher](#descriptormatcher)
					- [DescriptorMatcher::add](#descriptormatcheradd)
					- [DescriptorMatcher::getTrainDescriptors](#descriptormatchergettraindescriptors)
					- [DescriptorMatcher::clear](#descriptormatcherclear)
					- [DescriptorMatcher::empty](#descriptormatcherempty)
					- [DescriptorMatcher::isMaskSupported](#descriptormatcherismasksupported)
					- [DescriptorMatcher::train](#descriptormatchertrain)
					- [DescriptorMatcher::match](#descriptormatchermatch)
					- [DescriptorMatcher::knnMatch](#descriptormatcherknnmatch)
					- [DescriptorMatcher::radiusMatch](#descriptormatcherradiusmatch)
				- [BruteForceMatcher](#bruteforcematcher)
				- [FlannBasedMatcher](#flannbasedmatcher)
			- [图像特征关键点及关键点匹配绘制函数](#图像特征关键点及关键点匹配绘制函数)
				- [drawMatches](#drawmatches)
				- [drawKeypoints](#drawkeypoints)
		- [RANSAC](#ransac)
			- [findFundamentalMat](#findfundamentalmat)

<!-- /TOC -->


### 特征点检测和图像匹配

角点匹配可以分为以下四个步骤：

	1. 提取检测子：在两张待匹配的图像中寻找那些最容易识别的像素点（角点），比如纹理丰富的物体边缘点等。

	2. 提取描述子：对于检测出的角点，用一些数学上的特征对其进行描述，如梯度直方图，局部随机二值特征等。

		检测子和描述子的常用提取方法有:sift, harris, surf, fast, agast, brisk, freak, brisk,orb等。

	3. 匹配：通过各个角点的描述子来判断它们在两张图像中的对应关系。常用方法如 flann

	4. 去外点：去除错误匹配的外点，保留正确的内点。常用方法有Ransac, GTM。

#### 特征检测子（关键点）与描述子

#### 特征检测子公用接口

OpenCV封装了一些特征检测子(特征点)算法，使得用户能够解决该问题时候方便使用各种算法。这章用来计算的描述子匹配被表达成一个高维空间的向量 vector。所有实现 vector 特征关键点检测子部分继承了 FeatureDetector 接口。

##### class KeyPoint

数据结构如下：

    Point2f pt 			//特征点的坐标

    float size 			//特征点的直径

    float angle			//特征点的方向，[0,360)，（-1代表不可用）

    float response		//特征点的可靠度

    int octave			//特征点所在的图像金字塔的组

    int class_id		//用于聚类的id

##### FeatureDetector

`void FeatureDetector::detect(const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const`

* *image* – Image.

* *mask* – Mask specifying where to look for keypoints (optional). It must be a 8-bit integer matrix with non-zero values in the region of interest.


`void FeatureDetector::detect(const vector<Mat>& images, vector<vector<KeyPoint>>& keypoints, const vector<Mat>& masks=vector<Mat>() ) const`

* *images* – Image set.

* *keypoints* – The detected keypoints. In the second variant of the method keypoints[i] is a set of keypoints detected in images[i] .

* *masks* – Masks for each input image specifying where to look for keypoints (optional). masks[i] is a mask for images[i].

`Ptr<FeatureDetector> FeatureDetector::create(const string& detectorType)`

* *Parameters* — detectorType – Feature detector type.

The following detector types are supported:

    "FAST" – FastFeatureDetector
    "STAR" – StarFeatureDetector
    "SIFT" – SiftFeatureDetector
    "SURF" – SurfFeatureDetector
    "ORB" – OrbFeatureDetector
    "MSER" – MserFeatureDetector
    "GFTT" – GoodFeaturesToTrackDetector
    "HARRIS" – GoodFeaturesToTrackDetector with Harris detector enabled
    "Dense" – DenseFeatureDetector
    "SimpleBlob" – SimpleBlobDetector

Also a combined format is supported: feature detector adapter name ( "Grid" – GridAdaptedFeatureDetector, "Pyramid" – PyramidAdaptedFeatureDetector ) + feature detector name , for example: "GridFAST", "PyramidSTAR" .


```
FastFeatureDetector				//用:ocv:func:FAST 方法封装的特征检测子的类

GoodFeaturesToTrackDetector		//用 goodFeaturesToTrack() 函数实现的特征检测子封装类

MserFeatureDetector				//用 MSER 函数实现的特征检测子封装类.

StarFeatureDetector				//用 StarDetector 函数实现的特征检测子封装类

SiftFeatureDetector				//用 SIFT 函数实现的特征检测子封装类

SurfFeatureDetector				//用 SURF 函数实现的特征检测子封装类

OrbFeatureDetector				//用 ORB 函数实现的特征检测子封装类

```

#### 特征描述子提取公用接口

OpenCV封装了一些特征描述子提取算法，使得用户能够解决该问题时候方便使用各种算法。这章用来计算的描述子提取被表达成一个高维空间的向量 vector。所有实现 vector 特征描述子子提取的部分继承了 DescriptorExtractor 接口。

##### DescriptorExtractor

在这个接口中, 一个关键点的特征描述子可以被表达成密集(dense)，固定维数的向量。 大部分特征描述子按照这种模式每隔固定个像素计算。特征描述子的集合被表达成 Mat , 其中每一行是一个关键的特征描述子。

`void DescriptorExtractor::compute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const`

* *image* — 图像。

* *keypoints* — 输入的特征关键点。不能被计算特征描述子的关键点被略过。

* *descriptors*  – 计算特征描述子。

`Ptr<DescriptorExtractor> DescriptorExtractor::create(const string& descriptorExtractorType)`

* *descriptorExtractorType* – Descriptor extractor type.

	现有实现支持以下几个类型的特征描述子提取方法:

	"SIFT" – SiftDescriptorExtractor

	"SURF" – SurfDescriptorExtractor

	"ORB" – OrbDescriptorExtractor

	"BRIEF" – BriefDescriptorExtractor


```
SiftDescriptorExtractor				//应用:ocv:class:SIFT 来封装的用于计算特征描述子的类.

SurfDescriptorExtractor				//应用:ocv:class:SURF 来封装的用于计算特征描述子的类.

OrbDescriptorExtractor				//应用:ocv:class:ORB 来封装的用于计算特征描述子的类.

CalonderDescriptorExtractor			//应用:ocv:class:RTreeClassifier 来封装的用于计算特征描述子的类.

OpponentColorDescriptorExtractor	//实现了在d对立颜色空间(Opponent Color Space)中计算特征描述子的类.

BriefDescriptorExtractor			//计算BRIEF描述子的类.
```

#### 特征描述子匹配公用接口

##### class DMatch

用于匹配特征关键点的特征描述子的类：查询特征描述子索引, 特征描述子索引, 训练图像索引, 以及不同特征描述子之间的距离。成员变量如下：

```
···
int queryIdx; // query descriptor index
int trainIdx; // train descriptor index
int imgIdx;   // train image index

float distance;
···
```
##### DescriptorMatcher

用于特征关键点描述子匹配的抽象基类。有两类匹配任务：匹配两个图像之间的特征描述子，或者匹配一个图像集与另外一个图像集的特征描述子。

###### DescriptorMatcher::add

`void DescriptorMatcher::add(const vector<Mat>& descriptors)`

增加特征描述子用于特征描述子集训练。

* *descriptors* – Descriptors to add.

###### DescriptorMatcher::getTrainDescriptors

`const vector<Mat>& DescriptorMatcher::getTrainDescriptors() const`

对于特征描述子训练集 trainDescCollection 返回一个值。

###### DescriptorMatcher::clear

`void DescriptorMatcher::clear()`

清空特征描述子训练集。

###### DescriptorMatcher::empty

`bool DescriptorMatcher::empty() const`

返回真，如果特征描述子匹配中没有训练的特征描述子。

###### DescriptorMatcher::isMaskSupported

`bool DescriptorMatcher::isMaskSupported()`

返回真，如果特征描述子匹配不支持 masking permissible matches。

###### DescriptorMatcher::train

`void DescriptorMatcher::train()`

训练一个特征描述子匹配。

###### DescriptorMatcher::match

`void DescriptorMatcher::match(const Mat& queryDescriptors, const Mat& trainDescriptors, vector<DMatch>& matches, const Mat& mask=Mat() ) const`

`void DescriptorMatcher::match(const Mat& queryDescriptors, vector<DMatch>& matches, const vector<Mat>& masks=vector<Mat>() )`

* *queryDescriptors* – 特征描述子查询集.

* *trainDescriptors* – 待训练的特征描述子集. 这个集没被加载到类的对象中.

* *matches* – Matches. matches 尺寸小雨超汛特征描述子的数量.

* *mask* – 特定的在输入查询和训练特征描述子集之间的Mask permissible匹配.

* *masks* – masks集. 每个 masks[i] 特定标记出了在输入查询特征描述子和存储的从第i个图像中提取的特征描述子集 trainDescCollection[i] .

###### DescriptorMatcher::knnMatch

`void DescriptorMatcher::knnMatch(const Mat& queryDescriptors, const Mat& trainDescriptors, vector<vector<DMatch>>& matches, int k, const Mat& mask=Mat(), bool compactResult=false ) const`

`void DescriptorMatcher::knnMatch(const Mat& queryDescriptors, vector<vector<DMatch>>& matches, int k, const vector<Mat>& masks=vector<Mat>(), bool compactResult=false )`

给定查询集合中的每个特征描述子，寻找 k个最佳匹配。

* *queryDescriptors* – Query set of descriptors.

* *trainDescriptors* – Train set of descriptors. This set is not added to the train descriptors collection stored in the class object.

* *mask* – Mask specifying permissible matches between an input query and train matrices of descriptors.

* *masks* – Set of masks. Each masks[i] specifies permissible matches between the input query descriptors and stored train descriptors from the i-th image trainDescCollection[i].

* *matches* – Matches. Each matches[i] is k or less matches for the same query descriptor.

* *k* – Count of best matches found per each query descriptor or less if a query descriptor has less than k possible matches in total.

* *compactResult* – Parameter used when the mask (or masks) is not empty. If compactResult is false, the matches vector has the same size as queryDescriptors rows. If compactResult is true, the matches vector does not contain matches for fully masked-out query descriptors.

###### DescriptorMatcher::radiusMatch

`void DescriptorMatcher::radiusMatch(const Mat& queryDescriptors, const Mat& trainDescriptors, vector<vector<DMatch>>& matches, float maxDistance, const Mat& mask=Mat(), bool compactResult=false ) const`

`void DescriptorMatcher::radiusMatch(const Mat& queryDescriptors, vector<vector<DMatch>>& matches, float maxDistance, const vector<Mat>& masks=vector<Mat>(), bool compactResult=false )`

对于每一个查询特征描述子, 在特定距离范围内寻找特征描述子.

* *queryDescriptors* – Query set of descriptors.

* *trainDescriptors* – Train set of descriptors. This set is not added to the train descriptors collection stored in the class object.

* *mask* – Mask specifying permissible matches between an input query and train matrices of descriptors.

* *masks* – Set of masks. Each masks[i] specifies permissible matches between the input query descriptors and stored train descriptors from the i-th image trainDescCollection[i].

* *matches* – Found matches.

* *compactResult* – Parameter used when the mask (or masks) is not empty. If compactResult is false, the matches vector has the same size as queryDescriptors rows. If compactResult is true, the matches vector does not contain matches for fully masked-out query descriptors.

* *maxDistance* – Threshold for the distance between matched descriptors.

`Ptr<DescriptorMatcher> DescriptorMatcher::clone(bool emptyTrainData) const`

拷贝匹配.

* *emptyTrainData* – If emptyTrainData is false, the method creates a deep copy of the object, that is, copies both parameters and train data. If emptyTrainData is true, the method creates an object copy with the current parameters but with empty train data.

`Ptr<DescriptorMatcher> DescriptorMatcher::create(const string& descriptorMatcherType)`

对于给定参数，创建特征描述子匹配(使用默认的构造函数).

* *descriptorMatcherType* – Descriptor matcher type. Now the following matcher types are supported:

    BruteForce (it uses L2 )

    BruteForce-L1

    BruteForce-Hamming

    BruteForce-HammingLUT

    FlannBased

##### BruteForceMatcher

暴力搜索特征点匹配. 对于第一集合中的特征描述子, 这个匹配寻找了在第二个集合中最近的特征描述子. 这种特征描述子匹配支持 masking permissible特征描述子集合匹配.

##### FlannBasedMatcher

基于Flann的特征描述子匹配. 这个匹配通过 flann::Index_ 在 特征描述子集上训练以及调用最近邻算法寻找最佳的匹配. 因此, 在大量训练样本数据集上，这个匹配会比暴力寻找快. 因为 flann::Index 的原因， FlannBasedMatcher 不支持masking permissible特征描述子集合匹配.

#### 图像特征关键点及关键点匹配绘制函数

##### drawMatches

`void drawMatches(const Mat& img1, const vector<KeyPoint>& keypoints1, const Mat& img2, const vector<KeyPoint>& keypoints2, const vector<DMatch>& matches1to2, Mat& outImg, const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1), const vector<char>& matchesMask=vector<char>(), int flags=DrawMatchesFlags::DEFAULT )`

`void drawMatches(const Mat& img1, const vector<KeyPoint>& keypoints1, const Mat& img2, const vector<KeyPoint>& keypoints2, const vector<vector<DMatch>>& matches1to2, Mat& outImg, const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1), const vector<vector<char>>& matchesMask=vector<vector<char> >(), int flags=DrawMatchesFlags::DEFAULT )`

给定两幅图像，绘制寻找到的特征关键点及其匹配.

* *img1* – First source image.

* *keypoints1* – Keypoints from the first source image.

* *img2* – Second source image.

* *keypoints2* – Keypoints from the second source image.

* *matches* – Matches from the first image to the second one, which means that keypoints1[i] has a corresponding point in keypoints2[matches[i]] .

* *outImg* – Output image. Its content depends on the flags value defining what is drawn in the output image. See possible flags bit values below.

* *matchColor* – Color of matches (lines and connected keypoints). If matchColor==Scalar::all(-1) , the color is generated randomly.

* *singlePointColor* – Color of single keypoints (circles), which means that keypoints do not have the matches. If singlePointColor==Scalar::all(-1) , the color is generated randomly.

* *matchesMask* – Mask determining which matches are drawn. If the mask is empty, all matches are drawn.

* *flags* – Flags setting drawing features. Possible flags bit values are defined by DrawMatchesFlags.

##### drawKeypoints

`void drawKeypoints(const Mat& image, const vector<KeyPoint>& keypoints, Mat& outImg, const Scalar& color=Scalar::all(-1), int flags=DrawMatchesFlags::DEFAULT )`

绘制特征关键点.

* *image* – Source image.

* *keypoints* – Keypoints from the source image.

* *outImg* – Output image. Its content depends on the flags value defining what is drawn in the output image. See possible flags bit values below.

* *color* – Color of keypoints.

* *flags* – Flags setting drawing features. Possible flags bit values are defined by DrawMatchesFlags. See details above in drawMatches() .

### RANSAC

#### findFundamentalMat

`CV_EXPORTS Mat findFundamentalMat( const Mat& points1, const Mat& points2, CV_OUT vector<uchar>& mask, int method=FM_RANSAC, double param1=3, double param2=0.99);`

`CV_EXPORTS_W Mat findFundamentalMat( const Mat& points1, const Mat& points2,int method=FM_RANSAC,double param1=3., double param2=0.99);`

* *points1* – Array of N points from the first image. The point coordinates should be floating-point (single or double precision).

* *points2* – Array of the second image points of the same size and format as points1 .

* *method* – Method for computing a fundamental matrix.

  *CV_FM_7POINT* for a 7-point algorithm.  N = 7

  *CV_FM_RANSAC* for the RANSAC algorithm.  N ≥ 8

  *CV_FM_LMEDS* for the LMedS algorithm.  N ≥ 8

* *param1* – Parameter used for RANSAC.

  It is the maximum distance from a point to an epipolar line in pixels, beyond which the point is considered an outlier and is not used for computing the final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the point localization, image resolution, and the image noise.

* *param2* – Parameter used for the RANSAC or LMedS methods only.
  It specifies a desirable level of confidence (probability) that the estimated matrix is correct.

* *status* – Output array of N elements, every element of which is set to 0 for outliers and to 1 for the other points. The array is computed only in the RANSAC and LMedS methods. For other methods, it is set to all 1’s.

**注意**：points1，points2均是Mat类型，不能直接使用 KeyPoint 类对象，所以要将 KeyPoint 类型对象转换为 Mat 类型对象，而且必须是2维，float类型，CV_32F的 Mat。
