# 1.數據類型
### 1.1數值類型
数值类型的张量是 TensorFlow 的主要数据载体，根据维度数来区分，可分为:標量/向量/矩陣/張量(Tensor).所有维度数dim > 2的数组统称为张量。张量的每个维度也作轴(Axis)， 一般维度代表了具体的物理含义。

    aa = tf.constant(1.2) # TF方式創建標量
    
     x = tf.constant([1,2.,3.3])
     x #打印TF張量的相關信息
     Out[2]:<tf.Tensor: id=165, shape=(3,), dtype=float32, numpy=array([1. , 2. , 3.3], dtype=float32)>
     
其中 id 是 TensorFlow 中内部索引对象的编号，shape 表示张量的形状，dtype 表示张量的数 值精度，张量 numpy()方法可以返回 Numpy.array 类型的数据，方便导出数据到系统的其他 模块，代码如下。
   
    In [3]: x.numpy() # 将TF张量的数据导出为numpy数组格式
    Out[3]: array([1. , 2. , 3.3], dtype=float32)
    
     a = tf.constant([1,2, 3.]) # 创建3个元素的向量
### 1.2字符串類型
    a = tf.constant('Hello, Deep Learning.') # 创建字符串
    Out:<tf.Tensor: id=17, shape=(), dtype=string, numpy=b'Hello, Deep Learning.'>
 在 tf.strings 模块中，提供了常见的字符串类型的工具函数，如小写化 lower()、拼接 join()、长度 length()、切分 split()等。例如，将字符串全部小写化实现为:
    
    tf.strings.lower(a) # 小写化字符串
    Out:<tf.Tensor: id=19, shape=(), dtype=string, numpy=b'hello, deep learning.'>
    
### 1.3布爾類型
    a = tf.constant(True) # 创建布尔类型标量
    Out:<tf.Tensor: id=22, shape=(), dtype=bool, numpy=True>
    
    a = tf.constant([True, False]) # 创建布尔类型向量
需要注意的是，TensorFlow 的布尔类型和 Python 语言的布尔类型并不等价，不能通 用，例如:
    
    a = tf.constant(True) # 创建TF布尔张量
    a is True # TF布尔类型张量与python布尔类型比较
    a == True # 仅数值比较
    Out:False # 对象不等价
    <tf.Tensor: id=8, shape=(), dtype=bool, numpy=True> # 数值比较结果
### 1.4數值精度
对于数值类型的张量，可以保存为不同字节长度的精度，如浮点数 3.14 既可以保存为 16 位(Bit)长度，也可以保存为 32 位甚至 64 位的精度。位越长，精度越高，同时占用的内 存空间也就越大。常用的精度类型有 tf.int16、tf.int32、tf.int64、tf.float16、tf.float32、 tf.float64 等，其中 tf.float64 即为 tf.double。

    tf.constant(123456789, dtype=tf.int16)
    tf.constant(123456789, dtype=tf.int32)
    Out:<tf.Tensor: id=33, shape=(), dtype=int16, numpy=-13035>
    <tf.Tensor: id=35, shape=(), dtype=int32, numpy=123456789>
可以看到，保存精度过低时，数据 123456789 发生了溢出，得到了错误的结果，一般使用 tf.int32、tf.int64 精度。
对于大部分深度学习算法，一般使用 tf.int32 和 tf.float32 可满足大部分场合的运算精 度要求，部分对精度要求较高的算法，如强化学习某些算法，可以选择使用 tf.int64 和 tf.float64 精度保存张量。
#### 讀取精度:
通过访问张量的 dtype 成员属性可以判断张量的保存精度
    
    print('before:',a.dtype) # 读取原有张量的数值精度
    if a.dtype != tf.float32: # 如果精度不符合要求，则进行转换
    a = tf.cast(a,tf.float32) # tf.cast函数可以完成精度转换
    print('after :',a.dtype) # 打印转换后的精度
    Out:before: <dtype: 'float16'>
        after : <dtype: 'float32'>
#### 类型转换:
系统的每个模块使用的数据类型、数值精度可能各不相同，对于不符合要求的张量的
类型及精度，需要通过 tf.cast 函数进行转换,例如:

    a = tf.constant(np.pi, dtype=tf.float16) # 创建tf.float16低精度张量
    tf.cast(a, tf.double) # 转换为高精度张量
    Out:<tf.Tensor: id=44, shape=(), dtype=float64, numpy=3.140625>
### 1.5待優化張量
为了区分需要计算梯度信息的张量与不需要计算梯度信息的张量，TensorFlow 增加了 一种专门的数据类型来支持梯度信息的记录:tf.Variable。tf.Variable 类型在普通的张量类 型基础上添加了 name，trainable 等属性来支持计算图的构建。由于梯度运算会消耗大量的 计算资源，而且会自动更新相关参数，对于不需要的优化的张量，如神经网络的输入𝑿， 不需要通过 tf.Variable 封装;相反，对于需要计算梯度并优化的张量，如神经网络层的𝑾 和𝒃，需要通过 tf.Variable 包裹以便 TensorFlow 跟踪相关梯度信息。

    a = tf.constant([-1, 0, 1, 2]) # 创建 TF 张量 
    aa = tf.Variable(a) # 转换为 Variable 类型 
    aa.name, aa.trainable # Variable类型张量的属性
    Out:('Variable:0', True)
    
    a = tf.Variable([[1,2],[3,4]]) # 直接创建Variable张量
    
待优化张量可视为普通张量的特殊类型，普通张量其实也可以通过 GradientTape.watch()方 法临时加入跟踪梯度信息的列表，从而支持自动求导功能。
#### 創建張量
通过 tf.convert_to_tensor 函数可以创建新 Tensor，并将保存在 Python List 对象或者 Numpy Array 对象中的数据导入到新 Tensor 中，例如:

    tf.convert_to_tensor([1,2.]) # 从列表创建张量
    tf.convert_to_tensor(np.array([[1,2.],[3,4]])) # 从数组中创建张量
需要注意的是，Numpy 浮点数数组默认使用 64 位精度保存数据，转换到 Tensor 类型时精 度为 tf.float64，可以在需要的时候将其转换为 tf.float32 类型。
实际上，tf.constant()和 tf.convert_to_tensor()都能够自动的把 Numpy 数组或者 Python 列表数据类型转化为 Tensor 类型.

#### 创建全0或全1张量

    tf.zeros([]),tf.ones([]) # 创建全0，全1的标量
    tf.zeros([1]),tf.ones([1]) # 创建全0，全1的向量
    tf.zeros([2,2]) # 创建全0矩阵，指定shape为2行2列
    tf.ones([3,2]) # 创建全1矩阵，指定shape为3行2列
通过 tf.zeros_like, tf.ones_like 可以方便地新建与某个张量 shape 一致，且内容为全 0 或 全 1 的张量。例如，创建与张量𝑨形状一样的全 0 张量:

    a = tf.ones([2,3]) # 创建一个矩阵
    tf.zeros_like(a) # 创建一个与a形状相同，但是全0的新矩阵
    Out:<tf.Tensor: id=113, shape=(2, 3), dtype=float32, numpy= array([[0., 0., 0.],[0., 0., 0.]], dtype=float32)>
    
tf.*_like 是一系列的便捷函数，可以通过 tf.zeros(a.shape)等方式实现。
#### 创建自定义数值张量
通过 tf.fill(shape, value)可以创建全为自定义数值 value 的张量，形状由 shape 参数指
定。例如，创建元素为−1的标量:
    
    tf.fill([], -1) # 创建-1的标量
    tf.fill([1], -1) # 创建-1的向量
    tf.fill([2,2], 99) # 创建2行2列，元素全为99的矩阵
#### 创建已知分布的张量
正态分布(Normal Distribution，或 Gaussian Distribution)和均匀分布(Uniform Distribution)是最常见的分布之一，创建采样自这 2 种分布的张量非常有用，比如在卷积神 经网络中，卷积核张量𝑾初始化为正态分布有利于网络的训练;在对抗生成网络中，隐藏 变量𝒛一般采样自均匀分布。
通过 tf.random.normal(shape, mean=0.0, stddev=1.0)可以创建形状为 shape，均值为 mean，标准差为 stddev 的正态分布𝒩(mean, stddev2)。例如，创建均值为 0，标准差为 1 的正态分布:

    tf.random.normal([2,2]) # 创建标准正态分布的张量
    tf.random.normal([2,2], mean=1,stddev=2) # 创建均值为 1，标准差为 2 的正态分布 
通过 tf.random.uniform(shape, minval=0, maxval=None, dtype=tf.float32)可以创建采样自 [minval, maxval)区间的均匀分布的张量。例如创建采样自区间[0,1)，shape 为[2,2]的矩 阵:

    tf.random.uniform([2,2]) # 创建采样自[0,1)均匀分布的矩阵
    Out:<tf.Tensor: id=158, shape=(2, 2), dtype=float32, numpy= array([[0.65483284, 0.63064325],[0.008816 , 0.81437767]], dtype=float32)>
创建采样自区间[0,10)，shape 为[2,2]的矩阵:

    tf.random.uniform([2,2],maxval=10) # 创建采样自[0,10)均匀分布的矩阵
如果需要均匀采样整形类型的数据，必须指定采样区间的最大值 maxval 参数，同时指 定数据类型为 tf.int*型:

    tf.random.uniform([2,2],maxval=100,dtype=tf.int32) # 创建采样自[0,100)均匀分布的整型矩阵
    Out:<tf.Tensor: id=171, shape=(2, 2), dtype=int32, numpy=array([[61, 21], [95, 75]])>
#### 創建序列
在循环计算或者对张量进行索引时，经常需要创建一段连续的整型序列，可以通过 tf.range()函数实现。tf.range(limit, delta=1)可以创建[0, limit)之间，步长为 delta 的整型序 列，不包含 limit 本身。例如，创建 0~10，步长为 1 的整型序列:

    tf.range(10) # 0~10，不包含10
    Out：<tf.Tensor: id=180, shape=(10,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5,6, 7, 8, 9])>
    tf.range(10,delta=2)  #创建 0~10，步长为 2 的整形序列:
    Out:<tf.Tensor: id=185, shape=(5,), dtype=int32, numpy=array([0, 2, 4, 6, 8])>
    #通过 tf.range(start, limit, delta=1)可以创建[start, limit)，步长为 delta 的序列，不包含 limit 本身:
    tf.range(1,10,delta=2) # 1~10
    Out:<tf.Tensor: id=190, shape=(5,), dtype=int32, numpy=array([1, 3, 5, 7, 9])>
#### 三維向量
三维的张量一个典型应用是表示序列信号，它的格式是
𝑿 = [𝑏, sequence len, feature len] 其中𝑏表示序列信号的数量，sequence len 表示序列信号在时间维度上的采样点数或步数，
feature len 表示每个点的特征长度。
考虑自然语言处理(Natural Language Processing，简称 NLP)中句子的表示，如评价句 子的是否为正面情绪的情感分类任务网络，如图 4.3 所示。为了能够方便字符串被神经网 络处理，一般将单词通过嵌入层(Embedding Layer)编码为固定长度的向量，比如“a”编码 为某个长度 3 的向量，那么 2 个等长(单词数量为 5)的句子序列可以表示为 shape 为[2,5,3] 的 3 维张量，其中 2 表示句子个数，5 表示单词数量，3 表示单词向量的长度。

#### 四维张量
四维张量在卷积神经网络中应用非常广泛，它用于保存特征图(Feature maps)数据，格
式一般定义为
[𝑏, h, , 𝑐]
其中𝑏表示输入样本的数量，h/ 分别表示特征图的高/宽，𝑐表示特征图的通道数，部分深 度学习框架也会使用[𝑏, 𝑐, h, w]格式的特征图张量，例如 PyTorch。图片数据是特征图的一 种，对于含有 RGB 3 个通道的彩色图片，每张图片包含了h行 列像素点，每个点需要 3 个数值表示 RGB 通道的颜色强度，因此一张图片可以表示为[h, w, 3]

    # 创建32x32的彩色图片输入，个数为4
    x = tf.random.normal([4,32,32,3])
    
### 1.6 索引與切片
####索引
在 TensorFlow 中，支持基本的[i][j] ⋯标准索引方式，也支持通过逗号分隔索引号的索 引方式。考虑输入x为 4 张32 × 32大小的彩色图片(为了方便演示，大部分张量都使用随机 分布模拟产生，后文同)，shape 为[4,32,32,3]，首先创建张量:
x = tf.random.normal([4,32,32,3]) # 创建4D张量 接下来我们使用索引方式读取张量的部分数据。
取第1张图片的数据，实现如下:

    In:x[0] # 程序中的第一的索引号应为0，容易混淆，不过不影响理解
    Out:<tf.Tensor: id=379, shape=(32, 32, 3), dtype=float32, numpy= array([[[ 1.3005302 , 1.5301839 , -0.32005513],
    [-1.3020388 , 1.7837263 , -1.0747638 ], ...
    [-1.1092019 , -1.045254 , -0.4980363 ],
    [-0.9099222 , 0.3947732 , -0.10433522]]], dtype=float32)>
取第1张图片的第2行，实现如下:

    In:x[0][1]
    Out:<tf.Tensor: id=388, shape=(32, 3), dtype=float32, numpy= array([[ 4.2904025e-01, 1.0574218e+00, 3.1540772e-01],
    [ 1.5800388e+00, -8.1637271e-02, 6.3147342e-01], ...,
    [ 2.8893018e-01, 5.8003378e-01, -1.1444757e+00],
    [ 9.6100050e-01, -1.0985689e+00, 1.0827581e+00]], dtype=float32)>
    
取第1张图片，第2行，第3列的数据，实现如下：
    
    In:x[0][1][2]
    Out:<tf.Tensor: id=401, shape=(3,), dtype=float32, numpy=array([-0.55954427, 0.14497331, 0.46424514], dtype=float32)>
    
取第3张图片，第2行，第1列的像素，B通道(第2个通道)颜色强度值，实现如下:
    
    In:x[2][1][0][1]
    Out:<tf.Tensor: id=418, shape=(), dtype=float32, numpy=-0.84922135>
当张量的维度数较高时，使用[i][j]. . . [k]的方式书写不方便，可以采用[i, j, ... , k]的方 式索引，它们是等价的。
取第2张图片，第10行，第3列的数据，实现如下:

    In:x[1,9,2]
    Out:<tf.Tensor: id=436, shape=(3,), dtype=float32, numpy=array([ 1.7487534 , - 0.41491988, -0.2944692 ], dtype=float32)>
    
#### 切片
通过start: end: step切片方式可以方便地提取一段数据，其中 start 为开始读取位置的索
引，end 为结束读取位置的索引(不包含 end 位)，step 为采样步长。
以 shape 为[4,32,32,3]的图片张量为例，我们解释如果切片获得不同位置的数据。例如
读取第 2,3 张图片，实现如下:


