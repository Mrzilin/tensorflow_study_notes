#1.數據類型
###1.1數值類型
数值类型的张量是 TensorFlow 的主要数据载体，根据维度数来区分，可分为:標量/向量/矩陣/張量(Tensor).所有维度数dim > 2的数组统称为张量。张量的每个维度也作轴(Axis)， 一般维度代表了具体的物理含义。

    aa = tf.constant(1.2) # TF方式創建標量
    
     x = tf.constant([1,2.,3.3])
     x #打印TF張量的相關信息
     Out[2]:<tf.Tensor: id=165, shape=(3,), dtype=float32, numpy=array([1. , 2. , 3.3], dtype=float32)>
其中 id 是 TensorFlow 中内部索引对象的编号，shape 表示张量的形状，dtype 表示张量的数 值精度，张量 numpy()方法可以返回 Numpy.array 类型的数据，方便导出数据到系统的其他 模块，代码如下。
    
    In [3]: x.numpy() # 将TF张量的数据导出为numpy数组格式
    Out[3]: array([1. , 2. , 3.3], dtype=float32)
    
     a = tf.constant([1,2, 3.]) # 创建3个元素的向量
###1.2字符串類型

    a = tf.constant('Hello, Deep Learning.') # 创建字符串
    Out:<tf.Tensor: id=17, shape=(), dtype=string, numpy=b'Hello, Deep Learning.'>
 在 tf.strings 模块中，提供了常见的字符串类型的工具函数，如小写化 lower()、拼接 join()、长度 length()、切分 split()等。例如，将字符串全部小写化实现为:
    
    tf.strings.lower(a) # 小写化字符串
    Out:<tf.Tensor: id=19, shape=(), dtype=string, numpy=b'hello, deep learning.'>
    
###1.3布爾類型

    a = tf.constant(True) # 创建布尔类型标量
    Out:<tf.Tensor: id=22, shape=(), dtype=bool, numpy=True>
    
    a = tf.constant([True, False]) # 创建布尔类型向量
需要注意的是，TensorFlow 的布尔类型和 Python 语言的布尔类型并不等价，不能通 用，例如:
    
    a = tf.constant(True) # 创建TF布尔张量
    a is True # TF布尔类型张量与python布尔类型比较
    a == True # 仅数值比较
    Out:False # 对象不等价
    <tf.Tensor: id=8, shape=(), dtype=bool, numpy=True> # 数值比较结果
###1.4數值精度
对于数值类型的张量，可以保存为不同字节长度的精度，如浮点数 3.14 既可以保存为 16 位(Bit)长度，也可以保存为 32 位甚至 64 位的精度。位越长，精度越高，同时占用的内 存空间也就越大。常用的精度类型有 tf.int16、tf.int32、tf.int64、tf.float16、tf.float32、 tf.float64 等，其中 tf.float64 即为 tf.double。

    tf.constant(123456789, dtype=tf.int16)
    tf.constant(123456789, dtype=tf.int32)
    Out:<tf.Tensor: id=33, shape=(), dtype=int16, numpy=-13035>
    <tf.Tensor: id=35, shape=(), dtype=int32, numpy=123456789>
可以看到，保存精度过低时，数据 123456789 发生了溢出，得到了错误的结果，一般使用 tf.int32、tf.int64 精度。
对于大部分深度学习算法，一般使用 tf.int32 和 tf.float32 可满足大部分场合的运算精 度要求，部分对精度要求较高的算法，如强化学习某些算法，可以选择使用 tf.int64 和 tf.float64 精度保存张量。
####讀取精度:
通过访问张量的 dtype 成员属性可以判断张量的保存精度
    
    print('before:',a.dtype) # 读取原有张量的数值精度
    if a.dtype != tf.float32: # 如果精度不符合要求，则进行转换
    a = tf.cast(a,tf.float32) # tf.cast函数可以完成精度转换
    print('after :',a.dtype) # 打印转换后的精度
    Out:before: <dtype: 'float16'>
        after : <dtype: 'float32'>
####类型转换:
系统的每个模块使用的数据类型、数值精度可能各不相同，对于不符合要求的张量的
类型及精度，需要通过 tf.cast 函数进行转换,例如:

    a = tf.constant(np.pi, dtype=tf.float16) # 创建tf.float16低精度张量
    tf.cast(a, tf.double) # 转换为高精度张量
    Out:<tf.Tensor: id=44, shape=(), dtype=float64, numpy=3.140625>
###1.5待優化張量
为了区分需要计算梯度信息的张量与不需要计算梯度信息的张量，TensorFlow 增加了 一种专门的数据类型来支持梯度信息的记录:tf.Variable。tf.Variable 类型在普通的张量类 型基础上添加了 name，trainable 等属性来支持计算图的构建。由于梯度运算会消耗大量的 计算资源，而且会自动更新相关参数，对于不需要的优化的张量，如神经网络的输入𝑿， 不需要通过 tf.Variable 封装;相反，对于需要计算梯度并优化的张量，如神经网络层的𝑾 和𝒃，需要通过 tf.Variable 包裹以便 TensorFlow 跟踪相关梯度信息。

    