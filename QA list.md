# 程序语言问题

Java 内存管理
Java 编译过程生成什么
Scala在JVM基础上的原理
Scala 作为 functional language 的好处

##软copy和硬copy的区别
###软的话等同于快捷方式,硬的话等同于同一个文件,对应的硬盘内同一个node
http://blog.csdn.net/hairetz/article/details/4168296

##Java Hashmap如何实现 
### 数组+链表
##如何做Hash,如何确保不重复
###先用一个List(bucket)保存index,index由"hash(key)%len"求得,然后如果index重复的话,就挂靠在随后的linklist,每一个Entry都是由(key,value)组成,最后找到对应的key和value

http://yikun.github.io/2015/04/01/Java-HashMap工作原理及实现/
http://blog.csdn.net/vking_wang/article/details/14166593


##加减乘除如何做开方
###(如何使用泰勒公式开方)


##QSort怎么写
http://stackoverflow.com/questions/18262306/quick-sort-with-python

##Tuple和List的区别
Tuples are fixed size in nature whereas lists are dynamic.
In other words, a tuple is immutable whereas a list is mutable.
http://stackoverflow.com/questions/1708510/python-list-vs-tuple-when-to-use-each


# 简历项目问题



##Vanishing Gradients是什么,如何解决这个问题
###如果一个神经网络有超过1层比如4层的话,bp算法会使得四层学习的速度逐步递减,layer4>layer3>layer2>layer1,这意味着在前面的隐藏层中的神经元学习速度要慢于后面的隐藏层.
###BP的本质是对神经元的输出z进行纠正,通过对他求梯度作一个反方向的偏移,这里假设我们Layer1的b要进行反向传播更新权值,那么可以得出公式:"http://neuralnetworksanddeeplearning.com/images/tikz38.png", 这样可以观察到这个等式传播的时候有两个关键点,一个是w权值,一个是sigmoid'(z).sigmoid'(z)是一个max=0.25的正态分布图,所以随着Layers越多显然这个学习的速度就会至少降低75%,所以会产生这个Vanishing Gradients问题.同理,如果这个w权值非常大的话,那么理论上整个乘积也会放大,但是sigmoid'函数里面的z是等于(wx + b),这个数值越大则sigmoid'越靠两边其实数值越小,所以其实最后的学习率通常来讲都是越来越小的.

##Exploding gradient是什么,如何解决这个问题
###梯度爆炸就是当上面的w控制在一个非常大的值,同时sigmoid'也非常大,两个的乘积都大于1的时候,就有了梯度爆炸的问题.

###解决方法:通过对w进行pre-trained可以通过更改w权值来解决Vanishing Gradients,或者更改激活函数从sigmoid换成ReLU,这样ReLU是1/0函数,不会使得传播的时候持续缩小了

###解决方法:gradient clipping解决了Exploding gradient问题,把每次更新的gradient压缩在clip范围内,就不会无限制的连乘导致偏离了.

https://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf
http://www.jianshu.com/p/917f71b06499

##是因为BP训练才产生Exploding gradient和Vanishing Gradients的问题吗
### 应该是的,因为BP训练的过程才得到了上面的等式,才有了这么一系列推论.

http://neuralnetworksanddeeplearning.com/chap5.html


传统方法如何解决文本分类问题
word embedding的作用是什么
word2vec的原理
pre-trained有什么好处
神经网络如何解决overfitting问题
word-level和char-level的区别
不同架构是如何解决文本分类问题(CNN CNN+LSTM RNN)
LSTM为何能记录长期的记忆
RNN和LSTM比 LSTM有何优点
layers之间的影响
编码之间的影响
梯度下降法的原理,还有什么类似的方法
bp的原理
为何bp要求处处可导
Elastic Net, SVM, Random forest, Gradient boosting区别
Terrier Genism LSA算法分别对数据进行了怎样的处理
Mapreduce伪代码应该怎么写
如何对垃圾邮件数据进行预处理,预处理一般有什么办法
逻辑回归为什么要用sigmoid函数
逻辑回归的loss function是什么
对数据进行标准化有什么方法
为什么对数据进行标准化
逻辑回归如何更新他们的parameters
LR如何解决overfitting问题
regularization有哪些 L0 L1 L2分别是什么
各个project数据量有多少
感知器算法的原理
binary feature的原理
Precision和Recall和F1 score有什么区别,作用是什么
log-liner model如何生成最后的结果
Fully connected layer作用是什么