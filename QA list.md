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

##常见排序算法的稳定性分析和结论 
http://www.oschina.net/question/565065_86352


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

##pre-trained有什么好处
###神经网络相比于传统ML,从解决凸函数转成解决非凸优化问题,不同的pre-trained方法可以减少落在局部最优区间的可能(解决深层结构相关的优化难题带来希望)

##神经网络如何解决overfitting问题 / layers之间的影响
### 使用深度结构多层叠加低级特征，以获取高级特征，会得到更好的Generalization。

##神经网络比起SVM, 决策树优于哪里
###这些利用局部数据做优化的浅层结构基于先验知识（Prior), 即，给定样本(xi,yi),尽可能从数值上做优化，使得训练出来的模型，对于近似的x，输出近似的y。然而一旦输入值做了泛型迁移，比如两种不同的鸟，鸟的颜色有别，且在图像中的比例不一，那么SVM、决策树几乎毫无用处。所以需要通过model来提取出特征,而不是做数值优化.

##梯度下降法的原理,还有什么类似的方法
###一阶导数（梯度下降）优化法、二阶导数（牛顿法）,后者不受坐标系影响,理论上就是不用做feature scaling也能直接求了吧.

http://www.cnblogs.com/neopenx/p/4575527.html


##为何W不能全部初始化为(相同的权值)0或者1
### 这样会导致每个节点都一样对称,无论是正向传播还是反向传播,每个神经元都是对称的,就无法提取出特征,就失去了neural network的作用了

http://blog.csdn.net/u012767526/article/details/51405701


##编码之间的影响
### 在stackoverflow上提问了,
http://stackoverflow.com/questions/38679431/whats-the-difference-between-text-encoding-when-using-convolution-neural-networ


##Batch Normalization的作用 / 为什么对数据进行标准化
###BN主要是对卷积之后的数据进行统一的scale,本来每一层的learning rate应该是不一样的,但是现在统一scale之后可以更好的使用大的lr了,另外也解决了之前的sigmoid带来的gradient vanishing问题,但是要注意的是其中两个参数gamma,beta是防止一直使用sigmoid中间那一段接近线性的部分.另外BN可以解决overfitting问题,以前的dropout或者l2都可以减低或者干脆去掉.如果只有4个象限的话,大部分数据可能会只落在第一象限,这样w和b要经过多次的训练才能找到dataset中,同时有可能在边上就overfitting了,通过z-score normalization或者BN可以使得数据更加分散开,这样有利于初始化w和b更快的找到合适的分割点.

http://blog.csdn.net/happynear/article/details/44238541

##对数据进行标准化/归一化(feature scaling)有什么方法,有什么帮助?
###常用Z-score,先求均值,再求方差,最后映射到均值=0 方差=1 的区间内.能把目标函数从比较扁转化成比较圆(假设只有两个特征),容易收敛.同时scale一样了,learning rate可以用同样的,比较大的数值,增加的效率,这个和BN是一个道理. 另外一些loss function要求Euclidean distance,不做归一化的话就会被某些特别大数值的特征给影响的太大了.模型是否具有伸缩不变性也是一个核心的问题,比如SVM需要,logistic regression可选(即前面分析的目标函数从扁拉回来).

https://www.zhihu.com/question/30038463
http://www.zhaokv.com/2016/01/normalization-and-standardization.html

##bp的原理
### gradient descent + chain rules吧 (复合函数的链式法则)

https://www.zhihu.com/question/27239198?rf=24827633
https://zhuanlan.zhihu.com/p/21407711?refer=intelligentunit

##word embedding的作用是什么 / word2vec的原理
### 把word映射到一个向量空间,其中相近意义的word会在这个空间内更加相近.

https://www.zhihu.com/question/32275069

##embedding layer的作用是什么
### 降低dim (Dimensionality reduction) + fixed size 统一了输入model的input

https://www.zhihu.com/question/48688908

##Fully connected layer作用是什么
### 特殊形式的卷积层,kernal size = input length,把多个feature组合起来提取更高维度的组合特征.如果前面是a,b,c,d...feature的话, FC layers应该是指a&b,b&c这样的feature,至于最后的output layer则是激活函数是softmax从而分类到比如说10类.

##为何bp要求处处可导
### 反向传播的时候需要求导,但是似乎有不能求导的点,可以用旁边的导数或者说恰好遇到的概率比较低.另外ReLU这种似乎是分成了软饱和/硬饱和激活函数.

http://chuansong.me/n/472464751936

## Sigmoid,Tanh,ReLU之间的对比,ReLU和dropout的对比
### Sigmoid, Tanh出现概率的时候必用
### ReLU能解决梯度消失问题,但是注意要clip来防止梯度爆炸问题.同时带来的稀疏性使得他也具有了非线性表达的能力. 另外这三个激活函数从前到后,W的解空间不断减少,有助于训练加快
### ReLU如果输入0,输出也是0,可以维持输入的稀疏.dropout更像是L2规范化,通过打压w来进行的.

https://www.zhihu.com/question/41841299
http://chuansong.me/n/472464751936


##如何理解神经网络中的非线性的拟合能力  /  感知器算法的原理
###都是线性函数的话,model就直接退化成线性分类器了.比如最后输出函数是sigmoid的话,前面就等同于sigmoid(w2w1x1 + w2b1 + b2),相当于一个线性回归+sigmoid = logistic regression,也就是一个线性分类器LDA了. 如果都是线性激活函数的话,相当于model只有input layer和output layer,这时候就是感知器算法了.
https://www.zhihu.com/question/30165798


## Precision和Recall和F1 score有什么区别,作用是什么


##regularization有哪些 L0 L1 L2分别是什么
### L2 (Ridge regression) L1 (Lasso) and L1+L2 ElasticNet.

http://blog.csdn.net/zouxy09/article/details/24971995

## LR如何解决overfitting问题

http://blog.csdn.net/u012162613/article/details/44261657





传统方法如何解决文本分类问题
不同架构是如何解决文本分类问题(CNN CNN+LSTM RNN)
LSTM为何能记录长期的记忆
RNN和LSTM比 LSTM有何优点
word-level和char-level的区别
Elastic Net, SVM, Random forest, Gradient boosting区别
Terrier Genism LSA算法分别对数据进行了怎样的处理
Mapreduce伪代码应该怎么写
如何对垃圾邮件数据进行预处理,预处理一般有什么办法
逻辑回归为什么要用sigmoid函数
逻辑回归的loss function是什么
逻辑回归如何更新他们的parameters
binary feature的原理
log-liner model如何生成最后的结果



