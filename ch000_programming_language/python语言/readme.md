> 文章出处：小小猿笔记（公众号，B站，知乎，CSDN 同名）
>
> 仓库链接：https://github.com/xiaoxiaojiea/myBlogSources.git



# Python系列文章复习总结



> 终于，python文章已经更新完了，这里做出一下总结复习，相当于是整合出一个目录以便大家使用。



<center><b>文章的章节梳理在下文中，针对文章的视频版梳理在B站 (B站账号：小小猿笔记) 。</b></center>

视频版Python语言知识点快速复习连接：[Python语言程序设计基础知识点复习梳理](https://www.bilibili.com/video/BV1QU4y1x7kW)



>**主要内容：**
>
>- 基础知识 & 环境搭建。
>- PyCharm的基础使用。
>- Python的基础语法。
>- Python的高级语法。
>- Python的文件 & 异常。
>- 飞机大战。
>- 爬虫初窥。
>



## 基础知识 & 环境搭建



- 首先，既然我们来学习`Python`，那大家肯定对他有着一定的了解，但是我还是觉得有必要再介绍一下Python可以做些什么。

- 然后，python与C语言不太一样，由于win电脑就是C语言写的，所以C语言的程序可以直接在win电脑上执行，然后我们为了方便写代码就安装了VC6.0来敲写代码。
- 但是，python程序是不能直接在win电脑上运行的，所以这个时候就需要安装python的运行环境，让他来执行python程序，同样，为了敲代码方便，所以我们安装PyCharm来敲代码。



**至于细节，这篇文章中就有提到。**

[Python零基础入门-01-基础知识&环境搭建](https://mp.weixin.qq.com/s?__biz=MzU2NjY5ODU0MQ==&mid=2247485130&idx=1&sn=21c753a0a70b46863f57ed0d0cd6747b&chksm=fca93188cbdeb89e2f5e83eed92ea07fde13b378e0864eb522d65b0f6090feba6405b3fe227e&token=1839830084&lang=zh_CN#rd) 



## PyCharm的基础使用



- 首先，PyCharm与VC6.0都只是一个工具而已，他们的目的是用来敲代码，但是PyCharm是一款功能很强大的编译器，不像VC6.0那么简单使用。
- 甚至，新手都不知道怎么在PyCharm中找到敲代码的地方，所以这里的文章就会首先给你讲讲这个软件的一些简单用法，让你不至于太恐慌。
- 回顾，上篇文章我们提到学习python要装两个东西：python运行环境 以及 pycharm。所以在使用pycharm的时候首先就需要将python运行环境引入到pycharm中，不然你敲完的代码都没法运行的。
- 同样，也会教你一些简单的有用的操作。



**至于细节，这篇文章中就有提到。**

[Python零基础入门-02-PyCharm基础使用](https://mp.weixin.qq.com/s?__biz=MzU2NjY5ODU0MQ==&mid=2247485165&idx=1&sn=b60fb4170ef5951db79decd4f36f1c04&chksm=fca931afcbdeb8b975e9ced4f3e2943524e181623a41161fb0c386f036ab09729a8e07f64db8&token=1839830084&lang=zh_CN#rd) 





## Python的基础语法



- 首先，任何一门编程语言的基础语法都是差不多的，都是一些 `数据类型，输入输出，选择循环结构，以及函数文件`等等。唯一不同的就是每个编程语言怎么定义这些语法，怎么使用这些语法上，有着区别。python的定义与使用是最简单的。
- 然后，这一节就会带着大家了解一些基础的语法，并且掺杂着案例，让大家见一见python的简洁之处。



**至于细节，这篇文章中就有提到。**

[Python零基础入门-03-Python基础语法](https://mp.weixin.qq.com/s?__biz=MzU2NjY5ODU0MQ==&mid=2247485249&idx=1&sn=be02f92b7b066177bd7f7b2f7805839e&chksm=fca93003cbdeb9156e6a3569f419dedc533a8a9e4bec4b2d4d61d47c020319fe4f693bb7201c&token=1839830084&lang=zh_CN#rd) 





## Python的高级语法



- 首先，刚才说了每个编程语言的基础语法都大相径庭，但是他们的高级语法就各有千秋了，其他的就不说了，这里就会讲一讲python的高级语法。
- 然后，任何编程语言的高级语法，不管再怎么多杂，唯一不变的就是，都必须有面向对象的内容。
- 最后，什么是面向对象呢？ 面向对象其实是一种思想，并不是具体的语法，大家带着这个疑问就往下看文章吧。



**至于细节，这篇文章中就有提到。**

[Python零基础入门-04-Python高级语法](https://mp.weixin.qq.com/s?__biz=MzU2NjY5ODU0MQ==&mid=2247485312&idx=1&sn=aadb0cef5258f6e4da4a5c0a1c6701e1&chksm=fca930c2cbdeb9d48f7b9c5c338f7927f2720fb23bcdf46244628dcba94e863ae71590e531be&token=1839830084&lang=zh_CN#rd) 



## Python的文件 & 异常



- 首先，前边讲基础语法的时候我们并没有讲文件，因为文件一般都是最后学的，哈哈哈哈。
- 文件，其实就是将用户操作完的数据给存到硬盘上，下次用户在用的时候再从文件中读取进来，以达到存档读档的效果。听我讲的热血澎湃的，其实啥也不是，就给你讲讲基础语法而已，哈哈哈哈。
- 异常，其实就是为了解决当程序整体出现错误的时候，不让程序崩溃，而提出的一种解决方式。



**至于细节，这篇文章中就有提到。**

[Python零基础入门-05-结尾(文件&异常)](https://mp.weixin.qq.com/s?__biz=MzU2NjY5ODU0MQ==&mid=2247485384&idx=1&sn=d9c4ef753b1eafc58202676fd9516f21&chksm=fca9308acbdeb99c7478a719d4e10cd42c4e9a0a1cfaba186b299da92e53407c02af487c6353&token=1839830084&lang=zh_CN#rd) 



## 飞机大战



- 在上次我推荐的python基础入门书......，后边有一个飞机大战的项目，但是我发现他运行不起来了，因为他的好像是基于python2的版本，然而现在都用的python3了，所以有一些语法已经跑不通了。
- 然后我自己按照他的思路来实现了一下飞机大战的基础功能，并且给了一些视频的讲解，让大家好理解一点。



> **tips**：pygame不要成为你的学习重点，把我的源码弄个下来跑一跑，了解一点代码以及思路就好，重点还是放在下一个爬虫的项目。



**至于细节，这个视频中就有提到。**

- 飞机大战B站视频链接：[python飞机大战视频讲解](https://www.bilibili.com/video/BV1QU4y1x7kW?p=15).
- 飞机大战源码链接：[飞机大战源码分享](https://github.com/xiaoxiaojiea/SharingFolder/tree/main/%E5%85%AC%E4%BC%97%E5%8F%B7%E5%88%86%E4%BA%AB/python%E7%9B%B8%E5%85%B3)



## 爬虫初窥



- 爬虫可以称之为一个就业方向，各大公司也是有需求的，但是要彻底学会是不太现实的，因为随着爬虫技术的崛起，反爬虫技术也在慢慢发展，现在的网站也是越来越难爬了。
- 但是，对于新手来说，需要学些什么，以及每个技术有什么用还是需要有一个清晰的认识的。
- 下边的文章以及视频可以给你一个清理明了的学习计划。



**至于细节，这个视频中就有提到。**

- 爬虫B站视频链接：[Python爬虫快速入门实战教程(附赠进阶教程)](https://www.bilibili.com/video/BV1Rb4y197s7)
- 爬虫源码链接：[爬虫快速入门源码分享](https://github.com/xiaoxiaojiea/SharingFolder/tree/main/%E5%85%AC%E4%BC%97%E5%8F%B7%E5%88%86%E4%BA%AB/python%E7%9B%B8%E5%85%B3)



**视频中提到的学习资料如下**：

- 教程获取方式：关注公众号 “ **小小猿笔记** ”，或者扫下边的码，然后在公众号内部回复 “ **崔庆才爬虫** ”，即可获取。
- 最新版爬虫教程链接： [52讲轻松搞定网络爬虫](https://kaiwu.lagou.com/course/courseInfo.htm?courseId=46&sid=460-PC_search_list-0#/sale) . （这个是官网，但是网上**免费版**的资源多的很）


