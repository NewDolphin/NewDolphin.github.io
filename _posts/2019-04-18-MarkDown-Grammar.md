---
layout:     post
title:      MarkDown 语法
subtitle:   语法基础
date:       2019-04-18
author:     XP
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - MarkDown
---
### 标题
如果一段文字被定义为标题，只要在这段文字前加 # 号即可。

\# 一级标题

\## 二级标题

\### 三级标题

以此类推，总共六级标题，建议在`#`号后加一个空格，这是最标准的 Markdown语法。一级、二级标题会有下分割线

### 列表

1. 无序列表  
    在段落前 加`*`号
    * 第一点
    * 第二点  

2. 有序列表  
	在段落前 加`1.`号
	1. 第一点
	2. 第二点  

> 1. 文字与符号之间有空格  
> 2. 嵌套列表需要在下一行，进行缩进
> 3. 改行末尾添加两个空格表示换行

### 转义字符
如果你的描述中需要用到 markdown 的符号，比如 _ # * 等，但又不想它被转义，这时候可以在这些符号前加反斜杠，如 `\_ \# \*` 进行避免。

### 强调
强调内容前后 + `

### 图片、链接、邮件地址
1. 图片  
`![Mou icon](http://mouapp.com/Mou_128.png)`
![Mou icon](http://mouapp.com/Mou_128.png)
2. 链接  
`[Baidu](www.baidu.com)`  
[Baidu](www.baidu.com)
3. 邮件地址  
`<uranusjr@gmail.com>`  
<uranusjr@gmail.com>

### 斜体、粗体
1. 粗体  
`**你好**` **你好**
2. 斜体  
`*你好*` *你好*

### 表格

1. --- 默认标题居中，内容左对齐
2. :-- 标题、内容全部左对齐
3. --: 标题、内容全部右对齐
4. :-: 标题、内容全部居中

```
| Tables        | Are           | Cool  |  
| :------------ | ------------: | :----:|  
| col 3 is      | right-aligned | $1600 |  
| col 2 is      | centered      |   $12 |  
| zebra stripes | are neat      |    $1 |  
```

| Tables        | Are           | Cool  |
| :------------ | ------------: | :----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |


### 引用、代码框、代码高亮
1. 例如想说明一个标点符号
`"` \`"`
2. 代码框 

	```
	print(add(1,2))
	
	def add(a, b, c=None):
	    result = a + b
	
	    if c is not None:
	        result = result * c
	
	    return result
	```

3. 代码高亮

	~~~python
	print(add(1,2))
	
	def add(a, b, c=None):
	    result = a + b
	
	    if c is not None:
	        result = result * c
	
	    return result
	~~~

### 分割线
`***`
***