date: 2026/10/29
resource: https://ysyx.oscc.cc/docs/ics-pa/linux.html



## 1. 探索命令行

```shell
命令名称 参数1 参数2 参数3 ...
```

| Command | Parameter        | Function                                                   |
| :------ | :--------------- | :--------------------------------------------------------- |
| `pwd`   |                  | 列出当前所在目录；                                                  |
| `mv`    | `mv SOURCE DEST` | 将`SOURCE`文件重命名为`DEST`文件，如果`DEST`是一个目录，则将`SOURCE`文件移动到该目录下； |
| `rm`    | `rm -r FILE`     | 删除`FILE`文件，如果使用`-r`选项，则可以递归删除一个目录；                         |
| `man`   | `man ls`         | 查看命令的帮助，`man ls`查看`ls`命令的使用方法；[[man Haiku]]                |


## 2. 统计代码行数

```shell
find . | grep '\.cs$\|\.h$' | xargs wc -l
```

+ `man find` : search for files in a directory hierarchy.
+ `man grep`' : print lines that match patterns. [[正则表达式]]
+ `man wc` : print newline, word, and byte counts for each file.
+ `A | B` : `|`这一符号的左右都是Shell命令，`A | B`表示创建两个进程`A`和`B`，并将`A`进程的标准输出连接到`B`进程的标准输入.
+ `xargs` : 将标准输入转换为参数，`xargs wc -l` 等价于`wc -l aaa.c bbb.c include/ccc.h ...`

## 3. 统计磁盘使用情况

```Shell
du -sc /usr/share/* | sort -nr
```

