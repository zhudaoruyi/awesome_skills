# Ubuntu环境下挂载新硬盘


一、硬盘分区 | Hard disk add new partition

**1、显示硬盘及所属分区情况。在终端窗口中输入如下命令：**

```bash
sudo fdisk -l
```



**显示当前的硬盘及所属分区的情况。如下图所示：**
**系统提示：DIsk /dev/sdb doesn't contain a valid partition table。**

![1.gif](https://s1.51cto.com/images/20180329/1522291142953258.gif?x-oss-process=image/watermark,size_16,text_QDUxQ1RP5Y2a5a6i,color_FFFFFF,t_100,g_se,x_10,y_10,shadow_90,type_ZmFuZ3poZW5naGVpdGk=)

**２、对硬盘进行分区。在终端窗口中输入如下命令：**

```bash
sudo fdisk /dev/sdb
```



如下图所示：
在Command (m for help)提示符后面输入m显示一个帮助菜单。

![2.gif](https://s1.51cto.com/images/20180329/1522291150631562.gif?x-oss-process=image/watermark,size_16,text_QDUxQ1RP5Y2a5a6i,color_FFFFFF,t_100,g_se,x_10,y_10,shadow_90,type_ZmFuZ3poZW5naGVpdGk=)

 **在Command (m for help)提示符后面输入n，执行 add a new partition 指令给硬盘增加一个新分区。 出现Command action时，输入e，指定分区为扩展分区（extended）。 出现Partition number(1-4)时，输入１表示只分一个区。 后续指定起启柱面（cylinder）号完成分区。**

![3.gif](https://s1.51cto.com/images/20180329/1522291159502739.gif?x-oss-process=image/watermark,size_16,text_QDUxQ1RP5Y2a5a6i,color_FFFFFF,t_100,g_se,x_10,y_10,shadow_90,type_ZmFuZ3poZW5naGVpdGk=)

**在Command (m for help)提示符后面输入p，显示分区表。系统提示如下：Device Boot                 Start                End                   Blocks          Id             System/dev/sdb1                           1            26108           209712478+           5          Extended**

![4.gif](https://s1.51cto.com/images/20180329/1522291166220638.gif?x-oss-process=image/watermark,size_16,text_QDUxQ1RP5Y2a5a6i,color_FFFFFF,t_100,g_se,x_10,y_10,shadow_90,type_ZmFuZ3poZW5naGVpdGk=)

**在Command (m for help)提示符后面输入w，保存分区表。系统提示：The partition table has been altered!**

![5.gif](https://s1.51cto.com/images/20180329/1522291175167083.gif?x-oss-process=image/watermark,size_16,text_QDUxQ1RP5Y2a5a6i,color_FFFFFF,t_100,g_se,x_10,y_10,shadow_90,type_ZmFuZ3poZW5naGVpdGk=)

**在终端窗口中输入如下命令：**

```bash
sudo fdisk -l
```



如下图所示：
系统已经识别了硬盘 /dev/sdb 的分区。

![6.gif](https://s1.51cto.com/images/20180329/1522291184648789.gif?x-oss-process=image/watermark,size_16,text_QDUxQ1RP5Y2a5a6i,color_FFFFFF,t_100,g_se,x_10,y_10,shadow_90,type_ZmFuZ3poZW5naGVpdGk=)



**二、硬盘格式化 | Format hard disk**

**1、显示硬盘及所属分区情况。在终端窗口中输入如下命令：**

```bash
sudo mkfs -t ext4 /dev/sdb
```



说明：
-t ext4 表示将分区格式化成ext4文件系统类型。

![7.png](https://s1.51cto.com/images/20180329/1522284674305157.png?x-oss-process=image/watermark,size_16,text_QDUxQ1RP5Y2a5a6i,color_FFFFFF,t_100,g_se,x_10,y_10,shadow_90,type_ZmFuZ3poZW5naGVpdGk=)



**三、挂载硬盘分区 | Mount hard disk partition**

**1、显示硬盘挂载情况。在终端窗口中输入如下命令：**

```bash
df -l
```



新硬盘分区没有挂载，无法进入和查看。

**在终端窗口中输入如下命令****：**

```bash
sudo mount -t ext4 /dev/sdb /devdata
```



**再次在终端窗口中输入如下命令****：**

```bash
df -l
```



新硬盘分区已经挂载，如下图最下面的红色方框内容。

![8.png](https://s1.51cto.com/images/20180329/1522284851211104.png?x-oss-process=image/watermark,size_16,text_QDUxQ1RP5Y2a5a6i,color_FFFFFF,t_100,g_se,x_10,y_10,shadow_90,type_ZmFuZ3poZW5naGVpdGk=)

**２、配置硬盘在系统启动自动挂载。在文件 /etc/fstab 中加入如下配置：**

```bash
/dev/sdb     /devdata    ext4     defaults       0 0
sudo blkid /dev/sda2  # 查询uuid

```




![image.png](https://s1.51cto.com/images/20180329/1522306284208399.png?x-oss-process=image/watermark,size_16,text_QDUxQ1RP5Y2a5a6i,color_FFFFFF,t_100,g_se,x_10,y_10,shadow_90,type_ZmFuZ3poZW5naGVpdGk=)

```UUID=88069947069936E2 /mnt/data ntfs defaults  0  2```
1
第一个数字：0表示开机不检查磁盘，1表示开机检查磁盘； 
第二个数字：0表示交换分区，1代表启动分区（Linux），2表示普通分区 

**至此ubuntu硬盘的挂载就完成了**