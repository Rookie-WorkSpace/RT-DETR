#!/bin/bash  

# 创建目录  
mkdir -p coco/images  
mkdir -p coco/annotations  

# 显示当前工作目录  
echo "正在将COCO数据集下载到: $(pwd)/coco"  
echo "该过程可能需要较长时间，取决于您的网络速度"  
echo "================================================================"  

# 下载数据集函数  
download_dataset() {  
    local url=$1  
    local filename=$2  
    local dest_dir=$3  
    
    echo "正在下载 $filename ..."  
    
    # 检查是否存在wget，否则使用curl  
    if command -v wget > /dev/null; then  
        wget -c $url -O $dest_dir/$filename  
    else  
        curl -L $url -o $dest_dir/$filename  
    fi  
    
    if [ $? -ne 0 ]; then  
        echo "下载 $filename 失败"  
        exit 1  
    fi  
}  

# 解压函数  
extract_zip() {  
    local filename=$1  
    local dest_dir=$2  
    
    echo "正在解压 $filename ..."  
    unzip -q $filename -d $dest_dir  
    
    if [ $? -ne 0 ]; then  
        echo "解压 $filename 失败"  
        exit 1  
    fi  
    
    # 删除zip文件以节省空间（可选，如果需要保留则注释此行）  
    # rm $filename  
}  

# 定义数据集URL  
TRAIN_URL="http://images.cocodataset.org/zips/train2017.zip"  
VAL_URL="http://images.cocodataset.org/zips/val2017.zip"  
TEST_URL="http://images.cocodataset.org/zips/test2017.zip"  
ANNOTATIONS_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"  
TEST_INFO_URL="http://images.cocodataset.org/annotations/image_info_test2017.zip"  

# 下载图像数据集  
download_dataset $TRAIN_URL "train2017.zip" "coco/images"  
download_dataset $VAL_URL "val2017.zip" "coco/images"  
download_dataset $TEST_URL "test2017.zip" "coco/images"  

# 下载标注文件  
download_dataset $ANNOTATIONS_URL "annotations_trainval2017.zip" "coco"  
download_dataset $TEST_INFO_URL "image_info_test2017.zip" "coco"  

# 解压图像数据集  
echo "正在解压图像数据集，这可能需要一些时间..."  
extract_zip "coco/images/train2017.zip" "coco/images"  
extract_zip "coco/images/val2017.zip" "coco/images"  
extract_zip "coco/images/test2017.zip" "coco/images"  

# 解压标注文件  
echo "正在解压标注文件..."  
extract_zip "coco/annotations_trainval2017.zip" "coco"  
extract_zip "coco/image_info_test2017.zip" "coco"  

echo "================================================================"  
echo "COCO数据集下载和解压完成！"  
echo "数据集位于: $(pwd)/coco"  
echo "- 训练图像: $(ls coco/images/train2017 | wc -l) 张"  
echo "- 验证图像: $(ls coco/images/val2017 | wc -l) 张"  
echo "- 测试图像: $(ls coco/images/test2017 | wc -l) 张"  
echo "================================================================"


