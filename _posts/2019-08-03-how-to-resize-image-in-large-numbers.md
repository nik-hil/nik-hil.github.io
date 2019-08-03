---
layout: post
title:  "How to resize images in large numbers"
date:   2019-08-03 2:24:42 +0530
tags: [training, image, resize, parallel, mogrify]
---

* toc
{:toc}

# Resizing

Following is useful when you have to work with a large number of images.
I m working with [kaggle competition](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data). I will preprocess the files to speed up the training process.

# Does preprocessing help?
 
 1. During training, many CPU cycles are wasted to recreate the desired image size. If images are lost, then in the new training cycle we have to recreate them
 1. Using parallel will reduce the time to preprocess the image.

# Steps

 1. Install kaggle `pip3 install kaggle` with token. See https://github.com/Kaggle/kaggle-api
 1. Download desired files `kaggle competitions download favorita-grocery-sales-forecasting -f test.csv.7z`
 1. Unzip files in a separate direcotry
    ```
    mkdir -p <target-dir>
    unzip <file.zip> -d <target-dir>
    ```
 1. Install `parallel` & `mogrify`
    ```
    sudo apt install parallel mogrify
    ```
 1. I will keep the converted files in separate directory.If needed I can create new size files.
    ```
    mkdir -p <target-dir>-224
    ```
 1. Run the conversion commands. 
    ```
    cd <target-dir>; find . -type f | egrep "*.jpg" | parallel mogrify -resize 224x224! -path ../<target-dir>-224
    ```
 1. I m converting all images to size of 224 x 224 irrespective of original size.
 1. Large cpu count will help. `parallel` will create new mogrify commands for each image on each cpu. 
 1. Use `ls -l <target-dir>-224 | wc -l` to find the progress of conversion.
 1. After using above steps, I was able to convert 17Gb compressed images to 700+Mb compressed images.
   ```
   $ ls -ltrh
   total 19G
   drwx------ 2 root root  16K Aug  3 04:00 lost+found
   -rw-rw-r-- 1 root root  18G Aug  3 04:10 train.zip
   drwxrwxr-x 2 root root 2.9M Aug  3 04:46 train
   -rw-rw-r-- 1 root root 768M Aug  3 06:38 train224.zip
   drwxrwxr-x 2 root root 2.8M Aug  3 06:41 train224
   ```
 

# See

 1. [https://github.com/Kaggle/kaggle-api](https://github.com/Kaggle/kaggle-api)
 1. [https://fedoramagazine.org/edit-images-parallel-imagemagick/](https://fedoramagazine.org/edit-images-parallel-imagemagick/)
 1. [https://askubuntu.com/questions/271776/how-to-resize-an-image-through-the-terminal](https://askubuntu.com/questions/271776/how-to-resize-an-image-through-the-terminal)
 1. [http://www.imagemagick.org/Usage/thumbnails/#creation](http://www.imagemagick.org/Usage/thumbnails/#creation)

