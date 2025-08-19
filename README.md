# This repository contains both the ​​image fusion dataset​​ and the ​​complete code​​ for generating fused images. You can either:

•​​Option 1​​: Directly download the provided fused image dataset to train your model.

•​​Option 2​​: Generate a ​​custom fusion dataset​​ using our code, with adjustable parameters including:
  •Window length
  •Step size
  •Range gate selection
  •Image dimensions

# How to use image dataset

You can use the ​​NTFD​​.m script to directly generate the corresponding images from the ​​IPIX dataset​​. The outputs of the three methods, along with the fused images, will be saved in ​​three separate folders​​.

•​​Parameter Adjustment​​: The ​​step size​​ and ​​window length​​ can be modified in the parameter selection section of the ​​NTFD​​ file.

•​​Save Path​​: The output directory can also be configured in the corresponding section of the file.

•Range Gate Selection​​: For different range gates, please refer to the ​​IPIX official website​​ (we used the ​​Dartmouth 1993 dataset​​):

🔗 http://soma.ece.mcmaster.ca/ipix/dartmouth/datasets.html

For the ​​Yantai dataset​​, use the ​​NTFD_yantai.m​​ script, which follows the ​​same structure​​ as described above. The raw Yantai dataset can be downloaded from:

🔗 https://radars.ac.cn/web/data/getData?dataType=DatasetofRadarDetectingSea

# ​​How to Train the EfficientNet Model​​
​​Requirements:​​

•​​Python 3.10​​ 

•SE_efficient.py​​ 

​​Steps to Train the Model:​​
Run the Script: ​​which will:

•Load the fused image dataset

•Train the EfficientNet model end-to-end

•Output classification results (e.g., accuracy, confusion matrix)


# Image dataset
This repository holds the fused image dataset obtained by the three methods used in the paper, you can download our pre-generated image dataset directly via this link: https://pan.baidu.com/s/1wpeyyCUKfFVbxMQ2E5iFig?pwd=q91d.  passwoard:q91d


