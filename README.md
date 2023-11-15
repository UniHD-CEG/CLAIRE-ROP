
# CLAIRE-ROP

<p align="center">
 <img width="623" alt="CLAIRE-ROP" src="https://github.com/UniHD-CEG/CLAIRE-ROP/assets/62182727/18f072b9-b006-4196-92f0-efffa06a684a">

</p>


 <div align="justify">
Deformable Image Registration (DIR) is a complex process that involves calculating a deformation field to map one image onto another. This field describes the spatial transformation between the two images using mathematical algorithms. Due to its complexity, DIR requires substantial computational resources and time, posing challenges for time-sensitive medical applications like image-guided radiation therapy.

Our method addresses this challenge by significantly reducing processing time without sacrificing registration accuracy. We achieve this by partitioning lung images into multiple partitions, enabling separate registration for each partition. These partitions are efficiently distributed across dedicated GPUs, eliminating communication overhead and enhancing scalability. This partition-based approach is scalable, allowing us to adapt seamlessly to an increased number of available GPUs.
 </div>

Registration framework:

- [CLAIRE](https://github.com/andreasmang/claire)

Dataset:

- [4DCT DIR-Lab](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html)

Required tools:

  - [SimpleITK](https://pypi.org/project/SimpleITK/)   
  - [NiBabel](https://nipy.org/nibabel/index.html)
  - [Totalsegmentator](https://github.com/wasserth/TotalSegmentator )


If you encounter any problems, have inquiries, or wish to provide feedback, please feel free to reach out to us via email (vahdaneh_kiani@ziti.uni-heidelberg.de).


