
# CLAIRE-ROP: Rapid Overlapped Partition-based Deformable Image Registration 

![Fig4](https://github.com/UniHD-CEG/CLAIRE-ROP/assets/62182727/e9baf1c8-b277-408f-9f7d-0c74848f35e4)




Deformable Image Registration (DIR) is a complex process that involves calculating a deformation field to map one image onto another. This field describes the spatial transformation between the two images using mathematical algorithms. Due to its complexity, DIR requires substantial computational resources and time, posing challenges for time-sensitive medical applications like image-guided radiation therapy.

Our innovative registration method addresses this challenge by significantly reducing processing time without sacrificing registration accuracy. We achieve this by partitioning lung images into multiple partitions, enabling separate registration for each partition. These partitions are efficiently distributed across dedicated GPUs, eliminating communication overhead and enhancing scalability. This partition-based approach is scalable, allowing us to adapt seamlessly to an increased number of available GPUs.



- [Registration framework](https://github.com/andreasmang/claire)

- [Dataset](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html)

- [Tool for segmentation](https://github.com/wasserth/TotalSegmentator )


If you encounter any problems, have inquiries, or wish to provide feedback, please feel free to reach out to us via email (vahdaneh_kiani@ziti.uni-heidelberg.de).


