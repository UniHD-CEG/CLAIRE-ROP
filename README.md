
# CLAIRE-ROP: Rapid Overlapped Partition-based Deformable Image Registration 

<img width="930" alt="CLAIRE-ROP" src="https://github.com/UniHD-CEG/CLAIRE-ROP/assets/62182727/054ef0fc-411a-4347-9437-96e80ce0fa51">




Deformable Image Registration (DIR) is a complex process that involves calculating a deformation field to map one image onto another. This field describes the spatial transformation between the two images using mathematical algorithms. Due to its complexity, DIR requires substantial computational resources and time, posing challenges for time-sensitive medical applications like image-guided radiation therapy.

Our method addresses this challenge by significantly reducing processing time without sacrificing registration accuracy. We achieve this by partitioning lung images into multiple partitions, enabling separate registration for each partition. These partitions are efficiently distributed across dedicated GPUs, eliminating communication overhead and enhancing scalability. This partition-based approach is scalable, allowing us to adapt seamlessly to an increased number of available GPUs.



- [Registration framework](https://github.com/andreasmang/claire)

- [Dataset](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html)

- [Tool for segmentation](https://github.com/wasserth/TotalSegmentator )


If you encounter any problems, have inquiries, or wish to provide feedback, please feel free to reach out to us via email (vahdaneh_kiani@ziti.uni-heidelberg.de).


