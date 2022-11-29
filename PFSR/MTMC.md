

## Abstract

DukeMTMC (Multi-Target, Multi-Camera) dataset is a large scale database of images captured by researchers in Duke’s Computer Science Department using Duke’s surveillance camera system in 2014. It was previously one of the largest and most frequently used datasets for video re-identification, a process that uses computer vision and artificial intelligence to identify and track targeted individuals within live video feeds. Duke researchers claim that the project, funded by the US Army Research Office and National Science Foundation, was originally intended to improve systems for motion detection of objects in video, regardless of “whether [the objects] are people, cars, fish or other” (Dr. Carlo Tomasi, source Duke Chronicle), and was available for download on the Duke Computer Vision research website.

The dataset was recently taken down, after it came under fire in April 2019 for ethical and privacy violations following an expose by researcher Adam Harvey that prompted Duke’s Institutional Review Board to revisit the terms of the collection and use of the image data. Investigation reveals that the collection and making public of the data (1) indicates a “significant deviation from the [IRB] approved protocol” (Michael Schoenfeld) and (2) the dataset has irreversibly been downloaded and implemented in computer vision, body tracking, and facial recognition systems by academic, governmental and military institutions across the globe. Significantly, Harvey has traced the dataset to research papers published by Chinese companies SenseNets and SenseTime Group Limited, both associated with contributing to the surveillance techniques used by the Chinese military to target and monitor Uyghur populations in rural China. As Harvey points out, this implementation isn’t all that different from the original motivation of the Duke researchers, who published a subsequent paper in 2017 titled “Tracking Social Groups Within and Across Cameras.” When scrolling through the dataset, which contains images of students entering and leaving academic buildings and places of worship, one might see how this information could easily be weaponized to target and discriminate against marked individuals. In Harvey’s analysis, this is but one example of “an egregious prioritization of surveillance technologies over individual rights” (Megapixels).

While this dataset is no longer available through official Duke affiliated websites, it is without doubt that it exists in many databases around the world. I easily recovered an edited version of the dataset from BaiduYun, where it was uploaded for a project titled “Beyond Part Models: Person Retrieval with Refined Part Pooling.” Of interest to me is thinking about where exactly the “human rights” that Harvey discusses are violated within this “information supply chain”-- is it a transgression of the individuals who had their image captured that day in 2014, who’s faces exist as blurry barely discernable pixel clusters in this evasive data trove? Or can we claim that the human rights of the surveilled Uyghurs are certainly much more harshly afflicted in this situation?

It is difficult to locate a direct chain of cause and effect in this circulation of information, and it perhaps the dynamic of circulation itself that inductively made this situation possible. Is the total erasure of individual rights in the interest of institutional or economic rationality necessary in order for this material to be extracted, compiled, circulated and adapted the way that it has? Does one have a right to their data? What about an image of their face? In a much more literal manner I wonder what it would look like to attempt to restore subjectivity back into this dataset where it has otherwise been compressed and cropped away? I propose to take the low resolution facial images contained in the dataset and process them using a 3D Face Reconstruction algorithm in order to produce 3D face models that I will then 3D print. While the information in the dataset certainly isn’t detailed enough to produce accurate renderings of the individuals captured, the 3D prints will be a symbolic representation of their identity as it exists as a datapoint within the current and future systems that use the DukeMTMC dataset.

### Technical Requirements

I've decided on a 2 step method to process the MTMC dataset that will consist of first, hyper-resolving the images so that facial features are discernible, and second, extrapolating a 3D model from the processed images. My eventual goal is to 3D print these face objects in full color.

#### Step 1:

I used Progressive Face Super-Resolution, a repository that uses Pytorch in order to super resolve facial features from low dimensional images. The model is trained on celebA, and is designed to reconstruct faces from images without distortion using a progressive training method that gradually increases image size.

In order to use this repository, I needed access to CUDA 9.0 or higher, so I learned how to use the Duke Computing Cluster's SLURM Queue. I successfully set up a conda environment to run the test images on the celebA data, but was running into problems when I tried to load in images from MTMC. I decided to use Google Colab to more easily visualize the code execution so that I could troubleshoot. I customized the Colab notebook provided by one of the authors of the PFSR repo. [https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/PFSR/main.py]

CelebA Examples created using PFSR model on DCC:

![](https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/PFSR/test/celebA/2_results.jpg)

This script essentially scales an image and processes it through a series of optimized tensors trained on celebA in order to output a higher dimensional and higher resolution image. Although it performs pretty well on images that are already relatively clear, my MTMC images were really blurry and needed to be preprocessed to crop specifically the part of the image where the face was located. I wrote a simple python script to batch crop the images, and selected the best ~200 to test. [https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/PFSR/test/img-crop.py]

As expected, the PFSR model hallucinated a lot of information that doesnt remotely resemble a face from the MTMC data. I spent a long time tweaking the different parameters of the model, the number of iterations, and the dimensions of the images I fed in with not too much improvement. However, my goal was never to hallucinate realistic facial images, so I dont really care about this. It would be interesting in the future to attempt to train a model on custom data that would have a greater ability to generate more realistic faces from blurry or indiscernible pixel data.

Test Results using random images of me:

![](https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/PFSR/test/results/cropy.png)
![](https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/PFSR/test/results/89866147-08ad-4fe2-8944-b5711f4ce480.png)

Test Results using MTMC Images:

![closer crop version](https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/PFSR/PFSR-MTMC/img/test/0_predictedresults%203.png)
![larger crop version](https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/PFSR/PFSR-MTMC/img/grid/0_predictedresults-1%203.png)

I ended up with way many useless images from all my testing. A few examples are uploaded in [https://github.com/rebeccauliasz/PFSR-DukeFace/tree/master/PFSR/PFSR-MTMC/img/grid]

### Step 2:

I was skeptical at this point about whether the images I had generated in fact looked enough like faces to be properly processed by a network designed to generate 3D face objects from 2D images. I planned to use 3DDFA (Face Alignment in Full Pose Range: A 3D Total Solution)[https://github.com/cleardusk/3DDFA], a repo written using pytorch trained that makes use of models from the dlib library. This code can be modified to run on a CPU, so I made a conda env on my machine to set up testing.

![](https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/3DDFA/test/self_3DDFA.jpg)
![](https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/3DDFA/test/self_pose.jpg)
![](https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/3DDFA/test/self_0_paf.jpg)
![](https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/3DDFA/test/self_pncc.png)

Once I got this working, I needed to preprocess the images I generated from the PFSR. The model had output image grids, so I wrote a script to batch crop these into individual JPGs. [https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/PFSR/test/img-crop-grid.py]

Here's a sample of them. I generated a couple hundred:

![](https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/3DDFA/3DDFA-MTMC/18crop_0_crop.jpg)
![](https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/3DDFA/3DDFA-MTMC/34crop_0_crop.jpg)
![](https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/3DDFA/3DDFA-MTMC/49crop_0_crop.jpg)

I ran the script to process these images. The dlib model could not always recognize a face, and sometimes found faces where there weren't any:

![](https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/3DDFA/3DDFA-MTMC/18crop_pose.jpg)
![](https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/3DDFA/3DDFA-MTMC/34crop_pose.jpg)
![](https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/3DDFA/3DDFA-MTMC/49crop_pose.jpg)

Depth mask:

![](https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/3DDFA/3DDFA-MTMC/18crop_depth.png)
![](https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/3DDFA/3DDFA-MTMC/34crop_depth.png)
![](https://github.com/rebeccauliasz/PFSR-DukeFace/blob/master/3DDFA/3DDFA-MTMC/49crop_depth.png)


The results are weird, but I wish the model didnt attempt so hard to find an "accurate face" or could tolerate more distortion. I tried another model, found at (https://cvl-demos.cs.nott.ac.uk/vrn/) "3D Face Reconstruction from a Single Image", which uses CNN regression and tends to produce much more warped outputs.

Here are some of the OBJ examples: [https://rebeccauliasz.github.io/PFSR-DukeFace/]

Note: This file is large and may take time to load!
