# Image Inpainting Detection through Artificial Intelligence Techniques: readme file
 An official implementation code for my thesis "Image Inpainting Detection through Artificial Intelligence Techniques"
 
 # Abstract 
Image inpainting is the process of repairing an area in an image, from which a part of the semantic information is missing and consequently there is a lack of semantic continuity. Image inpainting was initially designed to effectively repair damaged areas in images. Ηowever, it was quickly used for the purpose of forgery and deception. In recent years, methods of applying image inpainting through artificial intelligence techniques came up and achieved high quality results, producing images where the presence  of inpainting is almost impossible to detect with the human eye. Therefore, it is of critical importance to develop a method that will detect the affected areas in inpainted image. For this reason, the present thesis focuses on the study of image inpainting detection methods and the implementation of an artificial neural network capable of detecting areas where an image has been tampered by inpainting. A total of eight convolutional neural networks, based on two state of the art architectures, were trained and tested. The training process was based on two configurations sets (10 and 50 epochs respectively) adopting the binary cross entropy (BCE) as a loss function. Furthermore, it was also  studied to what extent the use of a training dataset consisting of images that have been inpainted in semantic areas helps more than one whose images have been inpainted in random-form areas helps more in the image inpainting detection. For this reason, two training sets were created. The first one, is consisting of images with random-form inpainting masks, while the second one is consisting of images with semantic masks (objects). To evaluate the trained models, a test set consisting of both forms of masks were created in order to give an objective interpretation of the results. The aim is to train a model, capable of producing a predicted mask Mo as output, given an image I as input. Finally, the two commonly used pixel-wise metrics, IoU and AUC, were adopted to evaluate the performance. The metrics were calculated by using the ground truth Mg and the predicted mask Mo and by making a 1-1 comparison of their corresponding pixels. Τhe study proved that, models trained with a set of images that have been tampered in random areas (random masks) achieve better results comparing to models that were trained with a train set of images that have been tampered in semantic areas (semantic masks).

 # Depedency 
 - Python 3
 - Torch 1.81

 # Depedency 
To train or thes the network:
```
python main.py {train, test}
```

Then the network will detect the images in the ./Demo Dataset/demo input/inp_images/ and save the results in the ./Demo Dataset/demo output/pr_masks/ directory.


