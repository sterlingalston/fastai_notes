[Practical Deep Learning for Coders - 1: Getting started](https://course.fast.ai/Lessons/lesson1.html)

![](assets/2026-01-21-19-41-35-image.png)

Is it a bird notebook

https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data

### DataBlock

- gives fastai all information it needs for computer vision model

- ![](assets/2026-01-26-18-56-18-image.png)

- easy to check because can look at it

- dataloader: pytorch iterates through to grab data at a time; feed the training algorithm with all images (batch) at once

- dls.show_batch: show me example of a batch of data passing into model

- no progress bar!!!!!

![](assets/2026-01-26-18-41-57-image.png)

- docs.fast.ai

- ![](assets/2026-01-26-19-10-31-image.png)

- ![](assets/2026-01-26-19-24-19-image.png)

- a *learner* is something that combines a model (neural network function to train) and data to train it with; nn function is the model (resnet18)

- timm.fast.ai has largest collection of image models
  
  - documentation:
  
  - ![](assets/2026-01-26-19-26-23-image.png)

- resnet should be fine for what you'd like to do
  
  - already trained this model to recognize 1 million images and 1000 different types and made weight available

- *fine_tune* takes pre-trained rates and adjusts to teach model differences between dataset and what it was originally trained for

### Book

[GitHub - fastai/fastbook: The fastai book, published as Jupyter Notebooks](https://github.com/fastai/fastbook)

repo for all the notebooks

view notebook readonly

https://nbviewer.org/

Last url:
[Practical Deep Learning for Coders - 1: Getting started](https://course.fast.ai/Lessons/lesson1.html)

Will be using PyTorch

TensorFlow is dying

![](assets/2026-01-26-18-34-24-image.png)

Fastai is built ontop of PyTorch

Going to be using Kaggle https://www.kaggle.com/work/code

### Segmentation

- color every pixel according to what it is
  
  - ![](assets/2026-01-30-15-53-17-image.png)

- left was manually plugged in and right is predicted

- ![](assets/2026-01-30-16-06-05-image.png)

- DataBlocks are intermediate flexible

- SegmentationDataLoaders

### Tabular

- ![](assets/2026-01-30-16-08-35-image.png)

- ![](assets/2026-01-30-16-08-50-image.png)

- URLs.ADULT_SAMPLE -- based on research

- ![](assets/2026-01-30-16-09-42-image.png)

- ![](assets/2026-01-30-16-13-46-image.png)

- fit_one_cycle -- no pre-trained model for what we want

### Collaborative filtering

- ![](assets/2026-01-30-16-14-43-image.png)

- basis for recommendations systems

- take dataset to say which users used some products; used to guess what other products users might like by looking at similar users
  
  - not demographics, but people who like the same products
  
  - i.e. if you used Spotify, music genre, etc. it'll play similar music

- ![](assets/2026-01-30-16-16-36-image.png)

- ![](assets/2026-01-30-16-21-35-image.png)

- ![](assets/2026-01-30-16-23-59-image.png)

- TURN NOTEBOOK INTO PRESENTATION SLIDES: rise.readthedocs.io

- ![](assets/2026-01-30-16-24-40-image.png)



### What can deep learning do?

![](assets/2026-01-30-16-30-21-image.png)

![](assets/2026-01-30-16-31-38-image.png)



### What's going on?

![](assets/2026-01-30-16-35-28-image.png)

![](assets/2026-01-30-16-35-55-image.png)

![](assets/2026-01-30-16-37-06-image.png)

![](assets/2026-01-30-16-39-29-image.png)
