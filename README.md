
# FrogGAN

FrogGan

Created and maintained by [Max Lamparth](https://github.com/maxlampe)

## Project status

This project is very much WIP and changes are constantly being made. The project is running on Google Colab and at the moment, only back-end modules are uploaded here.

The project is almost complete and an in detail presentation of the results will follow soon.

A final, minimized jupyter notebook will be uploaded as well.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Abstract / TL;DR

Art can be understood by everyone and transcends boundaries in society as well as between the complex 
topic of machine learning and the former. 
To this end, I implemented _deep convolutional generative adversarial networks_ (DCGANs) in _PyTorch_ from scratch 
to create images of drawn frogs in the style of the Austrian artist 
[ManBroDude](https://manbrodude.art/) aka [smalllebowsky](https://www.reddit.com/user/SmallLebowsky/) on _Reddit_. 

The target dataset was of small size (< 500 images with large variety) and to compensate, I pre-trained the networks on 
a larger dataset of japanese style comic faces (about 110k images) using transfer learning before adjusting all layers 
during fine-tuning. From multiple tried training adjustments, label smoothing yielded the best results. For 
hyper parameter optimization, _Frechet-Inception-Distance_ (FID) was used as objective value.
Experimental data augmentations (segmentation of object with and without background) together with regular data 
augmentations (randomized crops, flips, rotations, shearing, brightness and contrast fluctuations) have been applied.

- used datasets
- Show final result  

## Article

- introduction / state goal (credit original artist)
- emphasize challenge (target dataset < 500 images with large variety)
- state architecture used (future: 124 dim)
- Used techniques (DCGAN, pre-training, label smoothing, data augmentations, data pre-processing, FID)
- Tried without success (D noise, loss threshold training schedule)
- used libraries / codes (pytorch, optuna, FID code, pytorch example)
- used datasets
- hyper parameter chosen (optuna hp significance)
- show anime faces result and training parameters
- show frog training and results
- fid discussion
- augmentation discussion (show data pre-processing chain and different datasets)
- Show final result
- list of papers/articles
- link to reddit post

Reference
[This person does not exist](https://thispersondoesnotexist.com/)

