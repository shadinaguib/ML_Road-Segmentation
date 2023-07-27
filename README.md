# ML_RoadSegmentation

Research project on semantic segmentation of aerial images. The goal is to be able to separate two classes, Background and Road for satellite images.  

The structure of the image dataset and their groundtruth can be seen in the following image:

![Alt text](/Example.png?raw=true "Dataset image / Corresponding Groundtruth")


## Prerequisites
To be able to run the full project and train the different models the following packages need to be installed in the environement :

- Pytorch - 1.10 or above
- numpy - 1.21.5
- imgaug - 0.4.0
- scikit-image - 0.19.0
- scikit-learn - 1.0.1  
- Pandas - 1.3.5

## Structure of the Repository

<pre>
.  
├── data                    # Dataset for training/ testing the model  
├── report                  # Report of the project  
├── src                     # Source files
│    └── helpers            # Helpers for the implementation of the code  
│    └── models             # Savestate of the best models  
└── README.md  
</pre>

## Dataset

To run the model or to train it, the data has to be structured in a specific way with respect to the root directory where the 'Run.py' file is situated. 

<pre>
.  
├── ...  
├── data  
│   ├── training  
│           └── groundtruth         # Groundtruth of the training images (400x400)  
│           └── images              # Training images (400x400 RGB)  
│   └── test                        # Test images (608x608 RGB)  
└── ...  
</pre>

## Model used

![Net](https://user-images.githubusercontent.com/26313021/151218916-fc29920a-4dd3-43a0-9a9c-f678f34cfc08.png)

## Run the code 
In order to run the best model that was implemented the following command has to be sent in a terminal: i
```
python run.py
```
This has to be done in the 'src' folder with the data organised as mentioned above.

## Results
With the optimal model, one of the Unet versions the following results are obtained:


|           | Validation F1-score | Validation accuracy   | Test F1-score | Test Accuracy |
|:---------:|:-------------------:|:---------------------:|:-------------:|:-------------:|
| Unet-Beta |        -----        |         0.963         |     0.903     |     0.947     |

The prediction on an image is :

![Alt text](/results.png?raw=true "Dataset image / Corresponding Groundtruth")
## Authors

- Shadi Naguib [@shadinaguib](https://github.com/shadinaguib)
- Léo Alvarez [@leoalz](https://github.com/leoalz)
- Borja Rodriguez Mateos [@borjarc](https://github.com/borjarc)
