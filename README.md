# MELUS Project
Lung ultrasound images synthesis using DCGAN

## GAN Training

This code will train for 1000 epochs a DCGAN from images provided by a specified folder/directory (`TRAIN_IMAGES_DIRECTORY`) and save a certain number of fake images (`AMOUNT`) wherever the user needs (`OUTPUT_FAKE_IMAGES_DIRECTORY`).

(*Note: It will also generate a CSV file with the loss and accuracy results.*)

Execution (WSL): 

```
python3 GAN_train.py TRAIN_IMAGES_DIRECTORY OUTPUT_FAKE_IMAGES_DIRECTORY AMOUNT
```

**Example**

```
python3 GAN_train.py DB_LUS4MELUS/tif/ generated_test 3000
```

## FID calculation

This code will calculate the FID between 2 set of images, so you can compare the generated images within `FAKE_IMAGES_DIRECTORY` against the real ones in `REAL_IMAGES_DIRECTORY`.

(*Note: The resulting FID number will be displayed on the terminal*)

Execution (WSL): 
```
python3 image_evaluation.py REAL_IMAGES_DIRECTORY FAKE_IMAGES_DIRECTORY
```

**Example**

```
python3 image_evaluation.py DB_LUS4MELUS/tif/ generated_test
```

## Loss and Accuracy plot

This code will just save two PNG images of the Loss and Accuracy plot using the CSV generated on the training process.

Execution (WSL): 
```
python3 results_plot.py
```

## Additional Notes

1. "Fake images" must start with "generated_image".
2. version, hw, etc (COMPLETAR)
