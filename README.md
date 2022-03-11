# HECD (Human Evaluated Colourisation Dataset)

## This is a dataset of 20 scenes with 65 recolourisations of each scene.

Auto-colourisation is an ill-posed problem with many possible colourisations for a given grey-scale prior.
The current method for training deep neural networks for colourisation is to take any natural image dataset and convert it to a luminance-chrominance colour space. The luminance is then the prior and the two chrominance channels represent the ground truth target that the model must learn to predict. This allows only a single plausible colourisation for a grey-scale prior.

We have created at dataset of 20 scenes with 65 re-colourisations or each scene for a total of 20+1300=1320 images.
We then crowd sourced human opinion of the naturalness of the colourisation via the Amazon Mechanical Turk.  
The 1320 images can be found in the folder HECDImages.
The associated average score for each recolourisation can be found in Data/mean_zscores.csv. This file along with the image data can be used to compare with objective methods.

The details of what re-colourisations are included in the set can be found in Data/recolour_mod_details.csv and are explained in our paper.

The Raw data taken from the AMT can be found in Data/raw_data.csv.
Some problematic data is removed as explained in the paper and the cleaned data can be found at Data/clean_raw_data.csv.
As explained in the paper, the data is then processed and the processed data can be found in Data/reformatted_data.csv.


The bokeh_app folder, contains an interactive app to help navigate the statistics of the dataset.

To run this you will have to [install bokeh](https://docs.bokeh.org/en/latest/docs/first_steps.html#first-steps)

And then run the following command from inside the bokeh_app folder locally.

    bokeh serve --show bokeh_analysis.py

It should run in your web browser. It will require the files in Data, so make sure they are also available locally in the same structure as above.
