# HECD (Human Evaluated Colourisation Dataset)

## This is a dataset of 20 scenes with 65 recolourisations of each scene.

Auto-colourisation is an ill-posed problem with many possible colourisations for a given grey-scale prior.
The current method for training deep neural networks for colourisation is to take any natural image dataset and convert it to a luminance-chrominance colour space. The luminance is then the prior and the two chrominance channels represent the ground truth target that the model must learn to predict. This allows only a single plausible colourisation for a grey-scale prior.

Paper https://arxiv.org/abs/2204.05200

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
This is still a bit of a work in progress. In the legend A stands for images that are modifications from the ground truth while N stands for images that are modified from the White balanced corrected version.

  ![image](Bokeh_Screenshot.png)
  
## Comparing Human opinion with objective measures
The python code to compare the HECD results with many objective measures is contained in CompareHumanWithObjective.py which will output the Spearman and Kendal Correlation tables in CSV format. Note that every second column gives the p-value for the correlation score in the previous column.
Because the LPIPS method takes a long time to run (particularly for VGG) it is run from a separate file called CompareHumanWithLpips.py. This will also create separate csv files. Note that you may change the network 'vgg' or 'alex' depending on which network you wish LPIPS to be based.







If you use this database or its results please cite as follows

@misc{mullery2022HECD,
      title={Human vs Objective Evaluation of Colourisation Performance}, 
      author={Seán Mullery and Paul F. Whelan},
      year={2022},
      eprint={2204.05200},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
