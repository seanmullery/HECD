from scipy import stats
import colorcet as cc
import pandas as pd
import numpy as np
from numpy import linspace
from scipy.stats.kde import gaussian_kde
from bokeh.plotting import figure, show, ColumnDataSource,  curdoc, output_file
from bokeh.models import CategoricalColorMapper, Legend,Select, HoverTool, Span, ImageURL, Plot, LinearInterpolator
from bokeh.palettes import Category20, Viridis, Category10
from bokeh.layouts import column, row
from bokeh.io import curdoc
from bokeh.models.widgets import Div
from bokeh.embed import components




def ridge(category, data, scale=2):
    return list(zip([category]*len(data), scale*data))



def update_plot(attr, old, new):
    mod = mod_select.value
    gt = reference_select.value
    oneGt =  zMeanDf[zMeanDf['GTFileName']==gt]
    src = oneGt[oneGt['ReColorMod']==mod]
    sub = oneGt[oneGt['Mod Number']==0]
    sub = sub[sub['WB Corrected']=='A']


    wb_line_val = sub['zScore'].mean()
    wb_line.location = wb_line_val
    d.title.text = "Modification Data zScores for Reference file " +gt+' with mod '+ mod
    sourceD.data = src

    sourceDis3U = ColumnDataSource(data=dict(x=x))
    oneGt =  df_reformat[df_reformat['GTFileName']==gt]
    oneGt = oneGt[oneGt['ReColorMod']==mod]
    for i, cat in enumerate(reversed(modFiles)):
        subdf = oneGt[oneGt['Mod Number']==int(cat)]
        sub = list(subdf[subdf['WB Corrected']=='N'].zScore)
        if len(sub) >= 3:
            pdf = gaussian_kde(sub)

            y = ridge(cat, pdf(x))

        else:
            y = ridge(cat, np.zeros(500))
        sourceDis3U.add(y, cat)
        sourceDis3.data = dict(sourceDis3U.data)


    oneGt =  df_reformat[df_reformat['GTFileName']==gt]
    oneGt = oneGt[oneGt['ReColorMod']==mod]
    for i, cat in enumerate(reversed(modFiles)):
        sourceDis4U = ColumnDataSource(data=dict(x=x))
        subdf = oneGt[oneGt['Mod Number']==int(cat)]
        sub = list(subdf[subdf['WB Corrected']=='A'].zScore)
        if len(sub) >=3:
            pdf = gaussian_kde(sub)
            y = ridge(cat, pdf(x))
        else:
            y = ridge(cat, np.zeros(500))
        sourceDis4U.add(y,cat)

        sourceDis4[int(cat)].data = dict(sourceDis4U.data)
        div_image.text = """<img src="http://smullery.pythonanywhere.com/static/imgs/{}" alt="div_image">""".format(gt)




df_reformat = pd.read_csv('../Data/reformatted_data.csv')


num_rows = len(df_reformat)

gt_mean = df_reformat['GT_zScore'].mean()
gt_mean = np.ones(num_rows)*gt_mean
df_reformat['GT_mean'] = gt_mean


GTs = list(df_reformat['GTFileName'].unique())
reColMods = list(df_reformat['ReColorMod'].unique())

SurNums = list(df_reformat['SurNum'].unique())

MARKERS = ['hex', 'circle_x', 'triangle']
color_mapper = CategoricalColorMapper(
    factors =GTs,
    palette=Category20[20],
)
reColMeans = []
for gt in (GTs):
    gtDF = df_reformat[df_reformat['GTFileName']==gt]
    reCols = gtDF['ReColFileName'].unique()
    reColMods = gtDF['ReColorMod'].unique()
    #['GTFileName', 'ReColFileName','Mod Number','WB Corrected','ReColorMod', 'SOTA', 'zScore','GT_zScore', 'Details', 'Hue', 'Chroma','a','b']
    i=0
    for reColMod in reColMods:
        reColMeans.append([gt,gt,0, 'N',reColMod,0,gtDF['GT_zScore'].mean(),gtDF['GT_zScore'].mean(),'Ground Truth Reference Image' ])
        sub = gtDF[gtDF['Mod Number']==0]
        reCol = sub['ReColFileName'].unique()

        '''if len(reCol)==0:
            #I don't think this happens any more
            reCol = 0.0
            reColMeans.append([gt,reCol,0, 'A',reColMod,0.0,gtDF['GT_zScore'].mean(),'None'])
        else:'''
        reCol=reCol[0]
        reColMeans.append([gt,reCol,0, 'A',reColMod,0.0,sub['zScore'].mean(),gtDF['GT_zScore'].mean(),'White Balance Corrected Reference File' ])
        i+=1



    for reCol in reCols:
        temp =  gtDF[gtDF['ReColFileName']==reCol]
        reColMod = temp['ReColorMod'].unique()[0]
        wbCor = temp['WB Corrected'].unique()[0]
        modNum = str(temp['Mod Number'].unique()[0])

        details = temp['Global Details'].unique()[0]
        hue = temp['Hue'].unique()[0]
        chroma = temp['Chroma'].unique()[0]
        a = temp['a'].unique()[0]
        b = temp['b'].unique()[0]
        reColMeans.append([gt,reCol,modNum, wbCor,reColMod, temp['SOTA'].unique()[0], temp['zScore'].mean(),gtDF['GT_zScore'].mean(), details, hue, chroma, a, b ])


gt_file = df_reformat['GTFileName']
zMeanDf = pd.DataFrame(reColMeans, columns=['GTFileName', 'ReColFileName','Mod Number','WB Corrected','ReColorMod', 'SOTA', 'zScore','GT_zScore', 'Details', 'Hue', 'Chroma','a','b'])
zMeanDf.to_csv('HumanAggregatedResults.csv')
wb = list(zMeanDf['WB Corrected'].unique())
hover = HoverTool(tooltips="""
    <div>
        <h3>@ReColFileName </h3>
        <div><strong> zScore: </strong>  @zScore</div>
        <div>@Details </div>
        <div>a=@a, b=@b, Hue=@Hue, Chroma=@Chroma </div>
        <div><img src="http://smullery.pythonanywhere.com/static/imgs/@ReColFileName" width="200" ></div>
    </div>

""")
p = figure(title = "Mean Z zscores", background_fill_color="#fafafa",y_range=GTs,x_range=[-2,2])
p.xaxis.axis_label = 'Mean zScore '
p.yaxis.axis_label = 'Ground Truth file name'


sourceP = ColumnDataSource(zMeanDf)
p.add_layout(Legend(), 'right')
p.cross(x='zScore',
         y='GTFileName',
         source=sourceP,
          color={'field':'GTFileName', 'transform':color_mapper},
          legend_group='GTFileName',
        )


p.add_tools(hover)

color_mapper2 = CategoricalColorMapper(
    factors =wb,
    palette=Category10[4],
)

oneGt =  zMeanDf[zMeanDf['GTFileName']=='015004_gt.jpg']
oneGt = oneGt[oneGt['ReColorMod']=='Shift segment hue']

modFiles =list( ['0','1','2','3','4','5','6','7','8','9','10','11','12','13'])
d  = figure(title = "Mean Z zscores", background_fill_color="#fafafa",y_range=modFiles,x_range=[-5,5] )
d.add_tools(hover)
d.xaxis.axis_label = 'Mean zScore '
d.yaxis.axis_label = 'Modification number'

palette=[cc.rainbow[i*12] for i in range (20)]
sourceD = ColumnDataSource(oneGt)
d.add_layout(Legend(), 'right')


leg = {'N': 'GT', 'A':'WB'}

d.circle(x='zScore',
         y='Mod Number',
         source=sourceD,
          color={'field':'WB Corrected', 'transform':color_mapper2},
          legend_group='WB Corrected',
          fill_alpha=1, size=6)
#
gt_line_val = zMeanDf['GT_zScore'].mean()

gt_line = Span(location=gt_line_val,
                              dimension='height', line_color='blue',
                              line_dash='dashed', line_width=1)
oneGt =  zMeanDf[zMeanDf['GTFileName']=='015004_gt.jpg']

sub = oneGt[oneGt['Mod Number']==0]

sub = sub[sub['WB Corrected']=='A']


wb_line_val = sub['zScore'].mean()
wb_line = Span(location=wb_line_val,
                              dimension='height', line_color='orange',
                              line_dash='dashed', line_width=1)

d.add_layout(gt_line)
d.add_layout(wb_line)


x = linspace(-5,+5, 500)

sourceDis3 = ColumnDataSource(data=dict(x=x))
oneGt =  df_reformat[df_reformat['GTFileName']=='015004_gt.jpg']

oneGt = oneGt[oneGt['ReColorMod']=='Shift segment hue']

for i, cat in enumerate(reversed(modFiles)):
    subdf = oneGt[oneGt['Mod Number']==int(cat)]
    sub = list(subdf[subdf['WB Corrected']=='N'].zScore)
    if len(sub) >= 3:
        pdf = gaussian_kde(sub)

        y = ridge(cat, pdf(x))
        sourceDis3.add(y, cat)
        d.patch('x', cat,  alpha=0.2, line_color="black", source=sourceDis3)
sourceDis4 = []
for i in range(0,14):
    sourceDis4.append( ColumnDataSource(data=dict(x=x)))
oneGt =  df_reformat[df_reformat['GTFileName']=='015004_gt.jpg']
oneGt = oneGt[oneGt['ReColorMod']=='Shift segment hue']
for i, cat in enumerate(reversed(modFiles)):

    subdf = oneGt[oneGt['Mod Number']==int(cat)]
    sub = list(subdf[subdf['WB Corrected']=='A'].zScore)

    if len(sub) >= 1:
        pdf = gaussian_kde(sub)

        y = ridge(cat, pdf(x))
        sourceDis4[int(cat)].add(y, cat)
        d.patch('x', cat, color='orange', alpha=0.2, line_color="black", source=sourceDis4[int(cat)])

d.y_range.range_padding=0.1
GTs_all = list.copy(GTs)
GTs_all=   list.copy(GTs) +['All']
GTs_all.reverse()
print(GTs)


disFig = figure(title='Distribution of z-scores', y_range=GTs_all, width=500)
sourceDis = ColumnDataSource(data=dict(x=x))
subdf = df_reformat
sub = list(subdf.zScore)
pdf = gaussian_kde(sub)
y = ridge('All', pdf(x))
sourceDis.add(y, 'All')
disFig.patch('x', 'All', color='red', alpha=0.8, line_color="black", source=sourceDis)
for i, cat in enumerate(reversed(GTs)):
    print(cat)
    subdf = df_reformat[df_reformat['GTFileName']==cat]

    sub = list(subdf.zScore)

    pdf = gaussian_kde(sub)
    y = ridge(cat, pdf(x))
    sourceDis.add(y, cat)
    disFig.patch('x', cat, color=palette[i], alpha=0.8, line_color="black", source=sourceDis)





disFig.y_range.range_padding=0.1
disFig.add_layout(gt_line)
disFig.xaxis.axis_label = 'z-score'
mod = 'Shift segment hue'
gt = '015004_gt.jpg'
reference_select = Select(value=gt, title='Ground Truth Image', options=GTs)
mod_select = Select(value=mod, title='Mod Type', options=sorted(reColMods))
mod_select.on_change('value', update_plot)
reference_select.on_change('value', update_plot)


div_image = Div( text="""<img src="http://smullery.pythonanywhere.com/static/imgs/{}" alt="div_image">""".format(reference_select.value), width=500, height=500)

wb_fig = figure(title='Distribution of z-scores over full dataset' ,y_range = ['0','1'], width=500)
sourceWBfig = ColumnDataSource(data=dict(x=x))

subdf1 = df_reformat#[df_reformat['WB Corrected']=='N']
sub = list(subdf1.zScore)
pdf = gaussian_kde(sub)

xr = x[x>gt_line_val]

y = ridge('0', pdf(x))
sourceWBfig.add(y, '0')
wb_fig.patch('x', '0', color='green', alpha=0.8, line_color="black", source=sourceWBfig)

'''sourceWBfig = ColumnDataSource(data=dict(x=x))
subdf2 = df_reformat[df_reformat['WB Corrected']=='A']
sub = list(subdf2.zScore)
pdf = gaussian_kde(sub)
y = ridge('0', pdf(x))
sourceWBfig.add(y, '0')
wb_fig.patch('x', '0', color='red', alpha=0.3, line_color="black", source=sourceWBfig)'''

wb_fig.add_layout(gt_line)
wb_fig.add_layout(Legend(), 'right')
wb_fig.xaxis.axis_label = 'z-core'




#wb_line_val1 = subdf1['zScore'].mean()
'''wb_line1 = Span(location=wb_line_val1,
                              dimension='height', line_color='orange',
                              line_dash='dashed', line_width=1)
'''#wb_fig.add_layout(wb_line1)


ab_space = figure(y_range = (-100,+100),x_range=[-100,100],  height=600, width=600)
ab_space.xaxis.axis_label = 'a* '
ab_space.yaxis.axis_label = 'b*'
zScore_size = LinearInterpolator(
    x=[-2,+1.3 ],
    y = [1,30]
)
ab_space.add_tools(hover)
ab_space.circle(x='a',
         y='b',
         source=sourceD,
          color={'field':'WB Corrected', 'transform':color_mapper2},
          legend_group='WB Corrected',
          fill_alpha=0.3, size={'field':'zScore', 'transform':zScore_size})

sota_df = zMeanDf[zMeanDf['SOTA']==1]
gt_sota_line_value = sota_df['GT_zScore']

sota_names = ['Colourised by Iizuka', 'Colourised by Larsson', 'Colourised by DeOldify', 'Colourised by PhotoShop', 'Colourised by Zhang1', 'Colourised by Zhang2' ]

sota_fig = figure(title = "Distributions of SOTA colourisation algorithms", background_fill_color="#fafafa",y_range=sota_names,x_range=[-5,5] )
sota_fig.add_tools(hover)
sota_fig.xaxis.axis_label = 'z-score '
#sota_fig.yaxis.axis_label = 'Modification number'

palette=Category20[20]
source_sota_dis = ColumnDataSource(data=dict(x=x))
source_sota = ColumnDataSource(sota_df)
sota_fig.add_layout(Legend(), 'right')
#print(oneGt.head() )



sota_fig.circle(x='zScore',
         y='Details',
         source=source_sota,
          #color={'field':'WB Corrected', 'transform':color_mapper2},
          #legend_group='Details',
          fill_alpha=1, size=6)
#
gt_line_val = zMeanDf['GT_zScore'].mean()


gt_line = Span(location=gt_line_val,
                              dimension='height', line_color='blue',
                              line_dash='dashed',line_alpha=0.5, line_width=1)


sota_fig.add_layout(gt_line)

xr = x[x>gt_line_val]
for i, cat in enumerate(reversed(sota_names)):

    subdf = zMeanDf[zMeanDf['Details']==cat]

    sub_sota = list(subdf.zScore)

    pdf = gaussian_kde(sub_sota)
    #print(f'{cat} : {np.sum(pdf(xr))/np.sum(pdf(x))*100} : ')
    y = ridge(cat, pdf(x), scale=1)
    source_sota_dis.add(y, cat)
    sota_fig.patch('x', cat, color=palette[i*3], alpha=0.8, line_color="black", source=source_sota_dis)
sota_fig.y_range.range_padding=0.2
layout = column(row( column(d,mod_select,reference_select), ab_space, div_image), row(p,disFig, wb_fig, sota_fig))
curdoc().add_root(layout)





