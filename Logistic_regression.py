# LOGISTIC_REGRESSION.PY
# Nathaniel Heatwole, PhD (heatwolen@gmail.com)
# Uses logistic regression to predict probability of passing an exam using number of hours studied, using both sklearn and statsmodels
# Training data: synthetic data for 20 students (from https://en.wikipedia.org/wiki/Logistic_regression#Example)

import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from colorama import Fore, Style
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from matplotlib.backends.backend_pdf import PdfPages

time0 = time.time()
ver = ''  # version (empty or integer)

topic = 'Logistic regression'
topic_underscore = topic.replace(' ','_')

#--------------#
#  PARAMETERS  #
#--------------#

logit_methods = ['sklearn', 'statsmodels']

# regression variables
x_var = 'hours'
y_var = 'passed'

decimal_places = 2

#-----------------#
#  TRAINING DATA  #
#-----------------#

# lists (20 students)
hours_studied = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5]
passed_exam = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]

# dataframe
exam = pd.DataFrame()
exam[x_var] = hours_studied
exam[y_var] = passed_exam

total_obs = len(exam)

# regression data
x = pd.DataFrame(exam[x_var])
y = pd.DataFrame(exam[y_var])

#---------------#
#  REGRESSIONS  #
#---------------#

# initialize
preds = exam.copy(deep=True)
logits_summary = pd.DataFrame()
logits_summary.index = ['total obs', 'slope', 'intercept', 'accuracy', 'log-likelihood', 'avg probability']

# loop over logit methods
for m in range(len(logit_methods)):
    logit_method = logit_methods[m]
    suffix = ' m' + str(m + 1)  # 'm' stands for 'model' (1 or 2)
    
    # run logit
    if logit_method == 'sklearn':
        logit_method_1 = LogisticRegression()
        logit_fit_1 = logit_method_1.fit(x, np.resize(y, total_obs))
        preds['prob pass' + suffix] = logit_fit_1.predict_proba(x)[:,1]  # predicted probability
    elif logit_method == 'statsmodels':
        logit_method_2 = sm.Logit(y, sm.add_constant(x))  # 'add_constant' instructs the regression to fit an intercept
        logit_fit_2 = logit_method_2.fit()
        preds['prob pass' + suffix] = logit_fit_2.predict()  # predicted probability

    # predicted class
    preds['y pred' + suffix] = round(preds['prob pass' + suffix])
    preds['correct' + suffix] = [int(preds[y_var][i] == preds['y pred' + suffix][i]) for i in preds.index]  # correct indicator
    preds['false pos' + suffix] = [int(preds[y_var][i] == 0 and preds['y pred' + suffix][i] == 1) for i in preds.index]  # false positive indicator

    # regression parameters
    if logit_method == 'sklearn':
        slope = logit_method_1.coef_[0][0]
        intercept = logit_method_1.intercept_[0]
        accuracy = logit_fit_1.score(x, y)
    elif logit_method == 'statsmodels':
        slope = logit_fit_2.params.iloc[1]
        intercept = logit_fit_2.params.iloc[0]
        accuracy = sum(preds['correct' + suffix]) / total_obs
    log_lik = -log_loss(y, preds['prob pass' + suffix], normalize=False)  # log-likelihood (maximum is sought)
    avg_prob = np.mean(preds['prob pass' + suffix])  # average predicted probability

    slope = round(slope, decimal_places)
    intercept = round(intercept, decimal_places)
    log_lik = round(log_lik, decimal_places)
    avg_prob = round(avg_prob, decimal_places)

    # save
    logits_summary[logit_method] = [str(total_obs), slope, intercept, accuracy, log_lik, avg_prob]
    del logit_method, suffix, slope, intercept, accuracy, log_lik, avg_prob

#---------#
#  PLOTS  #
#---------#

# parameters
title_size = 11
axis_labels_size = 8
axis_ticks_size = 8
legend_size = 8
plot_point_size = 8
line_width = 1.25
p_ticks = 0.2
x_max_plot = np.ceil(max(x[x_var]))
x_margin_plot = 0.25
y_margin_plot = 0.05

# generate plot
fig1 = plt.figure()
plt.title(topic + ' - exam performance', fontsize=title_size, fontweight='bold')
plt.plot(x, preds['prob pass m1'], color='red', linewidth=line_width, label='sklearn', zorder=5)
plt.plot(x, preds['prob pass m2'], color='blue', linewidth=line_width, label='statsmodels', zorder=10)
plt.scatter(x, y, color='black', marker='*', s=plot_point_size, label='training data', zorder=15)
plt.xlabel('Hours studied', fontsize=axis_labels_size)
plt.ylabel('Pass probability', fontsize=axis_labels_size)
plt.legend(loc='center right', fontsize=legend_size, facecolor='white', framealpha=1)
plt.xlim(-x_margin_plot, x_max_plot + x_margin_plot)
plt.ylim(-y_margin_plot, y_margin_plot + 1)
plt.xticks(np.arange(0, x_max_plot + 1), fontsize=axis_ticks_size)
plt.yticks(np.arange(0, p_ticks + 1, p_ticks), fontsize=axis_ticks_size)
plt.grid(True, alpha=0.5, zorder=0)
plt.show(True)

#----------#
#  EXPORT  #
#----------#

# functions
def console_print(title, df):
    print(Fore.GREEN + '\033[1m' + '\n' + title + Style.RESET_ALL)
    print(df)
def txt_export(title, df, f):
    print(title, file=f)
    print(df, file=f)

# export summary (console, txt)
with open(topic_underscore + '_summary' + ver + '.txt', 'w') as f:
    df = logits_summary
    title = topic.upper() + ' SUMMARY'
    console_print(title, df)
    txt_export(title, df, f)
del title, f, df

# export plot (pdf)
pdf = PdfPages(topic_underscore + '_plots' + ver + '.pdf')
pdf.savefig(fig1)
pdf.close()
del pdf

###

# runtime
runtime_sec = round(time.time() - time0, 2)
if runtime_sec < 60:
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec')
else:
    runtime_min_sec = str(int(np.floor(runtime_sec / 60))) + ' min ' + str(round(runtime_sec % 60, 2)) + ' sec'
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec (' + runtime_min_sec + ')')
del time0


