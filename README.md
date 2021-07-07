# XReg

The objective of eXtreme Regression (XR) is to accurately predict the numerical degrees of relevance of an extremely large number of labels to a data point. XR can provide elegant solutions to many largescale ranking and recommendation applications including Dynamic Search Advertising (DSA). XR can learn more accurate models than the recently popular extreme classifiers which incorrectly assume strictly binary-valued label relevances. Traditional regression metrics which sum the errors over all the labels are unsuitable for XR problems since they could give extremely loose bounds for the label ranking quality. Also, the existing regression algorithms won’t efficiently scale to millions of labels. XR addresses these limitations through: (1) new evaluation metrics for XR which sum only the k largest regression errors; (2) a new algorithm called XReg which decomposes XR task into a hierarchy of much smaller regression problems thus leading to highly efficient training and prediction. This paper also introduces a (3) new labelwise prediction algorithm in XReg useful for DSA and other recommendation tasks. Please refer to the [paper](http://manikvarma.org/pubs/prabhu20.pdf) for more details.

This is the official codebase for the WSDM 2020 publication - "[Extreme Regression for Dynamic Search Advertising](http://manikvarma.org/pubs/prabhu20.pdf)". Please [cite](http://manikvarma.org/pubs/selfbib.html#Prabhu20) the paper if you happen to use this code base.

## License

This code is made available as is for non-commercial research purposes. Please make sure that you have read the license agreement in LICENSE.doc/pdf. Please do not install or use XReg unless you agree to the terms of the license.


## Build

Linux/Windows makefiles for compiling XReg have been provided with the source code. To compile, run `make` (Linux) or `nmake -f Makefile.win` (Windows) in the Source folder.

## Sample Usage
Please refer to [sample_run.sh](https://github.com/nilesh2797/XReg/blob/master/Source/sample_run.sh) which runs XReg over inverse propensity weighted EUR-Lex dataset.

## Training

`./xreg_train [input model folder name] [input feature file name] [input label file name] -s [start index of trees] -T [num thread] -t [num trees] -w [weighted] -k [internal_node_classifier_kind] -kleaf [leaf_node_classifier_kind] -c [classifier_cost] -m [max_leaf] -tcl [classifier_threshold] -ecl [classifier_eps] -n [classifier_maxiter] -r [use tail classifier]` 

| Option         | Meaning       | Default                |
| -------------    |:---------------:| -----------------------:|
|-r = param.tail_classifier | Train nearest centroid cluster tail classifier, 1=True/0=False | 0 |
|-s = param.start_tree |Starting index of the trees | 0 |
|-T = param.num_thread |Number of threads | 1 |
|-t = param.num_tree |Number of trees to be grown | 3 |
|-w = param.weighted |Whether input labels are binary or continuous probability scores, 1=continuous in `[0,1]`, 0=binary | 0 |
|-k = param.classifier_kind |Kind of linear classifier to use in internal nodes. 0=L2R_L2LOSS_SVC_DUAL, 1=L2R_LR_DUAL, 2=L2R_L2LOSS_SVC_PRIMAL (not yet supported), 3=L2R_LR_PRIMAL, 4=L2R_L2LOSS_SVR_DUAL, 5=L2R_L1LOSS_SVR_DUAL (Refer to Liblinear)| 0 |
|-kleaf = param.leaf_classifier_kind |Kind of linear classifier to use in leaf nodes. 0=L2R_L2LOSS_SVC_DUAL, 1=L2R_LR_DUAL, 2=L2R_L2LOSS_SVC_PRIMAL (not yet supported), 3=L2R_LR_PRIMAL, 4=L2R_L2LOSS_SVR_DUAL, 5=L2R_L1LOSS_SVR_DUAL  (Refer to Liblinear)	default=L2R_L2LOSS_SVC | 0 |
|-c = param.classifier_cost | Cost co-efficient for linear classifiers | 1.0 |
|-m = param.max_leaf |Maximum no. of labels in a leaf node. Larger nodes will be split into 2 balanced child nodes |100 |
|-tcl = param.classifier_threshold |Threshold value for sparsifying linear classifiers' trained weights to reduce model size |0.05 |
|-ecl = param.classifier_eps |Eps value for logistic regression | 0.1 |
|-n = param.classifier_maxiter |Maximum iterations of algorithm for training linear classifiers | 20 |

The feature and label input files are expected to be in sparse matrix text format. Refer to README.md for more details.
  
## Prediction  
`./xreg_predict [input model folder name] [output score file name] [input feature file name] -T [num threads] -s [start index of trees] -B [beam width] -p [per label predict] -pf [per label predict factor] -ps [per label predict slope] -r [use tail classifier] -a [alpha]`

| Option         | Meaning       | Default                |
| -------------    |:---------------:| -----------------------:|
|-r = param.tail_classifier |Predict using nearest centroid cluster tail classifier, 1=True/0=False | 0 |
|-a = param.alpha |Weight of XReg's score, if using tail classifier | 1.0 |
|-T = param.num_thread |Number of threads | 1 |
|-B = param.beam_width |Beam search width for fast, approximate prediction | 10 |
|-p = param.per_label_predict |Predict top test points per each label. Useful in DSA-like scenarios 0=predict top labels per point, 1=predict top points per label | 0 |
|-pf = param.per_label_predict_factor |`per_label_predict_factor*max_leaf number` of test points are finally passed down to each leaf node, if using per label predict option | 10.0 |
|-ps = param.per_label_predict_slope |slope of the linear function which decides how many test points are passed from parent to child. Function is linear in node depth, if using per label predict option | -0.05 |
|-s = param.start_tree |Starting index of the trees | 0 |

The feature and score files are expected to be in sparse matrix text format. Refer to README.md for more details

## Evaluation
`./xreg_metric [output score mat file name] [input feature file name] [K]  [weighted]`
