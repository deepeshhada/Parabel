#include <iostream>
#include <fstream>
#include <string>

#include "timer.h"
#include "xreg.h"
#include "stats.h"


using namespace std;


void help()
{
	cerr<<"Sample Usage :"<<endl;
	cerr<<"./xreg_train [input model folder name] [input feature file name] [input label file name] -s [start index of trees] -T [num thread] -t [num trees] -w [weighted] -k [internal_node_classifier_kind] -kleaf [leaf_node_classifier_kind] -c [classifier_cost] -m [max_leaf] -tcl [classifier_threshold] -ecl [classifier_eps] -n [classifier_maxiter]"<<endl<<endl;

	cerr<<"-r = param.tail_classifier              : Train nearest centroid cluster tail classifier, 1=True/0=False,								default=0"<<endl;
	cerr<<"-s = param.start_tree                   : Starting index of the trees								default=0"<<endl;
	cerr<<"-T = param.num_thread                   : Number of threads									default=1"<<endl;
	cerr<<"-t = param.num_tree                     : Number of trees to be grown								default=3"<<endl;
	cerr<<"-w = param.weighted                     : Whether input labels are binary or continuous probability scores, 1=continuous in [0,1], 0=binary. default=0"<<endl;
	cerr<<"-k = param.classifier_kind              : Kind of linear classifier to use in internal nodes. 0=L2R_L2LOSS_SVC_DUAL, 1=L2R_LR_DUAL, 2=L2R_L2LOSS_SVC_PRIMAL (not yet supported), 3=L2R_LR_PRIMAL, 4=L2R_L2LOSS_SVR_DUAL, 5=L2R_L1LOSS_SVR_DUAL (Refer to Liblinear)	default=L2R_L2LOSS_SVC"<<endl;
	cerr<<"-kleaf = param.leaf_classifier_kind     : Kind of linear classifier to use in leaf nodes. 0=L2R_L2LOSS_SVC_DUAL, 1=L2R_LR_DUAL, 2=L2R_L2LOSS_SVC_PRIMAL (not yet supported), 3=L2R_LR_PRIMAL, 4=L2R_L2LOSS_SVR_DUAL, 5=L2R_L1LOSS_SVR_DUAL  (Refer to Liblinear)	default=L2R_L2LOSS_SVC"<<endl;
	cerr<<"-c = param.classifier_cost              : Cost co-efficient for linear classifiers						default=1.0"<<endl;
	cerr<<"-m = param.max_leaf                     : Maximum no. of labels in a leaf node. Larger nodes will be split into 2 balanced child nodes.		default=100"<<endl;
	cerr<<"-tcl = param.classifier_threshold       : Threshold value for sparsifying linear classifiers' trained weights to reduce model size.		default=0.05"<<endl;
	cerr<<"-ecl = param.classifier_eps             : Eps value for logistic regression. default=0.1"<<endl;
	cerr<<"-n = param.classifier_maxiter           : Maximum iterations of algorithm for training linear classifiers			default=20"<<endl;
	
	cerr<<"The feature and label input files are expected to be in sparse matrix text format. Refer to README.md for more details."<<endl;
	exit(1);
}

std::ifstream::pos_type filesize(const char* filename)
{
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg(); 
}

int main(int argc, char* argv[])
{
	std::ios_base::sync_with_stdio(false);

	if(argc < 4)
		help();

	string model_folder = string( argv[1] );
	check_valid_foldername( model_folder );

	string trn_ft_file = string( argv[2] );
	check_valid_filename( trn_ft_file, true );

	string trn_lbl_file = string( argv[3] );
	check_valid_filename( trn_lbl_file, true );

	string param_file_name = model_folder + "/Params.txt";
	check_valid_filename( param_file_name, false );

	cout << "loading the datasets " << endl;

	SMatF* trn_X_Xf = new SMatF(trn_ft_file);
	SMatF* trn_X_Y = new SMatF(trn_lbl_file);

	cout << "The shape of trn_X_Xf is " << trn_X_Xf->nc << " " << trn_X_Xf->nr << endl;
	cout << "The shape of trn_X_Y is " << trn_X_Y->nc << " " << trn_X_Y->nr << endl;

	Param param;
	param.parse( argc-4, argv+4 );
	param.num_Xf = trn_X_Xf->nr;
	param.num_Y = trn_X_Y->nr;
	param.write( param_file_name );

	param.num_trn = trn_X_Xf->nc;

	_float model_size = 0;
	_float train_time = 0.0;
	
	if(param.tail_classifier)
	{
		_float tc_train_time;
		string model_file_name = model_folder + "/tail_classifier_model.bin";
		SMatF* model_mat = tail_classifier_train( trn_X_Xf, trn_X_Y, tc_train_time );
		train_time += tc_train_time;

		ofstream fout(model_file_name, ios::out | ios::binary);
		model_mat->writeBin(fout);

		delete model_mat;
		model_size += filesize(model_file_name.c_str());
	}
	
	Timer timer; timer.tic();

	trn_X_Xf->unit_normalize_columns();
	trn_X_Xf->append_bias_feat(param.bias_feat);
	SMatF* trn_Y_X = trn_X_Y->transpose();
	train_time += timer.toc();

	for(int i = 0; i < param.num_tree; ++i)
	{
		timer.tic();
		string tree_file_name = model_folder + "/Tree." + to_string( param.start_tree + i ) + ".bin";
		check_valid_filename( tree_file_name, false );

		ofstream fout;
		fout.open(tree_file_name, ios::out | ios::binary);
		
		XregTrain xreg_train(trn_X_Xf, trn_Y_X, param, param.start_tree + i, fout);
		train_time += timer.toc();

		xreg_train.train_tree(train_time);
		cout << "here " << tree_file_name << endl;

		model_size += filesize(tree_file_name.c_str());
	}

	cout << "\nmodel size : " << model_size / pow(2, 30) << " GB" << endl;
	cout << "training time : " << train_time * (0.00027777777) << " hrs" << endl;
	
	delete trn_Y_X;
	delete trn_X_Xf;
	delete trn_X_Y;
}
