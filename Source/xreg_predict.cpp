#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdio>

#include "xreg.h"

using namespace std;

void help()
{
	cerr<<"Sample Usage :"<<endl;
	cerr<<"./xreg_predict [input model folder name] [output score file name] [input feature file name] -s 0 -B 10 -p 0 -pf 10.0 -ps -0.05 -C -1"<<endl<<endl;

	cerr<<"-r = param.tail_classifier              : Predict using nearest centroid cluster tail classifier, 1=True/0=False,								default=0"<<endl;
	cerr<<"-a = param.alpha                        : Weight of XReg's score, if using tail classifier 								default=1.0"<<endl;
	cerr<<"-T = param.num_thread                   : Number of threads									default=1"<<endl;
	cerr<<"-B = param.beam_width                   : Beam search width for fast, approximate prediction					default=10"<<endl;
	cerr<<"-p = param.per_label_predict            : Predict top test points per each label. Useful in DSA-like scenarios 0=predict top labels per point, 1=predict top points per label default=[value saved in trained model]"<<endl;
	cerr<<"-pf = param.per_label_predict_factor    : per_label_predict_factor*max_leaf' number of test points are finally passed down to each leaf node. default=10.0"<<endl;
	cerr<<"-ps = param.per_label_predict_slope     : slope of the linear function which decides how many test points are passed from parent to child. Function is linear in node depth. default=-0.05"<<endl;
	cerr<<"-s = param.start_tree                   : Starting index of tree		default=[as saved in model, 0]"<<endl;

	cerr<<"The feature and score files are expected to be in sparse matrix text format. Refer to README.md for more details"<<endl;
	exit(1);
}

int main(int argc, char* argv[])
{
	std::ios_base::sync_with_stdio(false);

	if(argc < 4)
		help();

	string model_folder = string( argv[1] );
	check_valid_foldername( model_folder );

	string score_file_name = string( argv[2] );
	check_valid_filename( score_file_name, false );

	string tst_ft_file = string( argv[3] );
	check_valid_filename( tst_ft_file, true );

	string param_file_name = model_folder + "/Params.txt";
	check_valid_filename( param_file_name, true );

	Param param( param_file_name );
	param.parse( argc-4, argv+4 );

	cout << "Loading all mats... " << endl;

	SMatF* tst_X_Xf = new SMatF(tst_ft_file);
	cout << "Shape of tst_X_Xf " << tst_X_Xf->nc << " " << tst_X_Xf->nr << endl;

	_int num_X = tst_X_Xf->nc;
	_int num_Y = param.num_Y;

	SMatF* tail_classifier_model = NULL;	
	if(param.tail_classifier)
	{
	    ifstream f(model_folder + string("/tail_classifier_model.bin"));
	    tail_classifier_model = new SMatF();
	    tail_classifier_model->readBin(f);
	}

	_float predict_time = 0.0;
	Timer timer; timer.tic();

	tst_X_Xf->unit_normalize_columns();
	tst_X_Xf->append_bias_feat(param.bias_feat);

	predict_time += timer.toc();

	for(int i = 0; i < param.num_tree; ++i)
	{
		string tree_file_name = model_folder + "/Tree." + to_string( param.start_tree + i ) + ".bin";
		check_valid_filename( tree_file_name, true );

		string temp_score_file_name = string( argv[2] ) + to_string( param.start_tree + i );
		check_valid_filename( temp_score_file_name, false );

		Tree* tree = new Tree( tree_file_name );
		cout << "Total nodes in the tree are " << tree->nodes.size() << endl;

		ofstream fout(temp_score_file_name);

		timer.tic();
		XregPredict xreg_predict(tst_X_Xf, tree, param, fout, model_folder);
		predict_time += timer.toc();

		if(param.per_label_predict)
			xreg_predict.predict_tree_per_label(predict_time);
		else
			xreg_predict.predict_tree(predict_time);

		delete tree;
	}

	cout << "\nprediction time before taking ensemble : " << (predict_time * 1000.0) / num_X << " ms/point\n" << endl;
	timer.tic();

	SMatF* score_mat;
	if(param.per_label_predict)
		score_mat = new SMatF( num_X, num_Y );
	else
		score_mat = new SMatF( num_Y, num_X );

	for(int i = 0; i < param.num_tree; ++i)
	{
		cout << "Taking ensemble of predictions..." << endl;

		string temp_score_file_name = string( argv[2] ) + to_string( param.start_tree + i );
		ifstream fin(temp_score_file_name);

		string line;
		cout << "temp_score_file_name:" << temp_score_file_name << endl;
		// cout << "score mat j:" << score_mat->size[i] << endl;
		while(getline(fin, line))
		{
			std::istringstream iss(line);

			char colon;
			_int col_no; iss >> col_no;

			_int label_id;
			_float label_score;

			vector<pair<_int, _float>> scores;

			while(iss >> label_id >> colon >> label_score)
				scores.push_back(pair<_int, _float>(label_id, label_score));

			sort(scores.begin(), scores.end());

			pairIF* score_mat_col = score_mat->data[col_no]; _int score_mat_col_siz = score_mat->size[col_no];
			vector<pairIF> sum = add_s_to_s(score_mat_col, score_mat_col_siz, scores.data(), scores.size());

			score_mat->data[col_no] = getDeepCopy(sum.data(), sum.size());
			score_mat->size[col_no] = sum.size();
			
			if(score_mat_col) delete[] score_mat_col;
		}

		remove(temp_score_file_name.c_str());
	}

	for(_int i=0; i<score_mat->nc; i++) 
	{
		for(_int j=0; j<score_mat->size[i]; j++)
		{
			score_mat->data[i][j].second /= param.num_tree;
			// cout << "here" << score_mat->data[i][j].second << endl;
		}
	}
	cout << "nc:" << score_mat->nc << " nr:" << score_mat->nr << endl;

	if(param.per_label_predict)
	{
		SMatF* t_score_mat = score_mat->transpose();
		delete score_mat;
		score_mat = t_score_mat;
	}

	if( param.tail_classifier )
	{
		_float tc_prediction_time;
		_float tc_model_size;
	
		SMatF* new_score_mat = tail_classifier_predict( tst_X_Xf, score_mat, tail_classifier_model, param.alpha, 0.0, tc_prediction_time, tc_model_size );
		delete score_mat;
		score_mat = new_score_mat;
	}
	cout << "sample pred:" << score_mat->data[5][5].second << endl;

	predict_time += timer.toc();
	cout << "\nprediction time : " << (predict_time * 1000.0) / num_X << " ms/point" << endl;

	ofstream fout( score_file_name );
	fout << (*score_mat);
	fout.close();

	delete tst_X_Xf;
	delete score_mat;
}
