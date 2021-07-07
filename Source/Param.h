#pragma once

// TODO : remove unecessary headers

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <iomanip>
#include <random>
#include <thread>
#include <mutex>
#include <functional>
#include <unordered_map>
#include <omp.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <ctime>

#include "config.h"
#include "utils.h"
#include "mat.h"
#include "timer.h"
#include "algos.h"
#include "data.h"
#include "stats.h"

using namespace std;

enum _Classifier_Kind { L2R_L2LOSS_SVC_DUAL=0, L2R_LR_DUAL, L2R_L2LOSS_SVC_PRIMAL, L2R_LR_PRIMAL, L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL, NONE };
const vector<string> _Classifier_Kind_Name = {"L2R_L2LOSS_SVC_DUAL", "L2R_LR_DUAL", "L2R_L2LOSS_SVC_PRIMAL", "L2R_LR_PRIMAL", "L2R_L2LOSS_SVR_DUAL", "L2R_L1LOSS_SVR_DUAL", "NONE"};
enum VersionType { CLASSIFIER1 = 0, CLASSIFIER2 = 1, BOTH_CLASSIFIER = 2 };

class Param
{
public:
	_int num_trn;
	_int num_Xf;
	_int num_Y;
	_int num_thread;
	_int start_tree;
	_int num_tree;
	_float classifier_cost;
	_int max_leaf;
	_float bias_feat;
	_float classifier_threshold;
	_float centroid_threshold;
	_float classifier_eps;
	_float classifier_p;
	_float clustering_eps;
	_int classifier_maxitr;
	_Classifier_Kind classifier_kind;
	_Classifier_Kind leaf_classifier_kind;
	_bool quiet;
	_int beam_width;
	_bool weighted;
	_bool per_label_predict;
	_float per_label_predict_factor;
	_float per_label_predict_slope;
	_float per_label_predict_intercept;
	_bool tail_classifier;
	_float alpha;
	_int num_pred;
	_bool pattern_predict;
	_int prediction_chunk_size;

	Param()
	{
		num_Xf = 0;
		num_Y = 0;
		num_thread = 1;
		start_tree = 0;
		num_tree = 1;
		classifier_cost = 10.0;
		max_leaf = 20;
		bias_feat = 1.0;
		classifier_threshold = 0.05;
		centroid_threshold = 0;
		classifier_eps = 0.1;
		classifier_p = 0.005;
		clustering_eps = 1e-4;
		classifier_maxitr = 100;
		classifier_kind = L2R_L2LOSS_SVC_DUAL;
		leaf_classifier_kind = L2R_L2LOSS_SVC_DUAL;
		quiet = false;
		beam_width = 10;
		weighted = false;
		per_label_predict = false;
		per_label_predict_factor = 10.0; // reduce this further if max_leaf is increased from 20, eg. if max_leaf=100, per_label_predict_factor=5.0 might give best accuracy vs prediction time trade off
		per_label_predict_slope = -0.05;
		tail_classifier = false;
		alpha = 1.0;
		num_pred = 1000;
		pattern_predict = false;
		prediction_chunk_size = -1;
	}

	Param(string fname)
	{
		check_valid_filename(fname,true);
		ifstream fin;
		fin.open(fname);

		fin>>num_Xf;
		fin>>num_Y;
		fin>>num_thread;
		fin>>start_tree;
		fin>>num_tree;
		fin>>classifier_cost;
		fin>>max_leaf;
		fin>>bias_feat;
		fin>>classifier_threshold;
		fin>>centroid_threshold;
		fin>>classifier_eps;
		fin>>classifier_p;
		fin>>clustering_eps;
		fin>>classifier_maxitr;
		_int ck;
		fin>>ck;
		classifier_kind = (_Classifier_Kind)ck;
		fin>>ck;
		leaf_classifier_kind = (_Classifier_Kind)ck;
		fin>>quiet;
		fin>>beam_width;
		fin>>weighted;
		fin>>per_label_predict;
		fin>>per_label_predict_factor;
		fin>>per_label_predict_slope;
		fin>>tail_classifier;
		fin>>alpha;
		fin>>num_pred;
		fin>>pattern_predict;
		fin>>prediction_chunk_size;

		fin.close();
	}

	void write(string fname)
	{
		check_valid_filename(fname,false);
		ofstream fout;
		fout.open(fname);

		fout<<num_Xf<<"\n";
		fout<<num_Y<<"\n";
		fout<<num_thread<<"\n";
		fout<<start_tree<<"\n";
		fout<<num_tree<<"\n";
		fout<<classifier_cost<<"\n";
		fout<<max_leaf<<"\n";
		fout<<bias_feat<<"\n";
		fout<<classifier_threshold<<"\n";
		fout<<centroid_threshold<<"\n";
		fout<<classifier_eps<<"\n";
		fout<<classifier_p<<"\n";
		fout<<clustering_eps<<"\n";
		fout<<classifier_maxitr<<"\n";
		fout<<classifier_kind<<"\n";
		fout<<leaf_classifier_kind<<"\n";
		fout<<quiet<<"\n";
		fout<<beam_width<<"\n";
		fout<<weighted<<"\n";
		fout<<per_label_predict<<"\n";
		fout<<per_label_predict_factor<<"\n";
		fout<<per_label_predict_slope<<"\n";
		fout<<tail_classifier<<"\n";
		fout<<alpha<<"\n";
		fout<<num_pred<<"\n";
		fout<<pattern_predict<<"\n";
		fout<<prediction_chunk_size<<"\n";

		fout.close();
	}

	void parse( _int argc, char* argv[] )
	{
		string opt;
		string sval;

		for(_int i=0; i<argc; i+=2)
		{
			opt = string(argv[i]);
			sval = string(argv[i+1]);

			if( opt == "-T" ) 
				num_thread = stoi( sval );
			else if( opt == "-s" )
				start_tree = stoi( sval );
			else if( opt == "-t" )
				num_tree = stoi( sval );
			else if( opt == "-b" )
				bias_feat = stof( sval );
			else if( opt == "-c" )
				classifier_cost = stof( sval );
			else if( opt == "-m" )
				max_leaf = stoi( sval );
			else if( opt == "-tcl" )
				classifier_threshold = stof( sval );
			else if( opt == "-tce" )
				centroid_threshold = stof( sval );
			else if( opt == "-ecl" )
				classifier_eps = stof( sval );
			else if( opt == "-P" )
				classifier_p = stof( sval );
			else if( opt == "-ece" )
				clustering_eps = stof( sval );
			else if( opt == "-n" )
				classifier_maxitr = stoi( sval );
			else if( opt == "-k" )
				classifier_kind = (_Classifier_Kind)(stoi( sval ));
			else if( opt == "-kleaf" )
				leaf_classifier_kind = (_Classifier_Kind)(stoi( sval ));
			else if( opt == "-q" )
				quiet = (_bool)(stoi( sval ));
			else if( opt == "-B" )
				beam_width = stoi( sval );
			else if( opt == "-w" )
				weighted = (_bool)(stoi( sval ));
			else if( opt == "-p" )
				per_label_predict = (_bool)(stoi( sval ));
			else if( opt == "-pf" )
				per_label_predict_factor = stof( sval );
			else if( opt == "-ps" )
				per_label_predict_slope = stof( sval );
			else if( opt == "-r" )
				tail_classifier = (_bool)(stoi( sval ));
			else if( opt == "-a" )
				alpha = stof( sval );
			else if( opt == "-N" )
				num_pred = stoi( sval );
			else if( opt == "-P" )
				pattern_predict = (_bool)(stoi( sval ));
			else if( opt == "-C" )
				prediction_chunk_size = stoi( sval );
		}
	}

	friend ostream& operator<<( ostream& fout, const Param& param )
	{
		fout<<param.num_Xf<<"\n";
		fout<<param.num_Y<<"\n";
		fout<<param.num_thread<<"\n";
		fout<<param.start_tree<<"\n";
		fout<<param.num_tree<<"\n";
		fout<<param.classifier_cost<<"\n";
		fout<<param.max_leaf<<"\n";
		fout<<param.bias_feat<<"\n";
		fout<<param.classifier_threshold<<"\n";
		fout<<param.centroid_threshold<<"\n";
		fout<<param.classifier_eps<<"\n";
		fout<<param.classifier_p<<"\n";
		fout<<param.clustering_eps<<"\n";
		fout<<param.classifier_maxitr<<"\n";
		fout<<param.classifier_kind<<"\n";
		fout<<param.leaf_classifier_kind<<"\n";
		fout<<param.quiet<<"\n";
		fout<<param.beam_width<<"\n";
		fout<<param.weighted<<"\n";
		fout<<param.per_label_predict<<"\n";
		fout<<param.per_label_predict_factor<<"\n";
		fout<<param.per_label_predict_slope<<"\n";
		fout<<param.tail_classifier<<"\n";
		fout<<param.alpha<<"\n";
		fout<<param.num_pred<<"\n";
		fout<<param.pattern_predict<<"\n";
		fout<<param.prediction_chunk_size<<"\n";

		return fout;
	}
};