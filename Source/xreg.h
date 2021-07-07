#pragma once

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
#include <queue>
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
#include "Param.h"
#include "LinearClassifier.h"


using namespace std;

class Node
{
public:
	_bool is_leaf;
	_int pos_child;
	_int neg_child;
	_int depth;
	_int id;
	VecI Y;
	SMatF* w;
	VecIF X;

	Node()
	{
		is_leaf = false;
		pos_child = neg_child = -1;
		depth = 0;
		w = NULL;
	}

	Node( VecI Y, _int depth, _int max_depth )
	{
		this->Y = Y;
		this->depth = depth;
		this->pos_child = -1;
		this->neg_child = -1;
		this->is_leaf = (depth >= max_depth-1);
		this->w = NULL;
	}

	~Node()
	{
		if(w)
			delete w;
	}

	_float get_ram()
	{
		_float ram = 0;
		ram += sizeof( Node );
		ram += sizeof( _int ) * Y.capacity();
		ram += w->get_ram();
		return ram;
	}

	friend ostream& operator<<( ostream& fout, const Node& node )
	{
		fout << node.id << " " << node.is_leaf << "\n";
		fout << node.pos_child << " " << node.neg_child << "\n";
		fout << node.depth << "\n";

		fout << node.Y.size();
		for( _int i=0; i<node.Y.size(); i++ )
			fout << " " << node.Y[i];
		fout << "\n";

		fout << (*node.w);
		return fout;
	}

	void writeBin(ofstream& fout)
    {
    	fout.write((char*)(&(id)), sizeof(_int));
        fout.write((char*)(&(is_leaf)), sizeof(_bool));
        fout.write((char*)(&(pos_child)), sizeof(_int));
        fout.write((char*)(&(neg_child)), sizeof(_int));
        fout.write((char*)(&(depth)), sizeof(_int));

        if (is_leaf)
        {
            _int Y_size = Y.size();
            fout.write((char*)(&(Y_size)), sizeof(_int));

            for (_int i = 0; i < Y_size; i++)
                fout.write((char*)(&(Y[i])), sizeof(_int));
        }

        w->writeBin(fout);
    }

    void readBin(ifstream& fin)
    {
        fin.read((char*)(&(is_leaf)), sizeof(_bool));
        fin.read((char*)(&(pos_child)), sizeof(_int));
        fin.read((char*)(&(neg_child)), sizeof(_int));
        fin.read((char*)(&(depth)), sizeof(_int));

        if (is_leaf)
        {
            _int Y_size;
            fin.read((char*)(&(Y_size)), sizeof(_int));
            Y.resize(Y_size);

            for (_int i = 0; i < Y_size; i++)
                fin.read((char*)(&(Y[i])), sizeof(_int));
        }

        w = new SMatF;
        w->readBin(fin);

    }

 	friend istream& operator>>( istream& fin, Node& node )
	{
		fin >> node.is_leaf;
		fin >> node.pos_child >> node.neg_child;
		fin >> node.depth;

		_int Y_size;
		fin >> Y_size;
		node.Y.resize( Y_size );

		for( _int i=0; i<Y_size; i++ )
			fin >> node.Y[i];

		node.w = new SMatF;
		fin >> (*node.w);
		return fin;
	} 
};

class Tree
{
public:
	_int num_Xf;
	_int num_Y;
	vector<Node*> nodes;

	Tree()
	{
		
	}

	void read_legacy_node(Node& node, ifstream& fin)
	{
		fin >> node.is_leaf;
		fin >> node.pos_child >> node.neg_child;
		fin >> node.depth;

		_int Y_size;
		fin >> Y_size;
		node.Y.resize( Y_size );

		for( _int i=0; i<Y_size; i++ )
			fin >> node.Y[i];

		string line;
		getline( fin, line );

		node.w = new SMatF;
		(node.w)->read_legacy_mat(fin);
	}

	Tree( string file_path, _bool binary=true )
	{
		ifstream fin;
		if(binary)
		{
			fin.open(file_path, ios::in | ios::binary);

            fin.read((char*)(&(num_Xf)), sizeof(_int));
            fin.read((char*)(&(num_Y)), sizeof(_int));
            _int num_node;
            fin.read((char*)(&(num_node)), sizeof(_int));

            cout << "total nodes to read: " << num_node << endl;

            for (_int i = 0; i < num_node; i++)
            {
                Node* node = new Node;
                nodes.push_back(node);
            }

            // cout << "init done " << endl;
            for(_int i = 0; i < num_node; i++)
			{
				_int node_id;
				fin.read((char*)(&(node_id)), sizeof(_int));

				// cout << i << " " << node_id << endl;
                nodes[node_id]->readBin(fin);
                nodes[node_id]->id = node_id;
            }

            cout << "Completed reading " << endl;
		}
		else
		{

			fin.open(file_path);

			fin >> num_Xf;
			fin >> num_Y;
			_int num_node;
			fin >> num_node;

			for(_int i = 0; i < num_node; i++)
			{
				Node* node = new Node;
				nodes.push_back(node);
			}

			for(_int i = 0; i < num_node; i++)
			{
				_int node_id;
				fin >> node_id;
				fin >> (*nodes[node_id]);
			}
		}
		fin.close();
	}

	// TODO : remove, this is only for legacy support in case of compare_trees.cpp
	Tree( string model_dir, _int tree_no )
	{
		ifstream fin;
		fin.open( model_dir + "/" + to_string( tree_no ) + ".tree" );
		fin >> num_Xf;
		fin >> num_Y;
		_int num_node;
		fin >> num_node;
		for( _int i=0; i<num_node; i++ )
		{
			Node* node = new Node;
			nodes.push_back( node );
		}
		for( _int i=0; i<num_node; i++ )
		{
			read_legacy_node(*nodes[i], fin);
		}
		fin.close();
	}

	~Tree()
	{
		for(_int i=0; i<nodes.size(); i++)
		{
			if(nodes[i])
				delete nodes[i];
		}
	}

	_float get_ram()
	{
		_float ram = 0;
		ram += sizeof( Tree );
		for(_int i=0; i<nodes.size(); i++)
			ram += nodes[i]->get_ram();
		return ram;
	}

	void write( string model_dir, _int tree_no )
	{
		ofstream fout;
		fout.open( model_dir + "/" + to_string( tree_no ) + ".tree" );

		fout << num_Xf << "\n";
		fout << num_Y << "\n";
		_int num_node = nodes.size();
		fout << num_node << "\n";

		for( _int i=0; i<num_node; i++ )
			fout << (*nodes[i]);

		fout.close();
	}
};

class XregTrain
{
private:
	SMatF* trn_X_Xf;
	SMatF* trn_Y_X;
	Param& param; 
	_int tree_no;
	ofstream fout;

	_int num_thread = 4;
	vector<mt19937> reng;
	vector<VecI> mask;

	_int num_X;
	_int num_Xf;
	_int num_Y;
	_int max_depth;
	_int num_nodes;

	vector<Node*> tree_nodes;
	vector<_float> normalizers;

public:
	XregTrain(SMatF* _trn_X_Xf, SMatF* _trn_Y_X, Param& _param, _int _tree_no, ofstream& _fout) : 	trn_X_Xf(_trn_X_Xf), 
																										trn_Y_X(_trn_Y_X),
																										param(_param), 
																										tree_no(_tree_no), 
																										fout(move(_fout)),
																										num_thread(param.num_thread)
	{
		reng.resize(num_thread);
		for(int i = 0; i < num_thread; ++i)
			reng[i].seed(tree_no);

		num_X 	= trn_X_Xf->nc;
		num_Xf 	= trn_X_Xf->nr;
		num_Y 	= trn_Y_X->nc;

		max_depth = ceil(log2((_float)num_Y / (_float)param.max_leaf)) + 1;
		num_nodes = pow(2, max_depth) - 1;

		Node* root 	= init_root(num_Y, max_depth);
		root->id 	= 0;

		tree_nodes.resize(num_nodes, NULL);
		tree_nodes[0] = root;

		_int max_n = max(max(num_X + 1, num_Xf + 1), num_Y + 1);
		mask.resize(num_thread, VecI(max_n, 0));

		logParameters();
	}

	void logParameters()
	{
		cout << "num_thread is " << num_thread << endl;
		cout << "num_Y and max_leaf are " << num_Y << " " << param.max_leaf << endl;
		cout << "max_depth:" << max_depth << endl;
	}

	void writeTreeHeader()
	{
		fout.write((char*)(&(num_Xf)), sizeof(_int));
	    fout.write((char*)(&(num_Y)), sizeof(_int));
	    fout.write((char*)(&(num_nodes)), sizeof(_int));
	}

	void train_tree(_float &train_time);

	void get_label_centroid_normalizers(_float &train_time);

	Node* init_root( _int num_Y, _int max_depth )
	{
		VecI lbls;
		for( _int i=0; i<num_Y; i++ )
			lbls.push_back(i);
		Node* root = new Node( lbls, 0, max_depth );
		return root;
	}
};

class XregPredict
{
private:
	SMatF* tst_X_Xf;
	Tree* tree;
	Param param;
	ofstream fout;

	_int num_X;
	_int num_Y;
	
	_int beam_size;
	_int max_leaf;
	_Classifier_Kind classifier_kind;
	
	_float discount = 1.0;
	_int max_depth;

	_float **mask;
	_int num_thread = 4;

	string model_dir;

public:
    XregPredict(SMatF* _tst_X_Xf, Tree* _tree, Param& _param, ofstream& _fout, string model_dir) : 	tst_X_Xf(_tst_X_Xf), 
    																									tree(_tree), 
    																									param(_param), 
    																									fout(move(_fout)), 
    																									model_dir(model_dir),
    																									num_thread(param.num_thread)
	{
		num_X = tst_X_Xf->nc;
		num_Y = param.num_Y;
		
		beam_size 		= param.beam_width;
		max_leaf 		= param.max_leaf;
		classifier_kind = param.classifier_kind;
		discount 		= 1.0;
		max_depth 		= ceil(log2((_float)num_Y / (_float)max_leaf)) + 1;

		mask = new _float*[num_thread];
		for (int i = 0; i < num_thread; i++) {
			mask[i] = new _float[tst_X_Xf->nr + 1]();
		}

		vector<Node*>& 	nodes 		= tree->nodes;
		_int 			num_nodes 	= nodes.size();

		param.per_label_predict_intercept = get_per_label_predict_intercept(	num_X, 
																				max_leaf, 
																				max_depth - 1, 
																				param.per_label_predict_factor, 
																				param.per_label_predict_slope	);
		
		logParameters();
	}

	void logParameters()
	{
		cout << "max depth is " << max_depth << endl;
		if(param.per_label_predict)
			cout << "the intercept and the slope are " << param.per_label_predict_intercept << " and " << param.per_label_predict_slope << endl; 
	}

	void predict_tree(_float &predict_time);
	void predict_tree_per_label(_float &predict_time);

	_float compute_leaf_num( _int num_tst, _int max_depth, _float per_label_predict_slope, _float per_label_predict_intercept );
	_float get_per_label_predict_intercept( _int num_tst, _int max_leaf, _int max_depth, _float per_label_predict_factor, _float per_label_predict_slope );

	_float getNewScore(_float prod, _float prev_score, _Classifier_Kind classifier_kind);

	void getAllScoresForNode(Node* node, _int inst, _int n, _int thread_id, vector<vector<pairIF>>& score_mat_private, VecIF& mapped_points);
	void leafPredictPerPoint(_int point_id, _int thread_id, VecIF& current_scores, VecIF& new_scores, vector<Node*>& nodes);
	void leafPredictPerLabel(Node* node, _int thread_id, VecIF& mapped_points);
};