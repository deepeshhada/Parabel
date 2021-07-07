#include "tail_classifiers.h"

using namespace std;

SMatF* tail_classifier_train( SMatF* trn_X_Xf, SMatF* trn_X_Y, _float& train_time )
{
	train_time = 0;

	Timer timer;
	timer.tic();

	SMatF* c_trn_X_Xf = new SMatF(trn_X_Xf);
	c_trn_X_Xf->unit_normalize_columns();

	SMatF* trn_Y_X = trn_X_Y->transpose();

	for(int i=0; i<trn_Y_X->nc; i++)
	{
		_float a = trn_Y_X->size[i]==0 ? 1.0 : 1.0/(trn_Y_X->size[i]);
		for(int j=0; j<trn_Y_X->size[i]; j++)
			trn_Y_X->data[i][j].second = a;
	}

	SMatF* model_mat = c_trn_X_Xf->prod(trn_Y_X);
	model_mat->unit_normalize_columns();

	train_time += timer.toc();

	cout << "Tail classifier's training time: " << train_time/3600.0 << " hr" << endl;

	delete trn_Y_X;
	delete c_trn_X_Xf;
	
	return model_mat;
}

void tail_classifier_predict_per_label( SMatF* tst_X_Xf, _int lbl, VecIF &node_score_mat, SMatF* model_mat, _float alpha, _float& predict_time, _float& model_size )
{
	// NOTE : assumption, tst_X_Xf is unit normalized + bias added

	_int num_tst = tst_X_Xf->nc;
	_int num_ft = tst_X_Xf->nr;
	_int num_lbl = model_mat->nc;

	_float* mask = new _float[num_ft - 1]();

	// densify model_mat->data[lbl]
	for( _int i = 0; i < model_mat->size[lbl]; i++ )
		mask[ (model_mat->data[lbl][i]).first ] = (model_mat->data[lbl][i]).second;

	for(_int i = 0; i < node_score_mat.size(); i++)
	{
		_int x = node_score_mat[i].first;
		_float score = node_score_mat[i].second;

		// NOTE : tst_X_Xf has extra bias feature that's why "tst_X_Xf->size[x]-1"
		_float prod = mult_d_s_vec(mask, tst_X_Xf->data[x], tst_X_Xf->size[x]-1);

		node_score_mat[i].second = pow(score, alpha)*pow(exp(prod), 1-alpha);
	}

	delete[] mask;
}

void tail_classifier_predict_per_point( SMatF* tst_X_Xf, _int x, VecIF &node_score_mat, SMatF* model_mat, _float alpha, _float& predict_time, _float& model_size )
{
	// NOTE : assumption, tst_X_Xf is unit normalized + bias added

	_int num_tst = tst_X_Xf->nc;
	_int num_ft = tst_X_Xf->nr;
	_int num_lbl = model_mat->nc;

	_float* mask = new _float[num_ft - 1]();

	// densify tst_X_Xf->data[x], NOTE : tst_X_Xf has extra bias feature that's why "i < tst_X_Xf->size[x]-1"
	for( _int i = 0; i < tst_X_Xf->size[x]-1; i++ )
		mask[ (tst_X_Xf->data[x][i]).first ] = (tst_X_Xf->data[x][i]).second;

	for(_int i = 0; i < node_score_mat.size(); i++)
	{
		_int lbl = node_score_mat[i].first;
		_float score = node_score_mat[i].second;

		_float prod = mult_d_s_vec(mask, model_mat->data[lbl], model_mat->size[lbl]);

		node_score_mat[i].second = pow(score, alpha)*pow(exp(prod), 1-alpha);
	}

	// reset
	for( _int i = 0; i < tst_X_Xf->size[x]-1; i++ )
		mask[ (tst_X_Xf->data[x][i]).first ] = 0;

	delete[] mask;
}

SMatF* tail_classifier_predict( SMatF* tst_X_Xf, SMatF* score_mat, SMatF* model_mat, _float alpha, _float threshold, _float& predict_time, _float& model_size )
{
	// NOTE : assumption, tst_X_Xf is unit normalized + bias added

	predict_time = 0;
	model_size = model_mat->get_ram();

	Timer timer;
	timer.tic();

	_int num_tst = tst_X_Xf->nc;
	_int num_ft = tst_X_Xf->nr-1;
	_int num_lbl = model_mat->nc;

	SMatF* tmat = score_mat->transpose();
	VecF mask(num_ft,0);

	for(_int i=0; i<num_lbl; i++)
	{
		for(_int j=0; j<model_mat->size[i]; j++)
			mask[model_mat->data[i][j].first] = model_mat->data[i][j].second;

		for(_int j=0; j<tmat->size[i]; j++)
		{
			_int inst = tmat->data[i][j].first;
			_float prod = 0;
			// NOTE : tst_X_Xf has extra bias feature that's why "k < tst_X_Xf->size[inst]-1"
			for(_int k=0; k<tst_X_Xf->size[inst]-1; k++)
			{
				_int id = tst_X_Xf->data[inst][k].first;	
				_float val = tst_X_Xf->data[inst][k].second;
				prod += mask[id]*val;
			}
			tmat->data[i][j].second = prod;
		}

		for(_int j=0; j<model_mat->size[i]; j++)
			mask[model_mat->data[i][j].first] = 0;
	}

	SMatF* tail_score_mat = tmat->transpose();

	/* combine PfastXML scores and tail classifier scores to arrive at final scores */ 
	for(_int i=0; i<num_tst; i++)
	{
		for(_int j=0; j<score_mat->size[i]; j++)
		{
			_float score = score_mat->data[i][j].second;
			_float tail_score = tail_score_mat->data[i][j].second;
			tail_score_mat->data[i][j].second = fabs(tail_score)>threshold ? pow(score,alpha)*pow(tail_score,1-alpha) : 0.0;
		}
	}

	// cout << "\nscore file name: " << tail_score_mat->data << endl;
	tail_score_mat->eliminate_zeros();

	predict_time += timer.toc();

	delete tmat;

	return tail_score_mat;
}