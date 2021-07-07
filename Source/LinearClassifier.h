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
#include "Param.h"

using namespace std;

#define LOG(__str, log_level) if(log_level >= global_log_level) \
								{ \
									giveTabs(); \
									cout << __str << endl; \
								}

#define printVar(__x) #__x << " is " << __x

class LinearClassifier
{
private:
	SMatF* trn_X_Xf;
	SMatF* trn_Y_X;

	_float C;
	_float P;
	_float classifier_cost;
	_float classifier_threshold;
	_float eps;
	_int num_trn;
	_int classifier_maxitr;

	_int num_Y;
	_int num_X;
	_int num_Xf;
	_llint nnz;

	_Classifier_Kind classifier_kind;
	bool weighted;

	enum LOG_LEVEL
	{
		ALL,
		DEBUG,
		INFO,
		NOLOG
	};

	int global_tab = 0;
	int global_log_level = NOLOG;

	void giveTabs()
	{
		for(int i = 0; i < global_tab; ++i)
			cout << "\t";
	}

public:
	LinearClassifier(SMatF* trn_X_Xf, SMatF* trn_Y_X, Param& param, _int explicit_weighted = -1, _Classifier_Kind explicit_classifier_kind = NONE)
																	 : 	trn_X_Xf(trn_X_Xf),
																		trn_Y_X(trn_Y_X),
																		classifier_cost(param.classifier_cost),
																		num_trn(param.num_trn),
																		classifier_maxitr(param.classifier_maxitr),
																		classifier_threshold(param.classifier_threshold),
																		eps(param.classifier_eps),
																		classifier_kind(param.classifier_kind),
																		weighted(param.weighted)

	{
		if(explicit_weighted > -1)
			weighted = explicit_weighted;

		if(explicit_classifier_kind != NONE)
			classifier_kind = explicit_classifier_kind;

		_float f;
		if( classifier_kind == L2R_LR_DUAL || classifier_kind == L2R_LR_PRIMAL )
			f = (_float)param.num_trn / (_float)trn_X_Xf->nc;
		else
			f = 1.0;

		C = param.classifier_cost * f;
		P = param.classifier_p;
		num_Y = trn_Y_X->nc;
		num_X = trn_X_Xf->nc;
		num_Xf = trn_X_Xf->nr;
		nnz = trn_X_Xf->get_nnz();
	}
	
	void prepareWtMat(VecF& inwts, SMatF*& copy_trn_X_Xf, _float*& wts, _int*& y, _float C, _int l )
	{
		copy_trn_X_Xf = new SMatF( trn_X_Xf, false); // shallow copy

		VecIF extras;
		for( _int i=0; i<trn_Y_X->size[l]; i++ )
		{
			_float val = trn_Y_X->data[l][i].second;
			if( val>0.0 && val<1.0 )
				extras.push_back( trn_Y_X->data[l][i] );
		}

		_float sum_inwts = 0;
		for( _int i=0; i<inwts.size(); i++ )
			sum_inwts += inwts[i];

		C /= sum_inwts;

		_int new_num_X = num_X + extras.size();
		wts = new _float[ new_num_X ];
		fill( wts, wts+new_num_X, C );

		for( int i=0; i<num_X; i++ )
			wts[i] *= inwts[i];

		for( int i=0; i<extras.size(); i++ )
		{
			int ind = extras[i].first;
			wts[ num_X + i ] *= inwts[ ind ];
		}

		y = new _int[ new_num_X ];
		fill( y, y+new_num_X, -1 );

		for( _int i=0; i<trn_Y_X->size[l]; i++ )
			y[ trn_Y_X->data[l][i].first ] = +1;

		for( _int i=0; i<extras.size(); i++ )
		{
			_int ind = extras[i].first;
			_float val = extras[i].second;
			wts[ ind ] *= val;
			wts[ num_X + i ] *= (1-val);

			copy_trn_X_Xf->addCol(trn_X_Xf->data[ind], trn_X_Xf->size[ind], false); // shallow copy of data[ind]
		}

		LOG(printVar(copy_trn_X_Xf->nc) << ", " << printVar(copy_trn_X_Xf->nr), DEBUG);
	}

	SMatF* train( VecF& inwts, mt19937& reng, bool print = false)
	{
		SMatF* w_mat = new SMatF( num_Xf, num_Y );
		_float* w = new _float[ num_Xf ];
		SMatF* copy_trn_X_Xf;

		for( _int l=0; l<num_Y; l++ )
		{
			for( _int i=0; i<num_Xf; i++ )
				w[i] = 0;

			_float* wts;
			_int* y = NULL;
			_float* fy = NULL;

			LOG(printVar(l), DEBUG);
			LOG("preparing matrices...", DEBUG);

			if( classifier_kind == L2R_L2LOSS_SVR_DUAL || classifier_kind == L2R_L1LOSS_SVR_DUAL )
			{
				copy_trn_X_Xf = trn_X_Xf;
				wts = new _float[ num_X ];
				fill( wts, wts+num_X, C ); 

				fy = new _float[ num_X ];
				fill( fy, fy+num_X, 0.0 );
				for( _int i=0; i < trn_Y_X->size[ l ]; i++ )
					fy[ trn_Y_X->data[l][i].first ] = trn_Y_X->data[l][i].second;
			}
			else if( weighted )
			{
				_float C1 = classifier_cost * (_float)num_trn;

				global_tab++;
				prepareWtMat( inwts, copy_trn_X_Xf, wts, y, C1, l);
				global_tab--;
			}
			else
			{
				copy_trn_X_Xf = trn_X_Xf;
				wts = new _float[ num_X ];
				fill( wts, wts+num_X, C ); 

				y = new _int[ num_X ];
				fill( y, y+num_X, -1 );
				for( _int i=0; i < trn_Y_X->size[ l ]; i++ )
					y[ trn_Y_X->data[l][i].first ] = +1;
			}

			LOG("prepared.", DEBUG);
			LOG("solving...", DEBUG);

			if( classifier_kind == L2R_L2LOSS_SVC_DUAL )
			{
				solve_l2r_l2loss_svc_dual( copy_trn_X_Xf, y, w, eps, wts, classifier_maxitr, reng );
			}
			else if( classifier_kind == L2R_LR_DUAL )
			{
				solve_l2r_lr_dual( copy_trn_X_Xf, y, w, eps, wts, classifier_maxitr, reng );
						}
			else if( classifier_kind == L2R_LR_PRIMAL )
			{
				solve_l2r_lr_primal( copy_trn_X_Xf, y, w, eps, wts, classifier_maxitr, reng );
			}
			else if( classifier_kind == L2R_L2LOSS_SVR_DUAL or classifier_kind == L2R_L1LOSS_SVR_DUAL )
			{
				string ckind;
				if( classifier_kind == L2R_L2LOSS_SVR_DUAL )
					ckind = "L2R_L2LOSS_SVR_DUAL";
				else if( classifier_kind == L2R_L1LOSS_SVR_DUAL )
					ckind = "L2R_L1LOSS_SVR_DUAL";
				
				solve_l2r_l1l2_svr_dual( copy_trn_X_Xf, fy, w, eps, P, wts, ckind, classifier_maxitr, reng );
			}
			/*
			else if( classifier_kind == L2R_L2LOSS_SVC_PRIMAL )
			{
				solve_l2r_l2loss_svc_primal( copy_trn_X_Xf, y, w, eps, wts, classifier_maxitr, reng );
			}
			*/

			LOG("solved.", DEBUG);
			LOG("thresholding...", DEBUG);
			_float max_val = -1, th;

			for( _int f=0; f<num_Xf; f++ )
			{
				_float val = fabs( w[f] );
				max_val = val>max_val ? val : max_val;
			}

			if( max_val == -1 )
				th = 0.0;
			else
				th = max_val*classifier_threshold;

			w_mat->data[ l ] = new pairIF[ num_Xf ]();
			_int siz = 0;

			for( _int f=0; f<num_Xf; f++ )
			{
				if( fabs( w[f] ) > th )
					w_mat->data[ l ][ siz++ ] = make_pair( f, w[f] );
			}
			Realloc( num_Xf, siz, w_mat->data[ l ] );
			w_mat->size[ l ] = siz;

			LOG("thresholding done.", DEBUG);

			delete [] wts;
			delete [] y;
			delete [] fy;

			if((weighted) and (not(classifier_kind == L2R_L2LOSS_SVR_DUAL or classifier_kind == L2R_L1LOSS_SVR_DUAL)))
				delete copy_trn_X_Xf;
		}

		delete [] w;

		return w_mat;
	}
};