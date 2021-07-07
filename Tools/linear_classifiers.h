#pragma once

#include <iostream>
#include <string>
#include <cmath>

#include "config.h"
#include "utils.h"
#include "mat.h"
#include "timer.h"
#include "tron.h"

class sparse_operator
{
public:
	static _float nrm2_sq( _int siz, pairIF* x )
	{
		_float ret = 0;
		for( _int i=0; i<siz; i++ )
		{
			ret += SQ( x[i].second );
		}
		return (ret);
	}

	static _float dot( const _float *s, _int siz, pairIF* x )
	{
		_float ret = 0;
		for( _int i=0; i<siz; i++ )
			ret += s[ x[i].first ] * x[i].second;
		return (ret);
	}

	static void axpy(const _float a, _int siz, pairIF* x, _float *y)
	{
		for( _int i=0; i<siz; i++ )
		{
			y[x[i].first ] += a * x[i].second;
		}
	}
};

class l2r_lr_fun: public tron_function
{
public:
	l2r_lr_fun( SMatF* X_Xf, _int* y, _float* C );
	//l2r_lr_fun(const problem *prob, double *C);
	~l2r_lr_fun();

	_float fun( _float* w );
	void grad( _float* w, _float* g );
	void Hv( _float* s, _float* Hs );

	int get_nr_variable( void );
	void get_diagH( _float* M );

private:
	void Xv( _float* v, _float* Xv );
	void XTv( _float* v, _float* XTv );

	_float* C;
	_float* z;
	_float* D;
	SMatF* X_Xf;
	_int* y;
};

/*
class l2r_l2_svc_fun: public tron_function
{
public:
	l2r_l2_svc_fun(const problem *prob, double *C);
	~l2r_l2_svc_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);
	void get_diagH(double *M);

protected:
	void Xv(double *v, double *Xv);
	void subXTv(double *v, double *XTv);

	double *C;
	double *z;
	int *I;
	int sizeI;
	const problem *prob;
};
*/

void solve_l2r_lr_dual( SMatF* X_Xf, _int* y, _float *w, _float eps, _float* wts, _int classifier_maxitr, mt19937& reng );
void solve_l2r_l2loss_svc_dual( SMatF* X_Xf, _int* y, _float *w, _float eps, _float* wts, _int classifier_maxitr, mt19937& reng );
void solve_l2r_lr_primal( SMatF* X_Xf, _int* y, _float *w, _float eps, _float* wts, _int classifier_maxitr, mt19937& reng );
//void solve_l2r_l2loss_svc_primal( SMatF* X_Xf, _int* y, _float *w, _float eps, _float* wts, _int classifier_maxitr, mt19937& reng );
void solve_l1r_lr( SMatF* Xf_X, _int* y, _float *w, _float eps, _float* C, _int classifier_maxitr, mt19937& reng );
void solve_l2r_l1l2_svr_dual( SMatF* X_Xf, float* y, float* w, float eps, float p, float* wts, string solver_type, int classifier_maxitr, mt19937& reng );