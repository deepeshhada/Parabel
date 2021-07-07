#include "linear_classifiers.h"

#define GETI(i) (y[i]+1)

void solve_l1r_lr( SMatF* Xf_X, _int* y, _float *w, _float eps, _float* C, _int classifier_maxitr, mt19937& reng )
{
    _int num_X = Xf_X->nr;
    _int num_Xf = Xf_X->nc;
    vector<int> size = Xf_X->size;
    vector<pairIF*> data = Xf_X->data;
    
	_int l = num_X;
	_int w_size = num_Xf;
	_int newton_iter=0, iter=0;
	_int max_newton_iter = classifier_maxitr;
	_int max_iter = 10;
	_int max_num_linesearch = 20;
	_int active_size;
	_int QP_active_size;

	_double nu = 1e-12;
	_double inner_eps = 1;
	_double sigma = 0.01;
	_double w_norm, w_norm_new;
	_double z, G, H;
	_double Gnorm1_init;
	_double Gmax_old = INF;
	_double Gmax_new, Gnorm1_new;
	_double QP_Gmax_old = INF;
	_double QP_Gmax_new, QP_Gnorm1_new;
	_double delta, negsum_xTd, cond;

    VecI index( num_Xf, 0 );
    VecD Hdiag( num_Xf, 0 );
    VecD Grad( num_Xf, 0 );
    VecD wpd( num_Xf, 0 );
    VecD xjneg_sum( num_Xf, 0 );
    VecD xTd( num_X, 0 );
    VecD exp_wTx( num_X, 0 );
    VecD exp_wTx_new( num_X, 0 );
    VecD tau( num_X, 0 );
    VecD D( num_X, 0 );
	
	w_norm = 0;
	for( _int i=0; i<w_size; i++ )
	{
		index[i] = i;

        for( _int j=0; j<size[i]; j++ )
		{
			_int inst = data[i][j].first;
			_float val = data[i][j].second;

			if(y[inst] == -1)
				xjneg_sum[i] += C[inst]*val;
		}
	}

	for( _int i=0; i<l; i++ )
	{
		exp_wTx[i] = exp(exp_wTx[i]);
		_double tau_tmp = 1/(1+exp_wTx[i]);
		tau[i] = C[i]*tau_tmp;
		D[i] = C[i]*exp_wTx[i]*SQ(tau_tmp);
	}

	while(newton_iter < max_newton_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;
		active_size = w_size;

		for(_int s=0; s<active_size; s++)
		{
			_int i = index[s];
			Hdiag[i] = nu;

			_double tmp = 0;
		
			for( _int j=0; j<size[i]; j++ )
			{
				_int inst = data[i][j].first;
				_float val = data[i][j].second;
				Hdiag[i] += SQ(val)*D[inst];
				tmp += val*tau[inst];
			}

			Grad[i] = -tmp + xjneg_sum[i];

			_double Gp = Grad[i]+1;
			_double Gn = Grad[i]-1;
			_double violation = 0;

			if(w[i] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				//outer-level shrinking
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[i] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;
		}

		if(newton_iter == 0)
			Gnorm1_init = Gnorm1_new;

		if(Gnorm1_new <= eps*Gnorm1_init)
			break;

		iter = 0;
		QP_Gmax_old = INF;
		QP_active_size = active_size;

		for(_int i=0; i<l; i++)
			xTd[i] = 0;

		// optimize QP over wpd
		while(iter < max_iter)
		{
			QP_Gmax_new = 0;
			QP_Gnorm1_new = 0;

			for(_int i=0; i<QP_active_size; i++)
			{
				//_llint r = reng();
				//_int j = i+r%(QP_active_size-i);
				_int j = i + get_rand_num( QP_active_size-i, reng );
				swap(index[j], index[i]);
			}

			for(_int s=0; s<QP_active_size; s++)
			{
				_int i = index[s];
				H = Hdiag[i];

				G = Grad[i] + (wpd[i]-w[i])*nu;
				for( _int j=0; j<size[i]; j++ )
				{
					_int inst = data[i][j].first;
					_float val = data[i][j].second;
					G += val*D[inst]*xTd[inst];
				}

				_double Gp = G+1;
				_double Gn = G-1;
				_double violation = 0;
				if(wpd[i] == 0)
				{
					if(Gp < 0)
						violation = -Gp;
					else if(Gn > 0)
						violation = Gn;
					//inner-level shrinking
					else if(Gp>QP_Gmax_old/l && Gn<-QP_Gmax_old/l)
					{
						QP_active_size--;
						swap(index[s], index[QP_active_size]);
						s--;
						continue;
					}
				}
				else if(wpd[i] > 0)
					violation = fabs(Gp);
				else
					violation = fabs(Gn);

				QP_Gmax_new = max(QP_Gmax_new, violation);
				QP_Gnorm1_new += violation;

				// obtain solution of one-variable problem
				if(Gp < H*wpd[i])
					z = -Gp/H;
				else if(Gn > H*wpd[i])
					z = -Gn/H;
				else
					z = -wpd[i];

				if(fabs(z) < 1.0e-12)
					continue;
				z = min(max(z,-10.0),10.0);

				wpd[i] += z;

				for( _int j=0; j<size[i]; j++ )
				{
					_int inst = data[i][j].first;
					_float val = data[i][j].second;
					xTd[inst] += val*z;
				}
			}

			iter++;

			if(QP_Gnorm1_new <= inner_eps*Gnorm1_init)
			{
				//inner stopping
				if(QP_active_size == active_size)
					break;
				//active set reactivation
				else
				{
					QP_active_size = active_size;
					QP_Gmax_old = INF;
					continue;
				}
			}

			QP_Gmax_old = QP_Gmax_new;
		}

		delta = 0;
		w_norm_new = 0;
		for(_int i=0; i<w_size; i++)
		{
			delta += Grad[i]*(wpd[i]-w[i]);
			if(wpd[i] != 0)
				w_norm_new += fabs(wpd[i]);
		}
		delta += (w_norm_new-w_norm);

		negsum_xTd = 0;
		for(_int i=0; i<l; i++)
		{
			if(y[i] == -1)
				negsum_xTd += C[i]*xTd[i];
		}

		_int num_linesearch;
		for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
		{
			_double cond = w_norm_new - w_norm + negsum_xTd - sigma*delta;

			for(_int i=0; i<l; i++)
			{
				_double exp_xTd = exp(xTd[i]);
				exp_wTx_new[i] = exp_wTx[i]*exp_xTd;
				cond += C[i]*log((1+exp_wTx_new[i])/(exp_xTd+exp_wTx_new[i]));
			}

			if(cond <= 0)
			{
				w_norm = w_norm_new;
				for(_int i=0; i<w_size; i++)
					w[i] = wpd[i];

				for(_int i=0; i<l; i++)
				{
					exp_wTx[i] = exp_wTx_new[i];
					_double tau_tmp = 1/(1+exp_wTx[i]);
					tau[i] = C[i]*tau_tmp;
					D[i] = C[i]*exp_wTx[i]*SQ(tau_tmp);
				}
				break;
			}
			else
			{
				w_norm_new = 0;
				for(_int i=0; i<w_size; i++)
				{
					wpd[i] = (w[i]+wpd[i])*0.5;

					if(wpd[i] != 0)
						w_norm_new += fabs(wpd[i]);
				}
				delta *= 0.5;
				negsum_xTd *= 0.5;
				for(_int i=0; i<l; i++)
					xTd[i] *= 0.5;
			}
		}

		// Recompute some info due to too many line search steps
		if(num_linesearch >= max_num_linesearch)
		{
			for(_int i=0; i<l; i++)
				exp_wTx[i] = 0;

			for(_int i=0; i<w_size; i++)
			{
				if(w[i]==0) continue;

				for( _int j=0; j<size[i]; j++ )
				{
					_int inst = data[i][j].first;
					_float val = data[i][j].second;
					exp_wTx[inst] += w[i]*val;
				}
			}

			for(_int i=0; i<l; i++)
				exp_wTx[i] = exp(exp_wTx[i]);
		}

		if(iter == 1)
			inner_eps *= 0.25;

		newton_iter++;
		Gmax_old = Gmax_new;
	}
}

void solve_l2r_lr_dual( SMatF* X_Xf, _int* y, _float *w, _float eps, _float* wts, _int classifier_maxitr, mt19937& reng )
{
	_int l = X_Xf->nc;
	_int w_size = X_Xf->nr;
	_int i, s, iter = 0;

	_double *xTx = new _double[l];
	_int max_iter = classifier_maxitr;
	_int *index = new _int[l];	
	_double *alpha = new _double[2*l]; // store alpha and C - alpha
	_int max_inner_iter = 100; // for inner Newton
	_double innereps = 1e-2;
	_double innereps_min = min(1e-8, (_double)eps);

	vector<int> size = X_Xf->size;
	vector<pairIF*> data = X_Xf->data;

	// Initial alpha can be set here. Note that
	for(i=0; i<l; i++)
	{
		alpha[2*i] = min(0.001*wts[i], 1e-8);
		alpha[2*i+1] = wts[i] - alpha[2*i];
	}

	for(i=0; i<l; i++)
	{
		xTx[i] = sparse_operator::nrm2_sq( size[i], data[i] );
		sparse_operator::axpy(y[i]*alpha[2*i], size[i], data[i], w);
		index[i] = i;
	}

	while (iter < max_iter)
	{
		for (i=0; i<l; i++)
		{
			_int j = i + get_rand_num( l-i, reng );
			swap(index[i], index[j]);
		}

		_int newton_iter = 0;
		_double Gmax = 0;
		for (s=0; s<l; s++)
		{
			i = index[s];
			const _int yi = y[i];
			_double C = wts[i];
			_double ywTx = 0, xisq = xTx[i];
			ywTx = yi*sparse_operator::dot( w, size[i], data[i] );
			_double a = xisq, b = ywTx;

			// Decide to minimize g_1(z) or g_2(z)
			_int ind1 = 2*i, ind2 = 2*i+1, sign = 1;
			if(0.5*a*(alpha[ind2]-alpha[ind1])+b < 0)
			{
				ind1 = 2*i+1;
				ind2 = 2*i;
				sign = -1;
			}

			_double alpha_old = alpha[ind1];
			_double z = alpha_old;
			if(C - z < 0.5 * C)
				z = 0.1*z;
			_double gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
			Gmax = max(Gmax, fabs(gp));

			// Newton method on the sub-problem
			const _double eta = 0.1; // xi in the paper
			_int inner_iter = 0;
			while (inner_iter <= max_inner_iter)
			{
				if(fabs(gp) < innereps)
					break;
				_double gpp = a + C/(C-z)/z;
				_double tmpz = z - gp/gpp;
				if(tmpz <= 0)
					z *= eta;
				else // tmpz in (0, C)
					z = tmpz;
				gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
				newton_iter++;
				inner_iter++;
			}

			if(inner_iter > 0) // update w
			{
				alpha[ind1] = z;
				alpha[ind2] = C-z;
				sparse_operator::axpy(sign*(z-alpha_old)*yi, size[i], data[i], w);
			}
		}

		iter++;

		if(Gmax < eps)
			break;

		if(newton_iter <= l/10)
			innereps = max(innereps_min, 0.1*innereps);

	}

	delete [] xTx;
	delete [] alpha;
	delete [] index;

	/*
	float sum_wts = 0;
	for( int i=0; i<l; i++ )
		sum_wts += wts[i];
	float A = 1.0/(sum_wts);

	float obj = 0;
	for( int i=0; i<w_size; i++ )
		obj += SQ( w[i] );
	obj *= 1/2.0;

	for( int i=0; i<l; i++ )
	{
		float score = 0;
		for( int j=0; j<size[i]; j++ )
			score += w[ data[i][j].first ] * data[i][j].second;
		obj += wts[i]*log( 1 + exp( -y[i]*score ) );
	}
	obj *= A;

	cout << "\t" << obj << "\n";
	*/
}

void solve_l2r_lr_sgd1( SMatF* X_Xf, _float common_weight, _float *w, _float eps, _float* wts, _int classifier_maxitr, mt19937& reng )
{
	classifier_maxitr = 20;
	int N = X_Xf->nc;
	int D = X_Xf->nr;

	for( int i=0; i<D; i++ )
		w[i] = 0.0;

	if( N==0 )
		return;

	float A = 1.0/common_weight;
	float L0 = 0.5;
	float Discount = 1000.0;
	int itr = 0;
	float wcoeff = 1.0;
	float obj = 1000.0, new_obj = 0;

	uniform_int_distribution<int> distribution( 0, N-1 );

	vector<int> size = X_Xf->size;
	vector<pairIF*> data = X_Xf->data;

	for( int outer_itr=0; outer_itr<classifier_maxitr; outer_itr++ )
	{
		for( int j=0; j<D; j++ )
			w[j] *= wcoeff;
		wcoeff = 1.0;

		for( int i=0; i<N; i++ )
		{
			itr++;
			int inst = distribution( reng );
			float score = 0;
			for( int j=0; j<size[inst]; j++ )
				score += w[ data[inst][j].first ] * data[inst][j].second;
			score *= wcoeff;
			float wt = wts[inst];
			float prob_coeff = wt*(-1/( 1 + exp(score) )) + (1-wt)*(1/( 1 + exp(-score) ));
			float L = L0/sqrt(1.0 + ((float)itr/Discount));
			wcoeff *= (1-L*A);
			assert( wcoeff > 0.0 );
			for( int j=0; j<size[inst]; j++ )
				w[ data[inst][j].first ] -= L/wcoeff * prob_coeff * data[inst][j].second;

			if( itr%1000==0 )
			{
				for( int j=0; j<D; j++ )
					w[j] *= wcoeff;
				wcoeff = 1.0;
			}
		}
	}

	for( int j=0; j<D; j++ )
		w[j] *= wcoeff;
	wcoeff = 1.0;
}

void solve_l2r_lr_top_sgd( SMatF* X_Xf, _float common_weight, _float *w, _float eps, _float* wts, _int classifier_maxitr, mt19937& reng )
{
	float top_frac = 0.05;

	classifier_maxitr = 100;
	int N = X_Xf->nc;
	int D = X_Xf->nr;

	for( int i=0; i<D; i++ )
		w[i] = 0.0;

	if( N==0 )
		return;

	float A = 1.0/common_weight;
	float L0 = 0.5;
	float Discount = 1000.0;
	int itr = 0;
	float wcoeff = 1.0;
	float obj = 1000.0, new_obj = 0;
	float lambda = 0.0;

	uniform_int_distribution<int> distribution( 0, N-1 );

	vector<int> size = X_Xf->size;
	vector<pairIF*> data = X_Xf->data;

	for( int outer_itr=0; outer_itr<classifier_maxitr; outer_itr++ )
	{
		for( int j=0; j<D; j++ )
			w[j] *= wcoeff;
		wcoeff = 1.0;

		for( int i=0; i<N; i++ )
		{
			itr++;
			int inst = distribution( reng );
			float score = 0;
			for( int j=0; j<size[inst]; j++ )
				score += w[ data[inst][j].first ] * data[inst][j].second;
			score *= wcoeff;
			float wt = wts[inst];
			
			float sig_score = 1 + exp( -score );
			float grad_coeff = 1.0/sig_score - wt;
			float obj = log( sig_score ) + score*(1-wt);
			int active_obj = (int)(obj > lambda);
			grad_coeff *= active_obj;
			//float grad_coeff = wt*(-1/( 1 + exp(score) )) + (1-wt)*(1/( 1 + exp(-score) ));
			
			float L = L0/sqrt(1.0 + ((float)itr/Discount));
			wcoeff *= (1-L*A);
			assert( wcoeff > 0.0 );
			for( int j=0; j<size[inst]; j++ )
				w[ data[inst][j].first ] -= L/wcoeff * grad_coeff * data[inst][j].second;

			lambda -= L*( top_frac - active_obj );
			lambda = lambda<0.0 ? 0.0 : lambda;

			if( itr%1000==0 )
			{
				for( int j=0; j<D; j++ )
					w[j] *= wcoeff;
				wcoeff = 1.0;
			}
		}
	}

	for( int j=0; j<D; j++ )
		w[j] *= wcoeff;
	wcoeff = 1.0;
}

void solve_l2r_lr_sgd( SMatF* X_Xf, _int* y, _float *w, _float eps, _float* wts, _int classifier_maxitr, mt19937& reng )
{
	classifier_maxitr = 20;
	int N = X_Xf->nc;
	int D = X_Xf->nr;

	for( int i=0; i<D; i++ )
		w[i] = 0.0;

	if( N==0 )
		return;

	float sum_wts = 0;
	for( int i=0; i<N; i++ )
		sum_wts += wts[i];

	float A = 1.0/(sum_wts);
	float L0 = 0.5;
	float Discount = 1000.0;
	int itr = 0;
	float wcoeff = 1.0;
	float obj = 1000.0, new_obj = 0;

	discrete_distribution<int> distribution( wts, wts+N );

	vector<int> size = X_Xf->size;
	vector<pairIF*> data = X_Xf->data;

	for( int outer_itr=0; outer_itr<classifier_maxitr; outer_itr++ )
	{
		for( int j=0; j<D; j++ )
			w[j] *= wcoeff;
		wcoeff = 1.0;

		for( int i=0; i<N; i++ )
		{
			itr++;
			int inst = distribution( reng );
			float score = 0;
			for( int j=0; j<size[inst]; j++ )
				score += w[ data[inst][j].first ] * data[inst][j].second;
			score *= wcoeff;
			float yinst = y[inst];
			float prob_coeff = -yinst/( 1 + exp(yinst*score) );
			float L = L0/sqrt(1.0 + ((float)itr/Discount));
			wcoeff *= (1-L*A);
			assert( wcoeff > 0.0 );
			for( int j=0; j<size[inst]; j++ )
				w[ data[inst][j].first ] -= L/wcoeff * prob_coeff * data[inst][j].second;

			if( itr%1000==0 )
			{
				for( int j=0; j<D; j++ )
					w[j] *= wcoeff;
				wcoeff = 1.0;
			}
		}
	}

	for( int j=0; j<D; j++ )
		w[j] *= wcoeff;
	wcoeff = 1.0;
}


void solve_l2r_l2loss_svc_dual( SMatF* X_Xf, _int* y, _float *w, _float eps, _float* wts, _int classifier_maxitr, mt19937& reng )
{
	_int l = X_Xf->nc;
	_int w_size = X_Xf->nr;

	_int i, s, iter = 0;
	_float C, d, G;
	_float *QD = new _float[l];
	_int max_iter = classifier_maxitr;
	_int *index = new _int[l];
	_float *alpha = new _float[l];
	_int active_size = l;

	_int tot_iter = 0;

	// PG: projected gradient, for shrinking and stopping
	_float PG;
	_float PGmax_old = INF;
	_float PGmin_old = -INF;
	_float PGmax_new, PGmin_new;

	// default solver_type: L2R_L2LOSS_SVC_DUAL

	_float* diag = new _float[l];
        for( _int i=0; i<l; i++ )
                diag[i] = (_float)0.5/wts[i];
	_float upper_bound[3] = {INF, 0, INF};

	vector<int> size = X_Xf->size;
	vector<pairIF*> data = X_Xf->data;

	//d = pwd;
	//Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]

	for(i=0; i<l; i++)
		alpha[i] = 0;

	for(i=0; i<l; i++)
	{
		QD[i] = diag[i];
		QD[i] += sparse_operator::nrm2_sq( size[i], data[i] );
		sparse_operator::axpy(y[i]*alpha[i], size[i], data[i], w);
		index[i] = i;
	}

	while (iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

		for (i=0; i<active_size; i++)
		{
			_int j = i + get_rand_num( active_size-i, reng );
			swap(index[i], index[j]);
		}

		for (s=0; s<active_size; s++)
		{
			tot_iter ++;

			i = index[s];
			const _int yi = y[i];

			G = yi*sparse_operator::dot( w, size[i], data[i] )-1;

			C = upper_bound[GETI(i)];
			G += alpha[i]*diag[i];

			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				_float alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], (_float)0.0), C);
				d = (alpha[i] - alpha_old)*yi;
				sparse_operator::axpy(d, size[i], data[i], w);
			}
		}

		iter++;

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}

	// calculate objective value

	delete [] diag;
	delete [] QD;
	delete [] alpha;
	delete [] index;
}


//static void solve_l2r_l1l2_svr_dual(	const problem *prob, double *w, const parameter *param,	int solver_type)
void solve_l2r_l1l2_svr_dual( SMatF* X_Xf, float* y, float* w, float eps, float p, float* wts, string solver_type, int classifier_maxitr, mt19937& reng )
{
	int l = X_Xf->nc;
	int w_size = X_Xf->nr;

	//int l = prob->l;
	//double C = param->C;
	//double p = param->p;
	//int w_size = prob->n;
	//double eps = param->eps;
	int i, s, iter = 0;
	//int max_iter = 1000;
	int max_iter = classifier_maxitr;
	int active_size = l;
	int *index = new int[l];

	double d, G, H;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double *beta = new double[l];
	double *QD = new double[l];
	//double *y = prob->y;

	// L2R_L2LOSS_SVR_DUAL
	//double lambda[1], upper_bound[1];
	//lambda[0] = 0.5/C;
	//upper_bound[0] = INF;

	double* lambda = new double[l];
	for( int i=0; i<l; i++ )
		lambda[i] = 0.5/wts[i];

	double* upper_bound = new double[l];
	for( int i=0; i<l; i++ )
		upper_bound[i] = INF;

	if(solver_type == "L2R_L1LOSS_SVR_DUAL")
	{
		//lambda[0] = 0;
		for( int i=0; i<l; i++ )
		{
			lambda[i] = 0;
			//upper_bound[0] = C;
			upper_bound[i] = wts[i];
		}
		
	}

	vector<int> size = X_Xf->size;
    vector<pairIF*> data = X_Xf->data;

	// Initial beta can be set here. Note that
	// -upper_bound <= beta[i] <= upper_bound
	for(i=0; i<l; i++)
		beta[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		QD[i] = sparse_operator::nrm2_sq( size[i], data[i] );
		sparse_operator::axpy(beta[i], size[i], data[i], w);
		//feature_node * const xi = prob->x[i];
		//QD[i] = sparse_operator::nrm2_sq(xi);
		//sparse_operator::axpy(beta[i], xi, w);
		index[i] = i;
	}


	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(i=0; i<active_size; i++)
		{
			int j = i + get_rand_num( active_size-i, reng );
			//int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			i = index[s];
			//G = -y[i] + lambda[GETI(i)]*beta[i];
			//H = QD[i] + lambda[GETI(i)];
			G = -y[i] + lambda[i]*beta[i];
			H = QD[i] + lambda[i];

			G += sparse_operator::dot( w, size[i], data[i] );
			//feature_node * const xi = prob->x[i];
			//G += sparse_operator::dot(w, xi);

			double Gp = G+p;
			double Gn = G-p;
			double violation = 0;
			if(beta[i] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old && Gn<-Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] >= upper_bound[i])
			{
				if(Gp > 0)
					violation = Gp;
				else if(Gp < -Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] <= -upper_bound[i])
			{
				if(Gn < 0)
					violation = -Gn;
				else if(Gn > Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*beta[i])
				d = -Gp/H;
			else if(Gn > H*beta[i])
				d = -Gn/H;
			else
				d = -beta[i];

			if(fabs(d) < 1.0e-12)
				continue;

			double beta_old = beta[i];
			beta[i] = min(max(beta[i]+d, -upper_bound[i]), upper_bound[i]);
			d = beta[i]-beta_old;

			if(d != 0)
				sparse_operator::axpy(d, size[i], data[i], w);
				//sparse_operator::axpy(d, xi, w);
		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;

		/*
		if(iter % 10 == 0)
			info(".");
		*/

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				//info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	/*
	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 11 may be faster\n\n");
	*/

	/*
	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0; i<l; i++)
	{
		v += p*fabs(beta[i]) - y[i]*beta[i] + 0.5*lambda[GETI(i)]*beta[i]*beta[i];
		if(beta[i] != 0)
			nSV++;
	}

	info("Objective value = %lf\n", v);
	info("nSV = %d\n",nSV);
	*/

	delete [] beta;
	delete [] QD;
	delete [] index;
	delete [] lambda;
	delete [] upper_bound;
}

l2r_lr_fun::l2r_lr_fun( SMatF* X_Xf, _int* y, _float* C )
{
	this->X_Xf = X_Xf;
	this->y = y;
	_int l = X_Xf->nc;
	z = new _float[ l ];
	D = new _float[ l ];
	this->C = C;
}

l2r_lr_fun::~l2r_lr_fun()
{
	delete[] z;
	delete[] D;
}

_float l2r_lr_fun::fun( _float* w )
{
	_float f=0;
	_int l = X_Xf->nc;
	_int w_size = X_Xf->nr;

	Xv(w, z);

	for( _int i=0; i<w_size; i++ )
		f += w[i]*w[i];
	f /= 2.0;

	for( _int i=0; i<l; i++ )
	{
		_float yz = y[i]*z[i];
		if (yz >= 0)
			f += C[i]*log(1 + exp(-yz));
		else
			f += C[i]*(-yz+log(1 + exp(yz)));
	}

	return f;
}

void l2r_lr_fun::grad( _float* w, _float* g )
{
	_int l = X_Xf->nc;
	_int w_size = X_Xf->nr;

	for( _int i=0; i<l; i++ )
	{
		z[i] = 1/(1 + exp(-y[i]*z[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C[i]*(z[i]-1)*y[i];
	}
	XTv(z, g);

	for( _int i=0; i<w_size; i++ )
		g[i] = w[i] + g[i];
}

_int l2r_lr_fun::get_nr_variable(void)
{
	return X_Xf->nr;
}

void l2r_lr_fun::Hv( _float* s, _float* Hs )
{
	_int l = X_Xf->nc;
	_int w_size = X_Xf->nr;

	for( _int i=0; i<w_size; i++ )
		Hs[i] = 0;

	for( _int i=0; i<l; i++ )
	{
		_float xTs = sparse_operator::dot( s, X_Xf->size[i], X_Xf->data[i] );
		xTs = C[i]*D[i]*xTs;
		sparse_operator::axpy( xTs, X_Xf->size[i], X_Xf->data[i], Hs );
	}
	for( _int i=0; i<w_size; i++ )
		Hs[i] = s[i] + Hs[i];
}

void l2r_lr_fun::get_diagH( _float* M )
{
	_int l = X_Xf->nc;
	_int w_size = X_Xf->nr;

	for ( _int i=0; i<w_size; i++ )
		M[i] = 1;

	for ( _int i=0; i<l; i++)
	{
		for( _int j=0; j<X_Xf->size[i]; j++ )
		{
			_int id = X_Xf->data[i][j].first;
			_float val = X_Xf->data[i][j].second;
			M[ id ] += SQ( val )*C[i]*D[i];
		}
	}
}

void l2r_lr_fun::Xv( _float* v, _float* Xv )
{
	_int l = X_Xf->nc;
	_int w_size = X_Xf->nr;

	for( _int i=0; i<l; i++ )
		Xv[i] = sparse_operator::dot( v, X_Xf->size[i], X_Xf->data[i] );
}

void l2r_lr_fun::XTv( _float* v, _float* XTv )
{
	_int l = X_Xf->nc;
	_int w_size = X_Xf->nr;

	for( _int i=0; i<w_size; i++ )
		XTv[i]=0;

	for( _int i=0; i<l; i++ )
		sparse_operator::axpy( v[i], X_Xf->size[i], X_Xf->data[i], XTv );
}

void solve_l2r_lr_primal( SMatF* X_Xf, _int* y, _float *w, _float eps, _float* wts, _int classifier_maxitr, mt19937& reng ) // Default value of eps is 0.1
{
	l2r_lr_fun* fun_obj = new l2r_lr_fun( X_Xf, y, wts );
	TRON tron_obj( fun_obj, 0.05, 0.05 );
	tron_obj.tron( w );
	delete fun_obj;
}


/*
l2r_l2_svc_fun::l2r_l2_svc_fun(const problem *prob, double *C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	I = new int[l];
	this->C = C;
}

l2r_l2_svc_fun::~l2r_l2_svc_fun()
{
	delete[] z;
	delete[] I;
}

double l2r_l2_svc_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
	{
		z[i] = y[i]*z[i];
		double d = 1-z[i];
		if (d > 0)
			f += C[i]*d*d;
	}

	return(f);
}

void l2r_l2_svc_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}

int l2r_l2_svc_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_l2_svc_fun::get_diagH(double *M)
{
	int i;
	int w_size=get_nr_variable();
	feature_node **x = prob->x;

	for (i=0; i<w_size; i++)
		M[i] = 1;

	for (i=0; i<sizeI; i++)
	{
		int idx = I[i];
		feature_node *s = x[idx];
		while (s->index!=-1)
		{
			M[s->index-1] += s->value*s->value*C[idx]*2;
			s++;
		}
	}
}

void l2r_l2_svc_fun::Hv(double *s, double *Hs)
{
	int i;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		Hs[i]=0;
	for(i=0;i<sizeI;i++)
	{
		feature_node * const xi=x[I[i]];
		double xTs = sparse_operator::dot(s, xi);

		xTs = C[I[i]]*xTs;

		sparse_operator::axpy(xTs, xi, Hs);
	}
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*Hs[i];
}

void l2r_l2_svc_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
		Xv[i]=sparse_operator::dot(v, x[i]);
}

void l2r_l2_svc_fun::subXTv(double *v, double *XTv)
{
	int i;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
		sparse_operator::axpy(v[i], x[I[i]], XTv);
}
*/