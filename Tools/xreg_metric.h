#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "config.h"
#include "mat.h"
#include "utils.h"

using namespace std;

VecF pointwise_ranking_metric( SMatF* score_mat, SMatF* tst_X_Y, string type, int K, bool weighted = true)
{
	int num_X = score_mat->nc;
	int num_Y = score_mat->nr;

	VecF weights( num_Y, 0 );

	if( type == "ndcg" )
	{
		for( int i=0; i<num_Y; i++ )
			weights[i] = 1.0/log2( (float)i+2 );
	}
	else if( type == "prec" )
	{
		for( int i=0; i<num_Y; i++ )
			weights[i] = 1.0;
	}

	VecF labels( num_Y, 0 );
	VecF nmetrics( K, 0 );
	VecF dmetrics( K, 0 );
	VecF metrics( K, 0 );

	for( int i=0; i<num_X; i++ )
	{
		VecIF toplabels;
		for( int j=0; j<tst_X_Y->size[i]; j++ )
		{
			labels[ tst_X_Y->data[i][j].first ] = tst_X_Y->data[i][j].second;
			toplabels.push_back( tst_X_Y->data[i][j] );
		}
		sort( toplabels.begin(), toplabels.end(), comp_pair_by_second_desc<int,float> );
		
		for( int j=0; j<min((float)K,(float)toplabels.size()); j++ )
		{
			int ind = toplabels[j].first;
			float lval = labels[ ind ];
			if(weighted)
			{
				for( int k=j; k<K; k++ )
					dmetrics[k] += weights[j]*lval/num_X;
			}
		}

		VecIF scores;
		for( int j=0; j<score_mat->size[i]; j++ )
			scores.push_back( score_mat->data[i][j] );
		sort( scores.begin(), scores.end(), comp_pair_by_second_desc<int,float> );

		for( int j=0; j<min((float)K,(float)scores.size()); j++ )
		{
			int ind = scores[j].first;
			float lval = labels[ ind ];
			for( int k=j; k<K; k++ )
				nmetrics[k] += weights[j]*lval/num_X;
		}

		for( int j=0; j<tst_X_Y->size[i]; j++ )
			labels[ tst_X_Y->data[i][j].first ] = 0.0;
	}

	for( int i=0; i<K; i++ )
	{
		if(weighted)
			dmetrics[i] = dmetrics[i]==0?1.0 : dmetrics[i];
		else
			dmetrics[i] = ((float)(i+1));
		metrics[i] = nmetrics[i]/dmetrics[i];
	}

	return metrics;
}

VecF pairwise_ranking_metric( SMatF* score_mat, SMatF* tst_X_Y, int K )
{
	int num_X = score_mat->nc;
	int num_Y = score_mat->nr;

	VecF metrics( K, 0 );
	VecF nmetrics( K, 0 );
	VecF dmetrics( K, 0 );
	VecF flatscores( num_Y, 0 );
	VecF flatlabels( num_Y, 0 );
	float EPSS = 1e-6;

	for( int i=0; i<num_X; i++ )
	{
		VecIF labels;
		for( int j=0; j<tst_X_Y->size[i]; j++ )
		{
			int ind = tst_X_Y->data[i][j].first;
			float val = tst_X_Y->data[i][j].second;
			labels.push_back( tst_X_Y->data[i][j] );
			flatlabels[ ind ] = val;
		}

		VecIF scores;
		for( int j=0; j<score_mat->size[i]; j++ )
		{
			int ind = score_mat->data[i][j].first;
			float val = score_mat->data[i][j].second;
			scores.push_back( score_mat->data[i][j] );
			flatscores[ ind ] = val;
		}
		sort( scores.begin(), scores.end(), comp_pair_by_second_desc<int,float> );

		for( int j=0; j<min((float)K,(float)scores.size()); j++ )
		{
			int ind1 = scores[j].first;
			float score1 = scores[j].second;
			float label1 = flatlabels[ ind1 ];

			int errors = 0;
			for( int k=0; k<labels.size(); k++ )
			{
				int ind2 = labels[k].first;
				float label2 = labels[k].second;
				float score2 = flatscores[ ind2 ];

				/*
				if(j==0)
					cout << "\t" << label1 << ":" << score1 << "\t" << label2 << ":" << score2 << "\n";
				*/
		
				//if( score2<score1 && (score1-score2)*(label1-label2) < 0 )
				if( label1<label2-EPSS && score1>score2+EPSS && (score1-score2)*(label1-label2) < 0 )
				{
					errors++;
				}
			}

			for( int k=j; k<K; k++ )
			{
				/*
				if(k==0)
					cout << i << "\t" << errors << "\t" << labels.size() << "\n";
				*/

				nmetrics[k] += (float)errors/num_X;
				dmetrics[k] += (float)labels.size()/num_X;
			}
		}

		for( int j=0; j<tst_X_Y->size[i]; j++ )
			flatlabels[ tst_X_Y->data[i][j].first ] = 0.0;

		for( int j=0; j<score_mat->size[i]; j++ )
			flatscores[ score_mat->data[i][j].first ] = 0.0;
	}

	for( int i=0; i<K; i++ )
	{
		dmetrics[i] = dmetrics[i]==0?1.0 : dmetrics[i];
		metrics[i] = 1.0 - nmetrics[i]/dmetrics[i];
	}


	return metrics;

	/*
	cout << "extreme_ranking_metric" << endl;
	for( int i=0; i<K; i++ )
		cout << "\t" << (i+1) << ":\t" << metrics[i] << endl;
	*/
}

VecF regression_metric( SMatF* score_mat, SMatF* tst_X_Y, string type, int K )
{
	int num_X = score_mat->nc;
	int num_Y = score_mat->nr;

	VecF labels( num_Y, 0 );
	VecF metrics( K, 0 );  

	for( int i=0; i<num_X; i++ )
	{
		VecF inst_metrics( K, 0 );

		for( int j=0; j<tst_X_Y->size[i]; j++ )
			labels[ tst_X_Y->data[i][j].first ] = tst_X_Y->data[i][j].second;

		VecIF scores;
		for( int j=0; j<score_mat->size[i]; j++ )
			scores.push_back( score_mat->data[i][j] );
		sort( scores.begin(), scores.end(), comp_pair_by_second_desc<int,float> );

		for( int j=0; j<min((float)K,(float)scores.size()); j++ )
		{
			int ind = scores[j].first;
			double sval = scores[j].second;
			double lval = labels[ ind ];

			float dist = 0;
			if( type=="xkld" || type=="xrkld" )
			{
				//cout << "lval: " << lval << "\t" << "sval: " << sval << "\t";
				double eps = 1e-8;
				sval = sval>1.0-eps ? 1.0 - eps : sval;
				sval = sval<eps ? eps : sval;
				lval = lval>1.0-eps ? 1.0 - eps : lval;
				lval = lval<eps ? eps : lval;
				dist = lval * log( lval/sval ) + (1-lval) * log( (1-lval)/(1-sval) );
				//cout << "dist: " << dist << "\t" << "lval: " << lval << "\t" << "sval: " << sval << "\n";
			}
			else if( type=="xrmse" || type=="xrrmse" )
			{
				dist = SQ(lval-sval);
			}
			else if( type=="xmad" || type=="xrmad" )
			{
				dist = fabs( lval-sval );
			}

			/*
			for( int k=j; k<K; k++ )
				metrics[k] += dist/(float)num_X;
			*/

			for( int k=j; k<K; k++ )
				inst_metrics[k] += dist;
		}

		for( int j=0; j<K; j++ )
		{
			inst_metrics[ j ] /= (float)(j+1);
			if( type=="xrmse" || type=="xrrmse" )
				inst_metrics[ j ] = sqrt( inst_metrics[ j ] );

			metrics[j] += inst_metrics[j] / (float)num_X;
		}

		for( int j=0; j<tst_X_Y->size[i]; j++ )
			labels[ tst_X_Y->data[i][j].first ] = 0.0;
	}

	/*
	for( int i=0; i<K; i++ )
		metrics[i] /= (float)(i+1);
	*/

	if( type=="xrrmse" || type=="xrmad" || type=="xrkld" )
	{
		VecF prec = pointwise_ranking_metric( score_mat, tst_X_Y, "prec", K );
		for( int i=0; i<K; i++ )
			metrics[i] /= prec[i];
	}


	return metrics;
}

void extreme_regression_metric( SMatF* score_mat, SMatF* tst_X_Y, string regtype, string ranktype, int K )
{
	VecF regmetrics = regression_metric( score_mat, tst_X_Y, regtype, K );
	VecF rankmetrics;
	if( ranktype=="auc" )
		rankmetrics = pairwise_ranking_metric( score_mat, tst_X_Y, K );
	else
		rankmetrics = pointwise_ranking_metric( score_mat, tst_X_Y, ranktype, K );

	cout << "extreme_regression_metric: " << regtype << " " << ranktype << endl;
	for( float alpha=0.0; alpha<1.05; alpha+=0.05 )
	{
		cout << "alpha: " << alpha << endl;
		for( int i=0; i<K; i++ )
		{
			float xmetric = alpha*regmetrics[i] + (1-alpha)*rankmetrics[i];
			cout << "\t" << (i+1) << ":\t" << xmetric << endl;
		}
	}
}

void extreme_regression_metric_alpha( SMatF* score_mat, SMatF* tst_X_Y, string regtype, string ranktype, int K, float alpha )
{
	VecF regmetrics = regression_metric( score_mat, tst_X_Y, regtype, K );
	VecF rankmetrics;
	if( ranktype=="auc" )
		rankmetrics = pairwise_ranking_metric( score_mat, tst_X_Y, K );
	else
		rankmetrics = pointwise_ranking_metric( score_mat, tst_X_Y, ranktype, K );

	cout << "alpha: " << alpha << endl;
	for( int i=0; i<K; i++ )
	{
		float xmetric = alpha*regmetrics[i] + (1-alpha)*rankmetrics[i];
		cout << "\t" << (i+1) << ":\t" << xmetric << endl;
	}
}

void cpp_compute_all_metrics_helper_weighted( SMatF* score_mat, SMatF* tst_X_Y, int K )
{
	VecF metrics;

	string type = "prec";
	metrics = pointwise_ranking_metric( score_mat, tst_X_Y, type, K, true);
	cout << type << endl;
	for( int i=0; i<K; i++ )
	{
		float metric = metrics[i];
		cout << "\t" << (i+1) << ":\t" << 100.0*metric << endl;
	}

	type = "ndcg";
	metrics = pointwise_ranking_metric( score_mat, tst_X_Y, type, K );
	cout << type << endl;
	for( int i=0; i<K; i++ )
	{
		float metric = metrics[i];
		cout << "\t" << (i+1) << ":\t" << 100.0*metric << endl;
	}

	type = "auc";
	metrics = pairwise_ranking_metric( score_mat, tst_X_Y, K );
	cout << type << endl;
	for( int i=0; i<K; i++ )
	{
		float metric = metrics[i];
		cout << "\t" << (i+1) << ":\t" << metric << endl;
	}

	type = "xrrmse";
	metrics =  regression_metric( score_mat, tst_X_Y, type, K );
	cout << "xrmse" << endl;
	for( int i=0; i<K; i++ )
	{
		float metric = metrics[i];
		cout << "\t" << (i+1) << ":\t" << metric << endl;
	}
	
	// type = "xrmse";
	// metrics =  regression_metric( score_mat, tst_X_Y, type, K );
	// cout << type << endl;
	// for( int i=0; i<K; i++ )
	// {
	// 	float metric = metrics[i];
	// 	cout << "\t" << (i+1) << ":\t" << metric << endl;
	// }

	type = "xrmad";
	metrics =  regression_metric( score_mat, tst_X_Y, type, K );
	cout << "xmad" << endl;
	for( int i=0; i<K; i++ )
	{
		float metric = metrics[i];
		cout << "\t" << (i+1) << ":\t" << metric << endl;
	}

	// type = "xmad";
	// metrics =  regression_metric( score_mat, tst_X_Y, type, K );
	// cout << type << endl;
	// for( int i=0; i<K; i++ )
	// {
	// 	float metric = metrics[i];
	// 	cout << "\t" << (i+1) << ":\t" << metric << endl;
	// }

	// type = "xkld";
	// metrics =  regression_metric( score_mat, tst_X_Y, type, K );
	// cout << type << endl;
	// for( int i=0; i<K; i++ )
	// {
	// 	float metric = metrics[i];
	// 	cout << "\t" << (i+1) << ":\t" << metric << endl;
	// }

	// type = "xrkld";
	// metrics =  regression_metric( score_mat, tst_X_Y, type, K );
	// cout << type << endl;
	// for( int i=0; i<K; i++ )
	// {
	// 	float metric = metrics[i];
	// 	cout << "\t" << (i+1) << ":\t" << metric << endl;
	// }
}

void cpp_compute_all_metrics_helper_unweighted( SMatF* score_mat, SMatF* tst_X_Y, int K )
{
	VecF metrics;

	string type = "prec";
	metrics = pointwise_ranking_metric( score_mat, tst_X_Y, type, K, false);
	cout << type << endl;
	for( int i=0; i<K; i++ )
	{
		float metric = metrics[i];
		cout << "\t" << (i+1) << ":\t" << 100.0*metric << endl;
	}

	type = "prec";
	metrics = pointwise_ranking_metric( score_mat, tst_X_Y, type, K, true);
	cout << "psp" << endl;
	for( int i=0; i<K; i++ )
	{
		float metric = metrics[i];
		cout << "\t" << (i+1) << ":\t" << 100.0*metric << endl;
	}

	type = "ndcg";
	metrics = pointwise_ranking_metric( score_mat, tst_X_Y, type, K, true);
	cout << type << endl;
	for( int i=0; i<K; i++ )
	{
		float metric = metrics[i];
		cout << "\t" << (i+1) << ":\t" << 100.0*metric << endl;
	}
}

void cpp_compute_all_metrics( SMatF* score_mat, SMatF* tst_X_Y, int K, bool weighted)
{
	cout << "pointwise metrics" << endl;
	if(weighted)
		cpp_compute_all_metrics_helper_weighted( score_mat, tst_X_Y, K );
	else
		cpp_compute_all_metrics_helper_unweighted( score_mat, tst_X_Y, K );

	SMatF* t_score_mat = score_mat->transpose();
	SMatF* t_tst_X_Y = tst_X_Y->transpose();
	cout << "labelwise metrics" << endl;

	if(weighted)
		cpp_compute_all_metrics_helper_weighted( t_score_mat, t_tst_X_Y, K );
	else
		cpp_compute_all_metrics_helper_unweighted( t_score_mat, t_tst_X_Y, K );

	delete t_score_mat;
	delete t_tst_X_Y;
}