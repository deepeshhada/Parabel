#include "xreg.h"

using namespace std;

/********************** TRAINING CODE **********************/

SMatF* partition_to_assign_mat( SMatF* Y_X, VecI& partition, VecI& countmap)
{
	_int num_Y = Y_X->nc;
	_int num_X = Y_X->nr;

	VecI pos_Y, neg_Y;
	for( _int i=0; i<num_Y; i++ )
	{
		if( partition[i]==1 )
			pos_Y.push_back( i );
		else
			neg_Y.push_back( i );
	}

	VecI pos_X, neg_X, pos_counts, neg_counts;
	Y_X->active_dims( pos_Y, pos_X, pos_counts, countmap );
	Y_X->active_dims( neg_Y, neg_X, neg_counts, countmap );

	SMatF* assign_mat = new SMatF( num_X, 2 );
	vector<_int>& size = assign_mat->size;
	vector<pairIF*>& data = assign_mat->data;
	
	size[0] = pos_X.size();
	size[1] = neg_X.size();
	data[0] = new pairIF[ pos_X.size() ];
	data[1] = new pairIF[ neg_X.size() ];

	for( _int i=0; i<pos_X.size(); i++)
		data[0][i] = make_pair( pos_X[i], 1 );

	for( _int i=0; i<neg_X.size(); i++)
		data[1][i] = make_pair( neg_X[i], 1 );

	return assign_mat;
}

SMatF* wt_partition_to_assign_mat( SMatF* Y_X, VecI& partition, _int nodeno, VecF& all_max, VecI& countmap)
{
	SMatF* assign_mat = partition_to_assign_mat( Y_X, partition, countmap);
	_int num_Y = Y_X->nc;
	_int num_X = Y_X->nr;

	VecI pos_ind, neg_ind, all_ind;
	for( _int i=0; i<num_Y; i++ )
	{
		all_ind.push_back( i );
		if( partition[i]==1 )
			pos_ind.push_back( i );
		else
			neg_ind.push_back( i );
	}

	VecF pos_max = Y_X->max( pos_ind, 1 );
	VecF neg_max = Y_X->max( neg_ind, 1 );
	// VecF all_max;

	if( nodeno==0 )
	{
		all_max.resize( num_X );
		fill( all_max.begin(), all_max.end(), 1.0 );
	}
	else
	{
		all_max = Y_X->max( all_ind, 1 );
	}

	// TODO : can be transformed into divison of column by dense vector
	for( _int i=0; i<assign_mat->nc; i++ )
	{
		for( _int j=0; j<assign_mat->size[i]; j++ )
		{
			_int ind = assign_mat->data[i][j].first;
			if( i==0 )
				assign_mat->data[i][j].second = pos_max[ ind ] / all_max[ ind ];
			else
				assign_mat->data[i][j].second = neg_max[ ind ] / all_max[ ind ];
		}
	}
	return assign_mat;
}

// learn classifiers for leaf node
void train_leaf_classifiers( Node* node, SMatF* X_Xf, SMatF* Y_X, _int nr, VecI& n_Xf, Param& param, bool X_Xf_deep_copy, mt19937& reng)
{
	_int num_Y = Y_X->nc;
	_int num_X = Y_X->nr;

	VecI all_ind;
	for( _int i=0; i<num_Y; i++ )
		all_ind.push_back( i );

	VecF all_max = Y_X->max( all_ind, 1 );

	SMatF* w_mat = NULL;

	if(param.leaf_classifier_kind == L2R_LR_DUAL or param.leaf_classifier_kind == L2R_LR_PRIMAL)
	{
		// TODO : can be transformed into divison of column by dense vector
		for( _int i=0; i<num_Y; i++ )
		{
			for( _int j=0; j<Y_X->size[i]; j++ )
			{
				_int ind = Y_X->data[i][j].first;
				_float val = Y_X->data[i][j].second;
				if( all_max[ind] > 0.0 )
					Y_X->data[i][j].second /= all_max[ ind ];
			}
		}

		LinearClassifier linear_classifier(X_Xf, Y_X, param, param.weighted, param.leaf_classifier_kind);
		w_mat = linear_classifier.train(all_max, reng);
	}
	else
	{
		LinearClassifier linear_classifier(X_Xf, Y_X, param, false /*explicit unweighted*/, param.leaf_classifier_kind);
		w_mat = linear_classifier.train(all_max, reng, true);
	}

	// TODO : comments
 	if(X_Xf_deep_copy)
		w_mat->reindex_rows( nr, n_Xf );

	node->w = w_mat;
}

// learn classifiers for an internal node
void train_internal_classifiers(Node* node, SMatF* X_Xf, SMatF* Y_X, _int nr, VecI& n_Xf, VecI& partition, Param& param, _int nodeno, VecI& countmap, bool X_Xf_deep_copy, mt19937& reng)
{
	SMatF* assign_mat;
	SMatF* w_mat;
	VecF wts;	

	if(param.classifier_kind == L2R_L2LOSS_SVC_DUAL or param.classifier_kind == L2R_L2LOSS_SVC_PRIMAL)
	{
		assign_mat = partition_to_assign_mat( Y_X, partition, countmap );

		LinearClassifier linear_classifier(X_Xf, assign_mat, param, false /*explicit unweighted*/);
		w_mat = linear_classifier.train(wts, reng);
	}
	else if(param.classifier_kind == L2R_LR_DUAL or param.classifier_kind == L2R_LR_PRIMAL)
	{
		assign_mat = wt_partition_to_assign_mat( Y_X, partition, nodeno, wts, countmap );

		LinearClassifier linear_classifier(X_Xf, assign_mat, param, true /* explicit weighted*/);
		w_mat = linear_classifier.train(wts, reng);
	}

	// X_Xf was deep copy then need to reindex rows to original indices otherwise if X_Xf was shallow copy then no need
	if(X_Xf_deep_copy)
		w_mat->reindex_rows( nr, n_Xf );

	node->w = w_mat;
	delete assign_mat; assign_mat = NULL;
}

/******************** BALANCED KMEANS ********************/
void add_s_to_d_vec3( pairIF* svec, _int siz, _float wt, _float* dvec)
{
	for( _int i=0; i<siz; i++ )
	{
		_int id = svec[i].first;
		_float val = svec[i].second;
		dvec[id] += (val * wt);
	}
}

pair<_float, _float> get_cosine_sim3(_int j, _float** cosines, SMatF* Y_X, vector<_float>& normalizers, VecI& n_Y)
{
	_float res0 = 0.0;
	_float res1 = 0.0;

	for(_int i = 0; i < Y_X->size[j]; i++)
	{
		_int x = Y_X->data[j][i].first;
		_float ctr = Y_X->data[j][i].second;
		res0 += cosines[0][x] * ctr;
		res1 += cosines[1][x] * ctr;
	}

	res0 /= normalizers[n_Y[j]];
	res1 /= normalizers[n_Y[j]];

	return make_pair(res0, res1);
}

void set_centers(SMatF* n_X_Xf, SMatF* n_Y_X, vector<_int> centers_indices, _float** centers, vector<_float>& centers_normalizing_factors)
{
	_int x;
	_float ctr;

	_int nr = n_X_Xf->nr;

	_int center_num = 0;
	for(_int col_index : centers_indices)
	{
		for(_int i = 0; i < n_Y_X->size[col_index]; i++)
		{
			x = n_Y_X->data[col_index][i].first;
			ctr = n_Y_X->data[col_index][i].second;
			for(_int j = 0; j < n_X_Xf->size[x]; j++)
				centers[center_num][n_X_Xf->data[x][j].first] += n_X_Xf->data[x][j].second * ctr;
		}
		for(_int i = 0; i < nr; i++)
			centers[center_num][i] /= centers_normalizing_factors[center_num];
		center_num += 1;
	}
}

void balanced_kmeans(SMatF* n_X_Xf, SMatF* n_Y_X, VecI& n_Y, vector<_float>& normalizers, _float acc, VecI& partition, VecI& countmap, mt19937& reng)
{
	_int nc = n_Y_X->nc;
	_int nr = n_X_Xf->nr;
	_int num_X = n_Y_X->nr;

 	vector<_int> centers_indices(2, -1);  // generate two random indices in range [0, nc]
	centers_indices[0] = get_rand_num(nc, reng);
	centers_indices[1] = centers_indices[0];
	while(centers_indices[1] == centers_indices[0])
		centers_indices[1] = get_rand_num(nc, reng);

	_float** centers;
	init_2d_float(2, nr, centers);
	reset_2d_float(2, nr, centers);

	vector<_float> centers_normalizing_factors({normalizers[n_Y[centers_indices[0]]], normalizers[n_Y[centers_indices[1]]]});
	set_centers(n_X_Xf, n_Y_X, centers_indices, centers, centers_normalizing_factors);
	
	centers[0][nr - 1] = 0.0;  // to nullify the effect of the bias term in X_Xf
	centers[1][nr - 1] = 0.0;

	partition.resize(nc);

	_float** cosines;
	init_2d_float(2, num_X, cosines);

	_float** centroid_cosines;
	init_2d_float(2, nc, centroid_cosines);
	
	pairIF* dcosines = new pairIF[nc];

	_float old_cos = -10000;
	_float new_cos = -1;

	while(new_cos - old_cos >= acc)
	{
		reset_2d_float(2, num_X, cosines);
		fill(countmap.begin(), countmap.end(), 0);

		for(_int i = 0; i < n_Y.size(); i++)  // calculate the cosines for all the points
		{
			for(_int j = 0; j < n_Y_X->size[i]; j++)
			{
				_int x = n_Y_X->data[i][j].first;
				if(countmap[x] == 0)  // calculate cosines for this x if not already done, fails if the cosine is 0
				{
					cosines[0][x] = mult_d_s_vec(centers[0], n_X_Xf->data[x], n_X_Xf->size[x]);
					cosines[1][x] = mult_d_s_vec(centers[1], n_X_Xf->data[x], n_X_Xf->size[x]);
					countmap[x] = 1;	
				}
			}
		}
		
		for(_int i=0; i < nc; i++)  // get cosines for lables as the ctr weighted average of the positive points
		{
			dcosines[i].first = i;
			pair<_float, _float> sims = get_cosine_sim3(i, cosines, n_Y_X, normalizers, n_Y);
			dcosines[i].second = sims.first - sims.second;
			centroid_cosines[0][i] = sims.first;
			centroid_cosines[1][i] = sims.second;
		}
		
		sort(dcosines, dcosines+nc, comp_pair_by_second_desc<_int,_float>);

		reset_2d_float(2, nr, centers);
		old_cos = new_cos;

		_int id, part, p;
		new_cos = 0.0;
		vector<vector<float>> wts(2, vector<float>(num_X, 0.0));

		for( _int i=0; i<nc; i++ )
		{
			id = dcosines[i].first;
			part = (_int)(i < nc/2);
			p = 1 - part;
			partition[id] = p;
			new_cos += centroid_cosines[p][id];

			for(_int j = 0; j < n_Y_X->size[id]; j++)
			{
				_int x = n_Y_X->data[id][j].first;
				_float ctr = n_Y_X->data[id][j].second;

				wts[p][x] += (ctr/normalizers[n_Y[id]]);
			}
		}

		for(_int p = 0; p < 2; ++p)
			for(_int x = 0; x < num_X; ++x)
				add_s_to_d_vec3(n_X_Xf->data[x], n_X_Xf->size[x], wts[p][x], centers[p]);

		new_cos /= nc;

		centers[0][nr - 1] = 0.0;
		centers[1][nr - 1] = 0.0;

		for( _int i=0; i<2; i++)
			normalize_d_vec(centers[i], nr);
	}

	delete_2d_float(2, nr, centers);
	delete_2d_float(2, nr, cosines);
	delete_2d_float(2, nr, centroid_cosines);

	fill(countmap.begin(), countmap.end(), 0);
}
/********************************************************/

void XregTrain::get_label_centroid_normalizers(_float &train_time)
{
	_int nc = trn_Y_X->nc;
	#pragma omp parallel num_threads(4)
	{
		unordered_map<_int, _float> centroid;
		_int x;
		_float ctr;
		Timer timer; timer.tic();

		#pragma omp for
		for(_int y = 0; y < nc; y++)
		{
			centroid.clear();

			for(_int i = 0; i < trn_Y_X->size[y]; i++)
			{
				x = trn_Y_X->data[y][i].first;
				ctr = trn_Y_X->data[y][i].second;
				for(_int j = 0; j < trn_X_Xf->size[x]; j++)
					centroid[trn_X_Xf->data[x][j].first] += trn_X_Xf->data[x][j].second * ctr;
			}

			_float normsq = 0.0;
			
			for(auto pair : centroid)
			{
				if(pair.first < trn_X_Xf->nr - 1) // avoid addition of bias term in normalizer
					normsq += SQ(pair.second);
			}
			
			normsq = sqrt(normsq);

			if(normsq == 0)
				normsq = 1.0;

			#pragma omp critical
				normalizers[y] = normsq;
		}

		#pragma omp critical
		{
			train_time += timer.toc();
		}
	}
}

void XregTrain::train_tree(_float &train_time)
{
	writeTreeHeader();

	cout << "getting normalizers " << endl;

	normalizers.resize(num_Y, 0.0);
	get_label_centroid_normalizers(train_time);

	for(int tree_level = 0; tree_level < max_depth; tree_level++)
	{
		int start_index = pow(2, tree_level) - 1;
		int end_index = 2 * (pow(2, tree_level) - 1);

		cout << "Tree level = " << tree_level << " " << start_index << " " << end_index << endl;

		#pragma omp parallel for num_threads(num_thread)
		for(int i = start_index; i <= end_index; i++)
		{
			Timer timer; timer.tic();

			Node* node = tree_nodes[i];
			VecI& n_Y = node->Y;
			SMatF* n_trn_X_Xf;
			SMatF* n_trn_Y_X;
			VecI n_X;
			VecI n_Xf;
			_int thread_id = omp_get_thread_num();
			_bool X_Xf_deep_copy = (tree_level > 5);
			
			trn_Y_X->shrink_mat(n_Y, n_trn_Y_X, n_X, mask[thread_id], false);

			if(X_Xf_deep_copy)
				trn_X_Xf->shrink_mat(n_X, n_trn_X_Xf, n_Xf, mask[thread_id], false);
			else
				trn_X_Xf->in_place_shrink_mat(n_X, n_trn_X_Xf, n_Xf, mask[thread_id]);

			if (node->is_leaf)
			{
				train_leaf_classifiers(node, n_trn_X_Xf, n_trn_Y_X, num_Xf, n_Xf, param, X_Xf_deep_copy, reng[thread_id]);
				delete n_trn_Y_X; n_trn_Y_X = NULL;
				delete n_trn_X_Xf; n_trn_X_Xf = NULL;
			}
			else
			{
				VecI partition;

				balanced_kmeans(n_trn_X_Xf, n_trn_Y_X, n_Y, normalizers, param.clustering_eps, partition, mask[thread_id], reng[thread_id]);
				train_internal_classifiers(node, n_trn_X_Xf, n_trn_Y_X, num_Xf, n_Xf, partition, param, i, mask[thread_id], X_Xf_deep_copy, reng[thread_id]);

				delete n_trn_Y_X; n_trn_Y_X = NULL;
				delete n_trn_X_Xf; n_trn_X_Xf = NULL;
				
				VecI pos_Y, neg_Y;
				for (_int j = 0; j < n_Y.size(); j++)
					if (partition[j])
						pos_Y.push_back(n_Y[j]);
					else
						neg_Y.push_back(n_Y[j]);

				Node* pos_node = new Node(pos_Y, node->depth + 1, max_depth);
				pos_node->id = node->id * 2 + 1;
				node->pos_child = pos_node->id;

				Node* neg_node = new Node(neg_Y, node->depth + 1, max_depth);
				neg_node->id = node->id * 2 + 2;
				node->neg_child = neg_node->id;

				tree_nodes[pos_node->id] = pos_node;
				tree_nodes[neg_node->id] = neg_node;
			}

			#pragma omp critical
			{
				node->writeBin(fout);
				delete node; node = NULL;
				train_time += timer.toc();
			}
		}
	}
}

/********************** PREDICTION CODE **********************/

_float XregPredict::compute_leaf_num( _int num_tst, _int max_depth, _float per_label_predict_slope, _float per_label_predict_intercept )
{
	_float prod = 1.0;
	_float frac = per_label_predict_intercept;
	for( _int i=0; i<max_depth; i++ )
	{
		prod *= frac;
		frac += per_label_predict_slope;
	}
	return (_float)num_tst * prod;
}

_float XregPredict::get_per_label_predict_intercept( _int num_tst, _int max_leaf, _int max_depth, _float per_label_predict_factor, _float per_label_predict_slope )
{
	_float ps = per_label_predict_slope;
	_float pf = per_label_predict_factor;
	_float required_leaf_num = (_float)max_leaf*per_label_predict_factor;

	_float l = 0.0;
	_float h = 1.0;
	_int ctr = 0;

	while( h-l > 1e-4 )
	{
		_float m = (h+l)/2;
		_float leaf_num = compute_leaf_num( num_tst, max_depth, per_label_predict_slope, m );

		cout << ctr << "\t" << m << "\t" << leaf_num << "\t" << required_leaf_num << endl;
		ctr++;

		if( leaf_num>required_leaf_num )
			h = m;
		else if( leaf_num<required_leaf_num )
			l = m;
		else
			break;
	}

	return (h+l)/2;
}

void print_vector_of_pair(VecIF vec, ofstream& fout, _int num_to_print=-1)
{
	if(num_to_print == -1)
		num_to_print = vec.size();

	for(_int i = 0; i < min((size_t)num_to_print, vec.size()); i++)
		fout << vec[i].first << ":" << vec[i].second << " ";
	fout << "\n";
}

_float XregPredict::getNewScore(_float prod, _float prev_score, _Classifier_Kind classifier_kind)
{
	_float newvalue = 0.0;

	if( classifier_kind == L2R_L2LOSS_SVC_DUAL )
		newvalue = - SQ( max( (_float)0.0, 1-prod ) );
	else if( classifier_kind == L2R_LR_DUAL )
		newvalue = - log( 1 + exp( -prod ) );
	else
	{
		prod = abs( prod );
		newvalue = log( prod );
	}

	newvalue += discount * prev_score;
	return newvalue;
}

void XregPredict::getAllScoresForNode(Node* node, _int inst, _int n, _int thread_id, vector<vector<pairIF>>& score_mat, VecIF& mapped_points)
{
	// NOTE : for per label prediction score_mat is of size num_nodes, for point prediction score_mat is of size num_X
	_int 	target_id 	= 2*n + 1;
	SMatF* 	wt_mat 		= node->w;

	for(_int target = 0; target < wt_mat->nc; target++)
	{
		pairIF* target_wt_arr 		= wt_mat->data[target];
		_int 	target_wt_arr_size 	= wt_mat->size[target];

		// densify weight array
		set_d_with_s(target_wt_arr, target_wt_arr_size, mask[thread_id]);

		for(auto point : mapped_points)
		{
			_int 	point_id 	= point.first;
			_float 	prev_score	= point.second;

			_float prod = mult_d_s_vec(mask[thread_id], tst_X_Xf->data[point_id], tst_X_Xf->size[point_id]);
					
			if(param.per_label_predict)
				score_mat[target_id].push_back(make_pair(point_id, getNewScore(prod, prev_score, param.classifier_kind)));
			else
				score_mat[point_id].push_back(make_pair(target_id, getNewScore(prod, prev_score, param.classifier_kind)));
		}

		// retain only top retain_num entries, if predicting tree per label
		if(param.per_label_predict)
		{
			_float frac = param.per_label_predict_intercept + node->depth * param.per_label_predict_slope;
			_int retain_num = round(score_mat[target_id].size() * frac);

			sort(score_mat[target_id].begin(), score_mat[target_id].end(), comp_pair_by_second_desc<_int,_float>);
			score_mat[target_id].resize(retain_num);
		}

		target_id += 1;
		reset_d_with_s(target_wt_arr, target_wt_arr_size, mask[thread_id]);
	}
}

void XregPredict::leafPredictPerPoint(_int point_id, _int thread_id, VecIF& current_scores, VecIF& new_scores, vector<Node*>& nodes)
{
	set_d_with_s(tst_X_Xf->data[point_id], tst_X_Xf->size[point_id], mask[thread_id]);

	for(auto node_score_pair : current_scores)
	{
		_int 	node_id 	= node_score_pair.first;
		_float 	prev_score 	= node_score_pair.second;

		SMatF* wt_mat = nodes[node_id]->w;

		for(_int target = 0; target < wt_mat->nc; target++)
		{
			_float prod = mult_d_s_vec(mask[thread_id], wt_mat->data[target], wt_mat->size[target]);
			new_scores.push_back(make_pair(nodes[node_id]->Y[target], exp(getNewScore(prod, prev_score, param.leaf_classifier_kind))));
		}
	}

	reset_d_with_s(tst_X_Xf->data[point_id], tst_X_Xf->size[point_id], mask[thread_id]);				
}

void XregPredict::leafPredictPerLabel(Node* node, _int thread_id, VecIF& mapped_points)
{
	SMatF* 	wt_mat 		= node->w;

	for(_int target = 0; target < wt_mat->nc; target++)
	{
		pairIF* target_wt_arr 		= wt_mat->data[target];
		_int 	target_wt_arr_size 	= wt_mat->size[target];
		_int 	target_id 			= node->Y[target];

		VecIF cur_scores;

		// densify weight array
		set_d_with_s(target_wt_arr, target_wt_arr_size, mask[thread_id]);

		for(auto point : mapped_points)
		{
			_int 	point_id 	= point.first;
			_float 	prev_score	= point.second;

			_float prod = mult_d_s_vec(mask[thread_id], tst_X_Xf->data[point_id], tst_X_Xf->size[point_id]);
			cur_scores.push_back(make_pair(point_id, exp(getNewScore(prod, prev_score, param.leaf_classifier_kind))));
		}

		sort(cur_scores.begin(), cur_scores.end(), comp_pair_by_second_desc<_int, _float>);

		#pragma omp critical
		{
			fout << target_id << "\t";
			print_vector_of_pair(cur_scores, fout, 100);
		}

		reset_d_with_s(target_wt_arr, target_wt_arr_size, mask[thread_id]);
	}
}

void XregPredict::predict_tree(_float &predict_time)
{	
	vector<Node*>& 	nodes 		= tree->nodes;
	_int 			num_node 	= nodes.size();

	vector<vector<pairIF> > score_mat(num_X, vector<pairIF>({{0, 0.0}}));

	for(_int tree_level = 0; tree_level < max_depth - 1; tree_level++)
	{
		Timer timer; timer.tic();

		int start_index = pow(2, tree_level) - 1;
		int end_index = 2 * (pow(2, tree_level) - 1);

		cout << "Tree level = " << tree_level << " " << start_index << " " << end_index << endl;

		vector<VecIF> points_mapped_to_node((int)pow(2, tree_level), VecIF({}));
		for(_int i = 0; i < num_X; i++)
		{	
			for(_int j = 0; j < score_mat[i].size(); j++)
			{
				points_mapped_to_node[score_mat[i][j].first - start_index].push_back(make_pair(i, score_mat[i][j].second));
			}
			score_mat[i].clear();
		}

		predict_time += timer.toc();

		#pragma omp parallel num_threads(num_thread)
		{
			Timer timer2; timer2.tic();

			_int thread_id = omp_get_thread_num();
			vector<vector<pairIF>> score_mat_private(num_X, vector<pairIF>({}));

			#pragma omp for schedule(dynamic) nowait
			for(_int n = start_index; n <= end_index; n++)
			{
				_int inst = n - start_index;
				getAllScoresForNode(nodes[n], inst, n, thread_id, score_mat_private, points_mapped_to_node[inst]);

				delete nodes[n]; nodes[n] = NULL;
			}

			#pragma omp critical
			{
				for(_int i = 0; i < num_X; i++)
					score_mat[i].insert(score_mat[i].end(), score_mat_private[i].begin(), score_mat_private[i].end());
				predict_time += timer2.toc();
			}
		}

		timer.tic();
		for(_int i = 0; i < num_X; i++)
		{	
			sort(score_mat[i].begin(), score_mat[i].end(), comp_pair_by_second_desc<_int, _float>);

			if(score_mat[i].size() > beam_size)
				score_mat[i].resize(beam_size);
		}
		predict_time += timer.toc();
	}

	cout << "doing leaf level " << endl;

	vector<VecIF> thread_scores(num_thread);

	#pragma omp parallel for num_threads(num_thread)
	for(_int x = 0; x < num_X; x++)
	{
		Timer timer; timer.tic();
		_int thread_id = omp_get_thread_num();

		leafPredictPerPoint(x, thread_id, score_mat[x], thread_scores[thread_id], nodes);

		sort(thread_scores[thread_id].begin(), thread_scores[thread_id].end(), comp_pair_by_second_desc<_int, _float>);
		
		#pragma omp critical
		{
			fout << x << "\t";
			print_vector_of_pair(thread_scores[thread_id], fout, 100);
			predict_time += timer.toc();
		}

		thread_scores[thread_id].clear();
		score_mat[x].clear();
	}
}

class Comparator
{
public:
    bool operator() (const pairIF& a, const pairIF& b)
    {
        return a.second > b.second;
    }
};


void XregPredict::predict_tree_per_label(_float &predict_time)
{
    vector<Node*>& nodes = tree->nodes;
    _int num_nodes = nodes.size();

    vector<vector<pairIF> > score_mat(num_nodes);
    for(_int i = 0; i < num_X; i++)
        score_mat[0].push_back(make_pair(i,  0.0));
    
    for(_int tree_level = 0; tree_level < max_depth - 1; tree_level++)
    {
        int start_index = pow(2, tree_level) - 1;
        int end_index = 2 * (pow(2, tree_level) - 1);

        cout << "Tree level = " << tree_level << " " << start_index << " " << end_index << endl;

        #pragma omp parallel num_threads(num_thread)
        {
        	Timer timer; timer.tic();
            _int thread_id = omp_get_thread_num();

            #pragma omp for schedule(dynamic) nowait
            for(_int n = start_index; n <= end_index; n++)
            {
                _int inst = n - start_index;

                getAllScoresForNode(nodes[n], inst, n, thread_id, score_mat, score_mat[n]);

                score_mat[n].clear();
                delete nodes[n]; nodes[n] = NULL;
            }
			
			cout << "breaking?" << endl;

            #pragma omp critical
            {
            	predict_time += timer.toc();
            }
        }
    }

    cout << "doing leaf level " << endl;

    int start_index = pow(2, max_depth - 1) - 1;
    int end_index = 2 * (pow(2, max_depth - 1) - 1);

	#pragma omp parallel for num_threads(num_thread)
	for(_int n = start_index; n <= end_index; n++)
	{
		Timer timer2; timer2.tic();
		_int thread_id = omp_get_thread_num();

		leafPredictPerLabel(nodes[n], thread_id, score_mat[n]);
		
		score_mat[n].clear();
		#pragma omp critical
        {
			predict_time += timer2.toc();
		}
	}
}