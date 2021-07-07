#include "algos.h"

using namespace std;

void balanced_kmeans( SMatF* mat, _float acc, VecI& partition, mt19937& reng )
{
	_int nc = mat->nc;
	_int nr = mat->nr;

	_int c[2] = {-1,-1};
	c[0] = get_rand_num( nc, reng );
	c[1] = c[0];
	while( c[1] == c[0] )
		c[1] = get_rand_num( nc, reng );

	_float** centers;
	init_2d_float( 2, nr, centers );
	reset_2d_float( 2, nr, centers );
	for( _int i=0; i<2; i++ )
		set_d_with_s( mat->data[c[i]], mat->size[c[i]], centers[i] );

	_float** cosines;
	init_2d_float( 2, nc, cosines );
	
	pairIF* dcosines = new pairIF[ nc ];

	partition.resize( nc );

	_float old_cos = -10000;
	_float new_cos = -1;

	while( new_cos - old_cos >= acc )
	{

		for( _int i=0; i<2; i++ )
		{
			for( _int j=0; j<nc; j++ )
				cosines[i][j] = mult_d_s_vec( centers[i], mat->data[j], mat->size[j] );
		}

		for( _int i=0; i<nc; i++ )
		{
			dcosines[i].first = i;
			dcosines[i].second = cosines[0][i] - cosines[1][i];
		}
		
		sort( dcosines, dcosines+nc, comp_pair_by_second_desc<_int,_float> );

		old_cos = new_cos;
		new_cos = 0;
		for( _int i=0; i<nc; i++ )
		{
			_int id = dcosines[i].first;
			_int part = (_int)(i < nc/2);
			partition[ id ] = 1 - part;
			new_cos += cosines[ partition[id] ][ id ];
		}
		new_cos /= nc;

		reset_2d_float( 2, nr, centers );

		for( _int i=0; i<nc; i++ )
		{
			_int p = partition[ i ];
			add_s_to_d_vec( mat->data[i], mat->size[i], centers[ p ] );
		}

		for( _int i=0; i<2; i++ )
			normalize_d_vec( centers[i], nr );
	}

	delete_2d_float( 2, nr, centers );
	delete_2d_float( 2, nc, cosines );
	delete [] dcosines;
}