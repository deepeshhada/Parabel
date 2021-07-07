#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <set>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <cassert>

#include "config.h"
#include "utils.h"
#include "timer.h"

using namespace std;

/* ------------------- Sparse and dense matrix and vector resources ---------------------- */

typedef vector<_int> VecI;
typedef vector<_float> VecF;
typedef vector<_double> VecD;
typedef vector<pairII> VecII;
typedef vector<pairIF> VecIF;
typedef vector<_bool> VecB;

/* ------------------- Helper functions Begin -----------------------*/

template <typename T>
T* getDeepCopy(T* arr, _int size)
{
	T* new_arr = new T[size];
	for(_int i = 0; i < size; ++i)
		new_arr[i] = arr[i];

	return new_arr;
}

/* ------------------- Helper functions End -----------------------*/

template <typename T>
class SMat // a column-major sparse matrix of type T
{
public:
	_bool contiguous = false;
	_int nc = 0;
	_int nr = 0;
	vector<_int> size;
	vector<pair<_int,T>*> data;
	pair<_int,T>* cdata = NULL;
	vector<_int> col_indices;
	bool owns_data = true;

	SMat( _bool contiguous = false ) : contiguous(contiguous) { }

	SMat( _int nr, _int nc, _bool contiguous = false ) : contiguous(contiguous), nr(nr), nc(nc)
	{
		size.resize(nc, 0);
		data.resize(nc, NULL);
	}

	SMat( _int nr, _int nc, _ullint nnz, _bool contiguous = false ) : contiguous(contiguous), nr(nr), nc(nc) 
	{
		size.resize(nc, 0);
		data.resize(nc, NULL);

		if( contiguous )
			cdata = new pair<_int,T>[ nnz ];
		else
			cdata = NULL;
	}

	SMat(SMat<T>* mat, bool deep_copy = true, bool mask = false, const VecI& active_cols = VecI())
	{
		// assumption : active_cols is sorted
		// NOTE : if mask is true then only columns present in active_cols will be set in new matrix

		nc = (mask ? active_cols.size() : mat->nc);
		nr = mat->nr;
		owns_data = deep_copy;
		
		size.resize(nc, 0);
		data.resize(nc, NULL);

		for(_int i=0; i<nc; i++)
		{
			_int active_col_id = (mask ? active_cols[i] : i);

			size[i] = mat->size[active_col_id];

			if(deep_copy)
				data[i] = getDeepCopy(mat->data[active_col_id], size[i]);
			else
				data[i] = mat->data[active_col_id];
		}
	}

	friend istream& operator>>( istream& fin, SMat<T>& mat )
	{
		string line;
		getline(fin, line);
		std::istringstream iss(line);

		iss >> mat.nc >> mat.nr;
		mat.size.resize(mat.nc, 0);
		mat.data.resize(mat.nc, NULL);

		_int col_no = 0; char colon;

		while(getline(fin, line))
		{
			std::istringstream iss(line);

			_int label_id;
			_float label_score;

			vector<pairIF> scores;

			while(iss >> label_id >> colon >> label_score)
				scores.push_back(pairIF(label_id, label_score));

			// sort to allow, mats with unsorted columns
			sort(scores.begin(), scores.end());

			mat.size[col_no] = scores.size();
			mat.data[col_no] = getDeepCopy(scores.data(), scores.size());

			col_no++;
			if(col_no > mat.nc)
				break;
		}

		return fin;
	}	

	SMat(string fname)
	{
		contiguous = false;
		check_valid_filename(fname,true);

		ifstream fin;
		fin.open(fname);	

		fin >> (*this);

		fin.close();
	}

	// For reading in Scope/Aether
	SMat( string fname, _int num_row )
	{
		contiguous = false;
		check_valid_filename(fname, true);

		ifstream fin;
		fin.open(fname);		

		_int col_index;
		vector<_int> inds;
		vector<T> vals;
		_int max_row_index = -1;

		_int capacity = 1;
		string line;
		_int i = 0;
		size.resize(capacity);
		data.resize(capacity);
		col_indices.resize(capacity);

		while( getline( fin, line ) )
		{
			line += "\n";
			inds.clear();
			vals.clear();

			_int pos = 0;
			_int next_pos;
			next_pos = line.find_first_of( "\t", pos );
			string indstr = line.substr( pos, next_pos-pos );
			col_index = stoi( indstr );
			pos = next_pos+1;

			while(next_pos=line.find_first_of(": \n",pos))
			{
				if((size_t)next_pos==string::npos)
					break;

				string indstr = line.substr(pos,next_pos-pos);
				if( indstr=="" )
					break;
				_int ind = stoi(indstr);

				pos = next_pos+1;
				next_pos = line.find_first_of(": \n",pos);

				if((size_t)next_pos==string::npos)
					break;
				string valstr = line.substr(pos,next_pos-pos);
				_float val = stof( valstr );

				pos = next_pos+1;

				if( num_row != -1 )
				{
					if( ind >= num_row )
						continue;
				}
				else
				{
					max_row_index = ind>max_row_index ?  ind : max_row_index;
				}

				inds.push_back( ind );				
				vals.push_back( val );
			}

			assert(inds.size()==vals.size());
			//assert(inds.size()==0 || inds[inds.size()-1]<nr);

			if( i == capacity-1 )
			{
				_int new_capacity = 2*capacity;

				size.resize(new_capacity, 0);
				data.resize(new_capacity, NULL);
				col_indices.resize(new_capacity, 0);

				capacity = new_capacity;
			}

			col_indices[i] = col_index;
			size[i] = inds.size();
			data[i] = new pair<_int,T>[inds.size()];

			for(_int j=0; j<size[i]; j++)
			{
				data[i][j].first = inds[j];
				data[i][j].second = (T)vals[j];
			}

			i++;
		}

		if( num_row == -1 )
			nr = max_row_index+1;
		else
			nr = num_row;

		nc = i;
		size.resize(nc, 0);
		data.resize(nc, NULL);
		col_indices.resize(nc, 0);

		fin.close();
	}

	void addCol(pair<_int, T>* new_col, _int new_col_size, bool deep_copy = true)
	{
		// TODO : write assumption
		size.push_back(new_col_size);
		nc += 1;

		data.push_back(NULL);
		if(deep_copy)
		{
			data[nc - 1] = getDeepCopy(new_col, new_col_size);
		}
		else
		{
			data[nc - 1] = new_col;
			owns_data = false;
		}
	}

	void reindex_rows(_int _nr, VecI& rows )
	{
		nr = _nr;
		for( _int i=0; i < nc; i++ )
		{
			for( _int j=0; j < size[i]; j++ )
				data[i][j].first = rows[ data[i][j].first ];
		}
	}

	_ullint get_nnz()
	{
		_ullint nnz = 0;
		for( _int i=0; i<nc; i++ )
			nnz += size[i];
		return nnz;
	}

	_float get_ram()
	{
		// TODO : verify
		_float ram = sizeof( SMat<T> );
		ram += sizeof( _int ) * nc;

		for( _int i=0; i<nc; i++ )
			ram += sizeof( pair<_int,T> ) * size[i];

		return ram;
	}

	SMat<T>* transpose()
	{
		SMat<T>* tmat = new SMat<T>(nc, nr);

		for(_int i=0; i<nc; i++)
		{
			for(_int j=0; j<size[i]; j++)
			{
				tmat->size[data[i][j].first]++;
			}
		}

		for(_int i=0; i<tmat->nc; i++)
		{
			tmat->data[i] = new pair<_int,T>[tmat->size[i]];
		}

		vector<_int> count(tmat->nc, 0);
		for(_int i=0; i<nc; i++)
		{
			for(_int j=0; j<size[i]; j++)
			{
				_int ind = data[i][j].first;
				T val = data[i][j].second;

				tmat->data[ind][count[ind]].first = i;
				tmat->data[ind][count[ind]].second = val;
				count[ind]++;
			}
		}

		return tmat;
	}

	void threshold( _float th )
	{
		for( _int i=0; i<nc; i++ )
		{
			_int count = 0;
			for( _int j=0; j < size[i]; j++ )
				count += (fabs(data[i][j].second) > th);

			pair<_int,T>* newvec = new pair<_int,T>[count];
			count = 0;

			for( _int j=0; j<size[i]; j++ )
			{
				_int id = data[i][j].first;
				T val = data[i][j].second;
				if( fabs(val) > th )
					newvec[ count++ ] = make_pair( id, val );
			}

			size[i] = count;
			delete [] data[i];
			data[i] = newvec;
		}
	}

	void unit_normalize_columns()
	{
		for(_int i=0; i<nc; i++)
		{
			T normsq = 0;
			for(_int j=0; j<size[i]; j++)
				normsq += SQ(data[i][j].second);
			normsq = sqrt(normsq);

			if(normsq==0)
				normsq = 1;

			for(_int j=0; j<size[i]; j++)
				data[i][j].second /= normsq;
		}
	}

	vector<T> column_norms()
	{
		vector<T> norms(nc,0);

		for(_int i=0; i<nc; i++)
		{
			T normsq = 0;
			for(_int j=0; j<size[i]; j++)
				normsq += SQ(data[i][j].second);
			norms[i] = sqrt(normsq);
		}

		return norms;
	}

	~SMat()
	{
		if( contiguous )
		{
			if(owns_data)
				if(cdata)
					delete [] cdata;
		}
		else
		{
			if(owns_data)
				for( _int i=0; i<nc; i++ )
					if(data[i])
						delete [] data[i];
		}
	}

	friend ostream& operator<<( ostream& fout, const SMat<T>& mat )
	{
		_int nc = mat.nc;
		_int nr = mat.nr;

		fout << nc << " " << nr << endl;

		for(_int i=0; i<nc; i++)
		{
			for(_int j=0; j<mat.size[i]; j++)
			{
				if(j==0)
					fout << mat.data[i][j].first << ":" << mat.data[i][j].second;
				else
					fout << " " << mat.data[i][j].first << ":" << mat.data[i][j].second;
			}
			fout<<endl;
		}

		return fout;
	}

	void write( string fname, _int precision=3 )
	{
		check_valid_filename(fname,false);

		ofstream fout;
		fout.open(fname);
		fout << fixed << setprecision( precision );
		fout << (*this);

		fout.close();
	}

	void write_scope( string fname, _int precision=3 )
	{
		check_valid_filename(fname,false);

		ofstream fout;
		fout.open(fname);
		fout << fixed << setprecision( precision );

		for( _int i=0; i<nc; i++ )
		{
			fout << col_indices[i] << "\t";
			for( _int j=0; j<size[i]; j++ )
				if( j==0 )
					fout << data[i][j].first << ":" << data[i][j].second;
				else
					fout << " " << data[i][j].first << ":" << data[i][j].second;
			fout << "\n";
		}

		fout.close();
	}

	void add(SMat<T>* smat)
	{
		if(nc != smat->nc || nr != smat->nr)
		{
			cerr << "SMat::add : Matrix dimensions do not match" << endl;
			cerr << "Matrix 1: " << nc << " x " << nr <<endl;
			cerr << "Matrix 2: " << smat->nc << " x " << smat->nr << endl;
			exit(1);
		}

		vector<bool> ind_mask(nr, 0);
		vector<T> sum(nr, 0);

		for(_int i=0; i < nc; i++)
		{
			vector<_int> inds;
			for(_int j=0; j < size[i]; j++)
			{
				_int ind = data[i][j].first;
				T val = data[i][j].second;

				sum[ind] += val;
				if(!ind_mask[ind])
				{
					ind_mask[ind] = true;
					inds.push_back(ind);
				}
			}

			for(_int j=0; j < smat->size[i]; j++)
			{
				_int ind = smat->data[i][j].first;
				T val = smat->data[i][j].second;

				sum[ind] += val;
				if(!ind_mask[ind])
				{
					ind_mask[ind] = true;
					inds.push_back(ind);
				}
			}

			sort(inds.begin(), inds.end());
			Realloc(size[i], inds.size(), data[i]);

			for(_int j=0; j<inds.size(); j++)
			{
				_int ind = inds[j];
				data[i][j] = make_pair(ind,sum[ind]);
				ind_mask[ind] = false;
				sum[ind] = 0;
			}
			size[i] = inds.size();
		}
	}

	void diff(SMat<T>* smat)
	{
		if(nc != smat->nc || nr != smat->nr)
		{
			cerr << "SMat::add : Matrix dimensions do not match" << endl;
			cerr << "Matrix 1: " << nc << " x " << nr <<endl;
			cerr << "Matrix 2: " << smat->nc << " x " << smat->nr << endl;
			exit(1);
		}

		vector<bool> ind_mask(nr, 0);
		vector<T> sum(nr, 0);

		for(_int i=0; i < nc; i++)
		{
			vector<_int> inds;
			for(_int j=0; j < size[i]; j++)
			{
				_int ind = data[i][j].first;
				T val = data[i][j].second;

				sum[ind] += val;
				if(!ind_mask[ind])
				{
					ind_mask[ind] = true;
					inds.push_back(ind);
				}
			}

			for(_int j=0; j < smat->size[i]; j++)
			{
				_int ind = smat->data[i][j].first;
				T val = smat->data[i][j].second;

				sum[ind] -= val;
				if(!ind_mask[ind])
				{
					ind_mask[ind] = true;
					inds.push_back(ind);
				}
			}

			sort(inds.begin(), inds.end());
			Realloc(size[i], inds.size(), data[i]);

			for(_int j=0; j<inds.size(); j++)
			{
				_int ind = inds[j];
				data[i][j] = make_pair(ind,sum[ind]);
				ind_mask[ind] = false;
				sum[ind] = 0;
			}
			size[i] = inds.size();
		}
	}

	void prod_helper( _int siz, pair<_int,T>* dat, vector<_int>& indices, vector<T>& sum )
	{		
		for( _int j=0; j<siz; j++ )
		{
			_int ind = dat[j].first;
			T prodval = dat[j].second;

			for(_int k=0; k<size[ind]; k++)
			{
				_int id = data[ind][k].first;
				T val = data[ind][k].second;

				if(sum[id]==0)
					indices.push_back(id);

				sum[id] += val*prodval;
			}
		}

		sort(indices.begin(), indices.end());
	}

	// Returns sparse product matrix by retaining only top k highest scoring rows of (*this) for every column in mat2 if k > -1 else returns just the product
	SMat<T>* prod(SMat<T>* mat2, _int k = -1)
	{
		bool retain = (k > -1);

		_int dim1 = nr;
		_int dim2 = mat2->nc;

		assert(nc==mat2->nr);

		SMat<T>* prodmat = new SMat<T>(dim1, dim2);
		vector<T> sum(dim1,0);

		for(_int i=0; i<dim2; i++)
		{
			vector<_int> indices;
			prod_helper( mat2->size[i], mat2->data[i], indices, sum );

			_int siz = indices.size();
			prodmat->size[i] = siz;
			prodmat->data[i] = new pair<_int,T>[siz];

			for(_int j=0; j<indices.size(); j++)
			{
				_int id = indices[j];
				T val = sum[id];
				prodmat->data[i][j] = make_pair(id,val);
				sum[id] = 0;
			}

			if(retain)
			{
				sort( prodmat->data[i], prodmat->data[i]+prodmat->size[i], comp_pair_by_second_desc<_int,_float> );
				_int retk = min( k, prodmat->size[i] );
				Realloc( prodmat->size[i], retk, prodmat->data[i] );
				sort( prodmat->data[i], prodmat->data[i]+retk, comp_pair_by_first<_int,_float> );
				prodmat->size[i] = retk;
			}
		}

		return prodmat;
	}

	// Returns sparse product matrix by retaining only top k highest scoring rows of (*this) for every column in mat2
	SMat<T>* top_prod( SMat<T>* mat2, _int k )
	{
		return prod(mat2, k);
	}

	// Returns sparse product matrix by retaining only those entries corresponding to non zeros in the pattern matrix (pat_mat)
	/*
	SMat<T>* sparse_prod( SMat<T>* mat2, SMat<T>* pat_mat )
	{
		_int dim1 = nr;
		_int dim2 = mat2->nc;

		assert(nc==mat2->nr);

		SMat<T>* prodmat = new SMat<T>(dim1,dim2);
		vector<T> sum(dim1,0);

		for(_int i=0; i<dim2; i++)
		{
			vector<_int> indices;
			prod_helper( mat2->size[i], mat2->data[i], indices, sum );

			_int siz = pat_mat->size[i];
			prodmat->size[i] = siz;
			prodmat->data[i] = new pair<_int,T>[siz];

			for( _int j=0; j<siz; j++ )
			{
				_int id = pat_mat->data[i][j].first;
				T val = sum[id];
				prodmat->data[i][j] = make_pair( id, val );
			}

			for(_int j=0; j<indices.size(); j++)
			{
				sum[indices[j]] = 0;
			}
		}

		return prodmat;
	}
	*/

	SMat<T>* sparse_prod( SMat<T>* mat2, SMat<T>* pat_mat )
	{
		_int dim1 = pat_mat->nr;
		_int dim2 = pat_mat->nc;
		_int dim = nr;

		assert( nr == mat2->nr );
		assert( nc == dim1 );
		assert( mat2->nc == dim2 );

		SMat<T>* prod_mat = new SMat<T>( pat_mat );
		vector<T> mask(nr,0);

		for( _int i=0; i<dim2; i++ )
		{
			for( _int j=0; j<mat2->size[i]; j++ )
				mask[ mat2->data[i][j].first ] = mat2->data[i][j].second;

			for( _int j=0; j<pat_mat->size[i]; j++ )
			{
				_int id = pat_mat->data[i][j].first;
				_float prod = 0;
				for( _int k=0; k<size[id]; k++ )
					prod += mask[ data[id][k].first ] * data[id][k].second;
				prod_mat->data[i][j].second = prod;
			}

			for( _int j=0; j<mat2->size[i]; j++ )
				mask[ mat2->data[i][j].first ] = 0;
		}

		return prod_mat;
	}

	SMat<T>* get_rank_mat( string order )
	{
		// order=="desc" or order=="asc" is the sorting order to use over nonzero elements. Zeros are ignored. Replaces the value of each nonzero element in *this matrix with its rank in its column
		SMat<T>* rmat = new SMat<T>( this );

		if( order == "desc" )
			for( _int i=0; i<rmat->nc; i++ )
				stable_sort( rmat->data[i], rmat->data[i]+rmat->size[i], comp_pair_by_second_desc<_int,T> );
		else  if( order == "asc" )
			for( _int i=0; i<rmat->nc; i++ )
				stable_sort( rmat->data[i], rmat->data[i]+rmat->size[i], comp_pair_by_second<_int,T> );
			
		for( _int i=0; i<rmat->nc; i++ )
			for( _int j=0; j<rmat->size[i]; j++ )
				rmat->data[i][j].second = (j+1);

		for( _int i=0; i<rmat->nc; i++ )
				sort( rmat->data[i], rmat->data[i]+rmat->size[i], comp_pair_by_first<_int,T> );

		return rmat;
	}

	void eliminate_zeros()
	{
		assert( !contiguous );

		for( _int i=0; i<nc; i++ )
		{
			_int siz = size[i];
			_int newsiz = 0;
			for( _int j=0; j<siz; j++ )
			{
				if( data[i][j].second != 0 )
				{
					data[i][newsiz] = data[i][j];
					newsiz++;
				}
			}
			size[i] = newsiz;
		}

		// TODO : memory not reallocated
	}

	void append_bias_feat( T bias_feat )
	{
		if( contiguous )
		{
			pair<_int,T>* new_cdata = new pair<_int,T>[ get_nnz()+nc ];
			_int ctr = 0;

			for( _int i=0; i<nc; i++ )
			{
				for( _int j=0; j<size[i]; j++ )
					new_cdata[ctr++] = data[i][j];

				new_cdata[ctr++] = make_pair( nr, bias_feat );
				size[i]++;
			}

			ctr = 0;
			for( _int i=0; i<nc; i++ )
			{
				data[i] = new_cdata+ctr;
				ctr += size[i];
			}
			delete [] cdata;
			cdata = new_cdata;
		}
		else
		{
			for( _int i=0; i<nc; i++ )
			{
				_int siz = size[i];
				Realloc( siz, siz+1, data[i] );
				data[i][siz] = make_pair( nr, bias_feat );
				size[i]++;	
			}
		}
		nr++;
	}

	void active_dims( VecI& cols, VecI& dims, VecI& counts, VecI& countmap )
	{
		dims.clear();
		counts.clear();

		for( _int i=0; i<cols.size(); i++ )
		{
			_int inst = cols[i];
			for( _int j=0; j<size[inst]; j++ )
			{
				_int dim = data[inst][j].first;
				if( countmap[ dim ]==0 )
					dims.push_back(dim);
				countmap[ dim ]++;
			}
		}

		sort(dims.begin(),dims.end());

		for( _int i=0; i<dims.size(); i++ )
		{
			counts.push_back( countmap[ dims[i] ] );
			countmap[ dims[i] ] = 0;
		}
	}

	void in_place_shrink_mat(VecI& cols, SMat<T>*& s_mat, VecI& rows, VecI& countmap)
	{
		s_mat = new SMat<T>(this, false, true, cols);

		VecI counts;
        active_dims( cols, rows, counts, countmap );
	}
    
    void shrink_mat( VecI& cols, SMat<T>*& s_mat, VecI& rows, VecI& countmap, _bool transpose )
    {
        _int s_nc = cols.size();
        VecI counts;
        active_dims( cols, rows, counts, countmap );

        _ullint nnz = 0;
        for( _int i=0; i<counts.size(); i++ )
            nnz += counts[i];

        _int* maps = new _int[ nr ];
        for( _int i=0; i<rows.size(); i++ )
            maps[ rows[i] ] = i;

        _int s_nr = rows.size();
        
        if( transpose )
        {
            s_mat = new SMat<T>( s_nc, s_nr, nnz, true );
        
            _int sumsize = 0;
            for( _int i=0; i<s_nr; i++ )
            {
                s_mat->size[i] = counts[i];
                s_mat->data[i] = s_mat->cdata + sumsize;
                sumsize += counts[i];
            }
            
            for( _int i=0; i<s_nr; i++ )
                counts[i] = 0;
        }
        else
        {
            s_mat = new SMat<T>( s_nr, s_nc, nnz, true );

            _int sumsize = 0;
            for( _int i=0; i<s_nc; i++)
            {
                _int col = cols[i];
                s_mat->size[i] = size[ col ];
                s_mat->data[i] = s_mat->cdata + sumsize;
                sumsize += size[ col ];
            }
        }
            
        for( _int i=0; i<s_nc; i++ )
        {	
            _int col = cols[ i ];
            for( _int j=0; j<size[ col ]; j++ )
            {
                _int row = maps[ data[ col ][ j ].first ];
                _float val = data[ col ][ j ].second;
                
                if( transpose )
                {
                    s_mat->data[row][counts[row]] = make_pair( i, val );
                    counts[row]++;
                }
                else
                    s_mat->data[i][j] = make_pair( row, val );
            }
        }

        delete [] maps;
    }

    void split_mat( _bool* split, SMat<T>*& mat1, SMat<T>*& mat2 )
    {
    	// split vector determines which columns are distributed to mat1 or mat2. If split[i]==false, ith column is given to mat1, else to mat2
    	_int nc1 = 0, nc2 = 0;
    	for( _int i=0; i<nc; i++ )
    	{
    		if( !split[i] )
    			nc1++;
    		else
    			nc2++;
    	}

    	mat1 = new SMat<T>( nr, nc1 );
    	mat2 = new SMat<T>( nr, nc2 );

    	_int i1=0, i2=0;
    	for( _int i=0; i<nc; i++ )
    	{
    		if( !split[i] )
    		{
    			mat1->size[ i1 ] = size[ i ];
    			mat1->data[ i1 ] = new pair<_int,T>[ size[ i ] ];
    			copy( data[ i ], data[ i ] + size[ i ], mat1->data[ i1 ] );
    			i1++;
    		}
    		else
    		{
    			mat2->size[ i2 ] = size[ i ];
    			mat2->data[ i2 ] = new pair<_int,T>[ size[ i ] ];
    			copy( data[ i ], data[ i ] + size[ i ], mat2->data[ i2 ] );
    			i2++;
    		}
    	}
    }

    vector<T> max( VecI& inds, _int axis ) // Only inds columns/rows are used for calculating max
    {
    	assert( axis==0 || axis==1 ); // axis==0 => max along each column, axis==1 => max along each row

    	if( axis==0 )
    	{
    		cout << "Not yet implemented" << endl;
    		exit(1);
    	}
    	else if( axis==1 )
    	{
    		vector<T> maxval( nr, NEG_INF );
    		for( _int i=0; i<inds.size(); i++ )
    		{
    			_int ind = inds[i];
    			for( _int j=0; j<size[ind]; j++ )
    			{
    				_int colind = data[ind][j].first;
    				T colval = data[ind][j].second;
    				maxval[ colind ] = maxval[colind] > colval ? maxval[colind] : colval;
    			}
    		}
    		for( _int i=0; i<nr; i++ )
    		{
    			if( maxval[i]==NEG_INF )
    				maxval[i] = 0;
    		}
    		return maxval;
    	}
    }

    SMat<T>* chunk_mat( _int start, _int num )
    {
    	_int end = start+num-1;
    	assert( start>=0 && start<nc );
    	assert( end>=0 && end<nc );
    	assert( end>=start );

    	_int chunk_nc = num;
    	_int chunk_nr = nr;
    	_ullint chunk_nnz = 0;

    	for( _int i=start; i<=end; i++ )
    		chunk_nnz += size[i];

    	SMat<T>* chunk = new SMat<T>( chunk_nr, chunk_nc, chunk_nnz, true );
    	_int ctr = 0;

    	for( _int i=0; i<num; i++ )
    	{    		
    		chunk->size[i] = size[i+start];
    		chunk->data[i] = chunk->cdata + ctr;

    		for( _int j=0; j<size[i+start]; j++ )
    			chunk->data[i][j] = data[i+start][j];

    		ctr += size[i+start];
    	}
    	return chunk;
    }

    void append_mat( SMat<T>* chunk )
    {
    	assert( nr == chunk->nr );
    	_int chunk_nc = chunk->nc;
    	_int new_nc = nc + chunk_nc;
    	size.resize(new_nc, 0);
    	data.resize(new_nc, NULL);

    	for( _int i=0; i<chunk_nc; i++ )
    	{
    		size[nc+i] = chunk->size[i];
    		data[nc+i] = new pair<_int,T>[ chunk->size[i] ];
    		for( _int j=0; j<chunk->size[i]; j++ )
    		{
    			data[nc+i][j] = chunk->data[i][j];
    		}
    	}

    	nc = new_nc;
    }

    void readBin(std::ifstream& fin)
	{
		fin.read((char *)(&(nc)), sizeof(_int));
		fin.read((char *)(&(nr)), sizeof(_int));

		size.resize(nc, 0);
		for (_int column = 0; column < (nc); ++column) {
			fin.read((char *)(&(size[column])), sizeof(_int));
		}

		data.resize(nc, NULL);
		for (_int column = 0; column < (nc); ++column) {
			data[column] = new std::pair<_int, T>[size[column]]();
			for (_int row = 0; row < (size[column]); ++row) {
				fin.read((char *)(&(data[column][row].first)), sizeof(_int));
				fin.read((char *)(&(data[column][row].second)), sizeof(T));
			}
		}

	}

	void read_legacy_mat(ifstream& fin)
	{
		// TODO : remove
		vector<_int> inds;
		vector<T> vals;

		string line;
		getline( fin, line );
		line += "\n";
		_int pos = 0;
		_int next_pos=line.find_first_of(" \n",pos);
		string s = line.substr(pos,next_pos-pos);
		nc = stoi( s );
		pos = next_pos+1;
		next_pos=line.find_first_of(" \n",pos);
		s = line.substr(pos,next_pos-pos);
		nr = stoi( s );

		size.resize(nc, 0);
		data.resize(nc, NULL);

		for(_int i=0; i<nc; i++)
		{
			inds.clear();
			vals.clear();
			string line;
			getline(fin,line);
			line += "\n";
			_int pos = 0;
			_int next_pos;

			while(next_pos=line.find_first_of(": \n",pos))
			{
				if((size_t)next_pos==string::npos)
					break;
				inds.push_back(stoi(line.substr(pos,next_pos-pos)));
				pos = next_pos+1;

				next_pos = line.find_first_of(": \n",pos);
				if((size_t)next_pos==string::npos)
					break;

				vals.push_back(stof(line.substr(pos,next_pos-pos)));
				pos = next_pos+1;

			}

			assert(inds.size()==vals.size());
			assert(inds.size()==0 || inds[inds.size()-1]<nr);

			size[i] = inds.size();
			data[i] = new pair<_int,T>[inds.size()];

			for(_int j=0; j<size[i]; j++)
			{
				data[i][j].first = inds[j];
				data[i][j].second = (T)vals[j];
			}
		}	
	}

	void writeBin(std::ofstream& fout)
	{
		fout.write((char *)(&(nc)), sizeof(_int));
		fout.write((char *)(&(nr)), sizeof(_int));

		for (_int column = 0; column < (nc); ++column) {
			fout.write((char *)(&(size[column])), sizeof(_int));
		}
		for (_int column = 0; column < (nc); ++column) {
			for (_int row = 0; row < (size[column]); ++row) {
				fout.write((char *)(&(data[column][row].first)), sizeof(_int));
				fout.write((char *)(&(data[column][row].second)), sizeof(T));
			}
		}
	}
};
                  
template <typename T>
class DMat // a column-major dense matrix of type T
{
public:
	_int nc;
	_int nr;
	T** data;

	DMat()
	{
		nc = 0;
		nr = 0;
		data = NULL;
	}

	DMat(_int nc, _int nr)
	{
		this->nc = nc;
		this->nr = nr;
		data = new T*[nc];
		for(_int i=0; i<nc; i++)
			data[i] = new T[nr]();
	}

	DMat(SMat<T>* mat)
	{
		nc = mat->nc;
		nr = mat->nr;
		data = new T*[nc];
		for(_int i=0; i<nc; i++)
			data[i] = new T[nr]();

		for(_int i=0; i<mat->nc; i++)
		{
			pair<_int,T>* vec = mat->data[i];
			for(_int j=0; j<mat->size[i]; j++)
			{
				data[i][vec[j].first] = vec[j].second;
			}
		}
	}

	~DMat()
	{
		for(_int i=0; i<nc; i++)
			delete [] data[i];
		delete [] data;
	}
};

typedef SMat<_float> SMatF;
typedef DMat<_float> DMatF;

void reindex_VecIF( VecIF& vec, VecI& index );

template <typename T>
inline T* read_vec( string fname )
{
	check_valid_filename( fname, true );
	ifstream fin;
	fin.open( fname );
	vector< T > vinp;
	T inp;
	while( fin >> inp )
	{
		vinp.push_back( inp );
	}
	fin.close();

	T* vptr = new T[ vinp.size() ];
	for( _int i=0; i<vinp.size(); i++ )
		vptr[i] = vinp[i];

	return vptr;
}

inline pairII get_pos_neg_count( VecI& pos_or_neg )
{
	pairII counts = make_pair(0,0);
	for( _int i=0; i<pos_or_neg.size(); i++ )
	{
		if(pos_or_neg[i]==+1)
			counts.first++;
		else
			counts.second++;
	}
	return counts;
}

inline void reset_d_with_s( pairIF* svec, _int siz, _float* dvec )
{
	for( _int i=0; i<siz; i++ )
		dvec[ svec[i].first ] = 0;
}

inline void set_d_with_s( pairIF* svec, _int siz, _float* dvec )
{
	for( _int i=0; i<siz; i++ )
		dvec[ svec[i].first ] = svec[i].second;
}

inline void init_2d_float( _int dim1, _int dim2, _float**& mat )
{
	mat = new _float*[ dim1 ];
	for( _int i=0; i<dim1; i++ )
		mat[i] = new _float[ dim2 ]; 
}

inline void delete_2d_float( _int dim1, _int dim2, _float**& mat )
{
	for( _int i=0; i<dim1; i++ )
		delete [] mat[i];
	delete [] mat;
	mat = NULL;
}

inline void reset_2d_float( _int dim1, _int dim2, _float**& mat )
{
	for( _int i=0; i<dim1; i++ )
		for( _int j=0; j<dim2; j++ )
			mat[i][j] = 0;
}

inline _float mult_d_s_vec( _float* dvec, pairIF* svec, _int siz )
{
	_float prod = 0;
	for( _int i=0; i<siz; i++ )
	{
		_int id = svec[i].first;
		_float val = svec[i].second;
		prod += dvec[ id ] * val;
	}
	return prod;
}

inline void add_s_to_d_vec( pairIF* svec, _int siz, _float* dvec )
{
	for( _int i=0; i<siz; i++ )
	{
		_int id = svec[i].first;
		_float val = svec[i].second;
		dvec[ id ] += val;
	}
}

inline vector<pairIF> add_s_to_s( pairIF* svec1, _int siz1, pairIF* svec2, _int siz2)
{
	vector<pairIF> svec_result;
	_int ctr1 = 0, ctr2 = 0;
	while(ctr1 < siz1 or ctr2 < siz2)
	{
		while(ctr1 < siz1 and (ctr2 == siz2 or svec1[ctr1].first < svec2[ctr2].first))
		{
			svec_result.push_back(svec1[ctr1]);
			ctr1++;
		}
		while(ctr2 < siz2 and (ctr1 == siz1 or svec2[ctr2].first < svec1[ctr1].first))
		{
			svec_result.push_back(svec2[ctr2]);
			ctr2++;
		}
		while(ctr1 < siz1 and ctr2 < siz2 and svec1[ctr1].first == svec2[ctr2].first)
		{
			svec_result.push_back(pairIF(svec1[ctr1].first, svec1[ctr1].second + svec2[ctr2].second));
			ctr1++; ctr2++;
		}
	}
	return svec_result;
}

inline _float get_norm_d_vec( _float* dvec, _int siz )
{
	_float norm = 0;
	for( _int i=0; i<siz; i++ )
		norm += SQ( dvec[i] );
	norm = sqrt( norm );
	return norm;
}

inline void div_d_vec_by_scalar( _float* dvec, _int siz, _float s )
{
	for( _int i=0; i<siz; i++)
		dvec[i] /= s;
}

inline void normalize_d_vec( _float* dvec, _int siz )
{
	_float norm = get_norm_d_vec( dvec, siz );
	if( norm>0 )
		div_d_vec_by_scalar( dvec, siz, norm );
}


/* Replicating these SMat<T> template functions to enable compatibility with cython code */
SMatF* p_copy( SMatF* inmat );
void p_add( SMatF* mat1, SMatF* mat2 );
void p_shrink_mat( SMatF* refmat, vector<int>& cols, SMatF*& s_mat, vector<int>& rows, vector<int>& countmap, bool transpose );
SMatF* p_get_rank_mat( SMatF* refmat, string order );
SMatF* p_transpose( SMatF* refmat );
SMatF* p_prod( SMatF* refmat, SMatF* mat2 );
SMatF* p_sparse_prod( SMatF* refmat, SMatF* mat2, SMatF* pat_mat );