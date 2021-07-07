#include "xreg_metric.h"

using namespace std;

int main(int argc, char const *argv[])
{
	if(argc != 5)
	{
		cerr << "please provide score mat file, true label file, K and weighted(0/1)" << endl;
		exit(-1);
	}

	SMatF* score_mat = new SMatF(string(argv[1]));
	SMatF* tst_X_Y = new SMatF(string(argv[2]));
	int K = atoi(argv[3]);
	bool weighted = (string(argv[4]) != "0");

	cpp_compute_all_metrics(score_mat, tst_X_Y, K, weighted);
	return 0;
}