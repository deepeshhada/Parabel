#pragma once

#include <iostream>
#include <string>
#include <cmath>

#include "config.h"
#include "utils.h"
#include "mat.h"
#include "timer.h"

SMatF* tail_classifier_train( SMatF* trn_X_Xf, SMatF* trn_X_Y, _float& train_time );
SMatF* tail_classifier_predict( SMatF* tst_X_Xf, SMatF* score_mat, SMatF* model_mat, _float alpha, _float threshold, _float& predict_time, _float& model_size );
void tail_classifier_predict_per_label( SMatF* tst_X_Xf, _int lbl, VecIF &node_score_mat, SMatF* model_mat, _float alpha, _float& predict_time, _float& model_size );
void tail_classifier_predict_per_point( SMatF* tst_X_Xf, _int x, VecIF &node_score_mat, SMatF* model_mat, _float alpha, _float& predict_time, _float& model_size );