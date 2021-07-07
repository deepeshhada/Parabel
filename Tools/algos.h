#pragma once

#include <iostream>
#include <random>

#include "utils.h"
#include "mat.h"
#include "timer.h"
#include "linear_classifiers.h"
#include "tail_classifiers.h"

void balanced_kmeans( SMatF* mat, _float acc, VecI& partition, mt19937& reng );