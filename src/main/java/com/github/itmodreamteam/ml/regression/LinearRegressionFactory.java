package com.github.itmodreamteam.ml.regression;

import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;

public interface LinearRegressionFactory {
    LinearRegression make(Matrix features, Vector expected);
}
