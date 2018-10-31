package com.github.itmodreamteam.ml.regression;

import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;

public interface LinearRegressionFactory {
    default LinearRegression make(Matrix features, Vector expected) {
        return make(features, expected, true, true);
    }

    LinearRegression make(Matrix features, Vector expected, boolean normalize, boolean appendBias);
}
