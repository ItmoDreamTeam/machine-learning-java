package com.github.itmodreamteam.ml.regression;

import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Matrixes;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;
import com.github.itmodreamteam.ml.utils.matrixes.Vectors;

public class NormalEquationSolverLinearRegressionFactory implements LinearRegressionFactory {
    @Override
    public LinearRegression make(Matrix features, Vector expected) {
        features = Matrixes.joinColumns(
                Vectors.ones(features.rows()),
                features
        );
        Matrix transposedFeatures = features.transpose();
        return LinearRegression.of(transposedFeatures.mult(features).inverse().mult(transposedFeatures).multColumn(expected));
    }
}
