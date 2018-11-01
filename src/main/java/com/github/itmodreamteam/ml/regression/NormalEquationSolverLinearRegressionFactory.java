package com.github.itmodreamteam.ml.regression;

import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;

public class NormalEquationSolverLinearRegressionFactory extends AbstractLinearRegressionFactory {
    @Override
    protected Vector doMake(Matrix features, Vector expected) {
        Matrix transposedFeatures = features.transpose();
        return transposedFeatures.mult(features).inverse().mult(transposedFeatures).multColumn(expected);
    }

    @Override
    public String toString() {
        return "NormalEquationSolverLinearRegressionFactory{}";
    }
}
