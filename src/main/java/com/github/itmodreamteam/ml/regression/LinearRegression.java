package com.github.itmodreamteam.ml.regression;

import com.github.itmodreamteam.ml.utils.matrixes.Vector;

public class LinearRegression {
    private final Vector featureWeights;

    public static LinearRegression of(Vector featureWeights) {
        return new LinearRegression(featureWeights);
    }

    private LinearRegression(Vector featureWeights) {
        this.featureWeights = featureWeights;
    }

    public double answer(Vector features) {
        return featureWeights.multInner(features);
    }
}
