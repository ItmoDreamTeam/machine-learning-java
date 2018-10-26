package com.github.itmodreamteam.ml.regression;

import com.github.itmodreamteam.ml.utils.matrixes.Vector;

import java.util.Arrays;

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

    @Override
    public String toString() {
        return "LinearRegression{" +
                "weights=" + Arrays.toString(featureWeights.toArray()) +
                '}';
    }
}
