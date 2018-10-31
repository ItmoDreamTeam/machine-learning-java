package com.github.itmodreamteam.ml.regression;

import com.github.itmodreamteam.ml.utils.matrixes.Operation;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;
import com.github.itmodreamteam.ml.utils.matrixes.Vectors;

import java.util.Arrays;
import java.util.List;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class LinearRegression {
    private final Vector featureWeights;
    private final List<Operation> normalizers;
    private final boolean appendBias;

    public static LinearRegression of(Vector featureWeights, boolean appendBias) {
        List<Operation> normalizers = Stream.iterate(Operation.IDENTITY, UnaryOperator.identity()).limit(featureWeights.size()).collect(Collectors.toList());
        return new LinearRegression(featureWeights, normalizers, appendBias);
    }

    public static LinearRegression of(Vector featureWeights, List<Operation> normalizers, boolean appendBias) {
        return new LinearRegression(featureWeights, normalizers, appendBias);
    }

    private LinearRegression(Vector featureWeights, List<Operation> normalizers, boolean appendBias) {
        this.featureWeights = featureWeights;
        this.normalizers = normalizers;
        this.appendBias = appendBias;
    }

    public double answer(Vector features) {
        double[] preparedFeatures = new double[features.size()];
        for (int featureNumber = 0; featureNumber < features.size(); ++featureNumber) {
            double feature = features.get(featureNumber);
            Operation normalizer = normalizers.get(featureNumber);
            preparedFeatures[featureNumber] = normalizer.apply(feature);
        }
        if (appendBias) {
            double[] extendedFeatures = new double[preparedFeatures.length + 1];
            System.arraycopy(preparedFeatures, 0, extendedFeatures, 1, 2);
            extendedFeatures[0] = 1;
            preparedFeatures = extendedFeatures;
        }
        return featureWeights.multInner(Vectors.dense(preparedFeatures));
    }

    @Override
    public String toString() {
        return "LinearRegression{" +
                "weights=" + Arrays.toString(featureWeights.toArray()) +
                '}';
    }
}
