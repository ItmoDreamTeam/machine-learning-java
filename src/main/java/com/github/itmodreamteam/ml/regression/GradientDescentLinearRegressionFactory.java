package com.github.itmodreamteam.ml.regression;

import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;
import com.github.itmodreamteam.ml.utils.matrixes.Vectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class GradientDescentLinearRegressionFactory implements LinearRegressionFactory {
    private static final Logger LOGGER = LoggerFactory.getLogger(GradientDescentLinearRegressionFactory.class);
    private final int numberOfIterations;
    private final double regularization;
    private int iteration;

    public GradientDescentLinearRegressionFactory(int numberOfIterations, double regularization) {
        this.numberOfIterations = numberOfIterations;
        this.regularization = regularization;
    }

    @Override
    public LinearRegression make(Matrix features, Vector expected) {
        int numberOfSamples = features.rows();
        int numberOfFeatures = features.cols();
        Matrix extendedFeatures = features.forEachColumn(Vector::normalize)
                .appendLeft(Vectors.ones(numberOfSamples));
        Vector featureWeights = Vectors.zeros(numberOfFeatures + 1);
        for (iteration = 0; iteration < numberOfIterations; ++iteration) {
            featureWeights = optimize(extendedFeatures, featureWeights, expected);
            int size = expected.size();
            double cost = expected.minus(extendedFeatures.multColumn(featureWeights)).power(2).sum() / (2 * size);
            if (needToLog()) {
                LOGGER.info("cost: {}", cost);
            }
        }
        return LinearRegression.of(featureWeights);
    }

    private Vector optimize(Matrix features, Vector featureWeights, Vector expected) {
        Vector gradient = computeGradient(features, expected, features.multColumn(featureWeights));
        return featureWeights.minus(gradient.mult(regularization));
    }

    private Vector computeGradient(Matrix features, Vector expected, Vector actual) {
        int numberOfFeatures = features.cols();
        double[] gradient = new double[numberOfFeatures];
        for (int featureNumber = 0; featureNumber < numberOfFeatures; ++featureNumber) {
            Vector feature = features.col(featureNumber);
            double partialDerivative = 0.0;
            int datasetSize = features.rows();
            for (int sampleNumber = 0; sampleNumber < datasetSize; ++sampleNumber) {
                partialDerivative += (actual.get(sampleNumber) - expected.get(sampleNumber)) * feature.get(sampleNumber);
            }
            partialDerivative /= datasetSize;
            gradient[featureNumber] = partialDerivative;
        }
        return Vectors.dense(gradient);
    }

    private boolean needToLog() {
        int rate = numberOfIterations / 100;
        return iteration % rate == 0;
    }
}
