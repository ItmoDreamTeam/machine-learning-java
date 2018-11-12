package com.github.itmodreamteam.ml.regression;

import com.github.itmodreamteam.ml.utils.CostFunctions;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;
import com.github.itmodreamteam.ml.utils.matrixes.Vectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class GradientDescentLinearRegressionFactory extends AbstractLinearRegressionFactory {
    private static final Logger LOGGER = LoggerFactory.getLogger(GradientDescentLinearRegressionFactory.class);
    private static final double MIN_RATE = 1e-12;
    private static final double EPS = 1e-4;
    private final int maximalNumberOfIterations;
    private final double dropRate;
    private double rate;
    private int iteration;
    private double bestCost = Double.MAX_VALUE;

    public GradientDescentLinearRegressionFactory(int maximalNumberOfIterations, double rate, double dropRate) {
        this.maximalNumberOfIterations = maximalNumberOfIterations;
        this.rate = rate;
        this.dropRate = dropRate;
    }

    @Override
    protected Vector doMake(Matrix features, Vector expected) {
        int numberOfFeatures = features.cols();
        Vector featureWeights = Vectors.zeros(numberOfFeatures);
        for (iteration = 0; iteration < maximalNumberOfIterations; ++iteration) {
            double cost = 0;
            while (true) {
                if (rateIsTooSmall()) {
                    break;
                }
                Vector candidateFeatureWeights = optimize(features, featureWeights, expected);
                cost = CostFunctions.computeMse(expected, features.multColumn(featureWeights)) / 2;

                if (bestCost < cost) {
                    rate *= dropRate;
                } else {
                    bestCost = cost;
                    featureWeights = candidateFeatureWeights;
                    break;
                }
            }
            if (Math.abs(cost - bestCost) / bestCost > EPS) {
                LOGGER.info("local minimum has been found, iteration: {}, cost: {}", iteration, bestCost);
                break;
            }
            if (needToLog()) {
                LOGGER.info("cost: {}", bestCost);
            }
        }
        return featureWeights;
    }

    private Vector optimize(Matrix features, Vector featureWeights, Vector expected) {
        Vector gradient = computeGradient(features, expected, features.multColumn(featureWeights));
        return featureWeights.minus(gradient.mult(rate));
    }

    private Vector computeGradient(Matrix features, Vector expected, Vector actual) {
        int numberOfFeatures = features.cols();
        double[] gradient = new double[numberOfFeatures];
        for (int featureNumber = 0; featureNumber < numberOfFeatures; ++featureNumber) {
            Vector feature = features.col(featureNumber);
            double partialDerivative = 0.0;
            int numberOfSamples = features.rows();
            for (int sampleNumber = 0; sampleNumber < numberOfSamples; ++sampleNumber) {
                partialDerivative += (actual.get(sampleNumber) - expected.get(sampleNumber)) * feature.get(sampleNumber);
            }
            partialDerivative /= numberOfSamples;
            gradient[featureNumber] = partialDerivative;
        }
        return Vectors.dense(gradient);
    }

    private boolean rateIsTooSmall() {
        return rate < MIN_RATE;
    }

    private boolean needToLog() {
        int rate = maximalNumberOfIterations / 100;
        return iteration % rate == 0;
    }

    @Override
    public String toString() {
        return "GradientDescentLinearRegressionFactory{" +
                "maximalNumberOfIterations=" + maximalNumberOfIterations +
                ", rate=" + rate +
                ", dropRate=" + dropRate +
                '}';
    }
}
