package com.github.itmodreamteam.ml.regression;

import com.github.itmodreamteam.ml.genetic.GeneticAlgorithm;
import com.github.itmodreamteam.ml.genetic.Individual;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;
import com.github.itmodreamteam.ml.utils.matrixes.Vectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Comparator;

public class GeneticLinearRegressionFactory extends AbstractLinearRegressionFactory {
    private static final Logger LOGGER = LoggerFactory.getLogger(GeneticLinearRegressionFactory.class);
    private final int numberOfIterations;
    private final int initialGenerationSize;
    private final int selectionSize;
    private final boolean killParents;
    private final int mutationProbability;

    public GeneticLinearRegressionFactory(int numberOfIterations, int initialGenerationSize, int selectionSize, boolean killParents, int mutationProbability) {
        this.numberOfIterations = numberOfIterations;
        this.initialGenerationSize = initialGenerationSize;
        this.selectionSize = selectionSize;
        this.killParents = killParents;
        this.mutationProbability = mutationProbability;
    }

    @Override
    public Vector doMake(Matrix features, Vector expected) {
        Comparator<Individual> comparator = new RegressionComparator(features, expected);
        GeneticAlgorithm algorithm = new GeneticAlgorithm(
                numberOfIterations,
                initialGenerationSize,
                selectionSize,
                features.cols(),
                comparator,
                killParents,
                mutationProbability
        );
        double[] featureWeights = algorithm.make().getGenes();
        return Vectors.dense(featureWeights);
    }

    public static class RegressionComparator implements Comparator<Individual> {
        private final Matrix features;
        private final Vector expected;

        public RegressionComparator(Matrix features, Vector expected) {
            this.features = features;
            this.expected = expected;
        }

        @Override
        public int compare(Individual o1, Individual o2) {
            Vector v1 = Vectors.dense(o2.getGenes());
            Vector v2 = Vectors.dense(o1.getGenes());
            return Double.compare(cost(v1), cost(v2));
        }

        private double cost(Vector featureWeights) {
            return expected.minus(features.multColumn(featureWeights)).power(2).sum() / (2 * features.rows());
        }

    }
}
