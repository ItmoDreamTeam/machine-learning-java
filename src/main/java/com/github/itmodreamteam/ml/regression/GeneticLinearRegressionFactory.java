package com.github.itmodreamteam.ml.regression;

import com.github.itmodreamteam.ml.genetic.GeneticAlgorithm;
import com.github.itmodreamteam.ml.genetic.Individual;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;
import com.github.itmodreamteam.ml.utils.matrixes.Vectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
        GeneticAlgorithm<RegressionIndividual> algorithm = new GeneticAlgorithm<>(
                numberOfIterations,
                initialGenerationSize,
                selectionSize,
                features.cols(),
                (in1, in2) -> Double.compare(in2.cost, in1.cost),
                killParents,
                mutationProbability,
                genes -> new RegressionIndividual(genes, features, expected));
        double[] featureWeights = algorithm.make().getGenes();
        return Vectors.dense(featureWeights);
    }

    private static class RegressionIndividual extends Individual {
        private final double cost;

        RegressionIndividual(double[] genes, Matrix features, Vector expected) {
            super(genes);
            Vector featureWeights = Vectors.dense(genes);
            this.cost = expected.minus(features.multColumn(featureWeights)).power(2).sum() / (2 * features.rows());
        }
    }
}
