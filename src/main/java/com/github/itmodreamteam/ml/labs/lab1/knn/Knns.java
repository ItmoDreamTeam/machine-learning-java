package com.github.itmodreamteam.ml.labs.lab1.knn;

import com.github.itmodreamteam.ml.utils.collections.IntList;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.classification.ClassifierFactory;

public class Knns implements ClassifierFactory {
    private final int numberOfNeighbor;
    private final KnnDistMeter core;
    private final KnnImportanceFunction importanceFunction;
    private final int numberOfClasses;

    private Knns(int numberOfNeighbor, KnnDistMeter core, KnnImportanceFunction importanceFunction, int numberOfClasses) {
        this.numberOfNeighbor = numberOfNeighbor;
        this.core = core;
        this.importanceFunction = importanceFunction;
        this.numberOfClasses = numberOfClasses;
    }

    public static Knns of(int numberOfNeighbor, KnnDistMeter core, KnnImportanceFunction importanceFunction, int numberOfClasses) {
        return new Knns(numberOfNeighbor, core, importanceFunction, numberOfClasses);
    }

    @Override
    public KnnClassifier build(Matrix trainFeatures, IntList trainClasses) {
        return new KnnClassifier(numberOfNeighbor, core, importanceFunction, trainFeatures, trainClasses, numberOfClasses);
    }
}
