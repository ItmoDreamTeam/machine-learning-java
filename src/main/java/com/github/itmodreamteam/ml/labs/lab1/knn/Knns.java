package com.github.itmodreamteam.ml.labs.lab1.knn;

import com.github.itmodreamteam.ml.utils.collections.IntList;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.classification.ClassifierFactory;

public class Knns implements ClassifierFactory {
    private final int numberOfNeighbors;
    private final KnnDistMeter meter;
    private final KnnImportanceFunction importanceFunction;
    private final int numberOfClasses;

    private Knns(int numberOfNeighbors, KnnDistMeter meter, KnnImportanceFunction importanceFunction, int numberOfClasses) {
        this.numberOfNeighbors = numberOfNeighbors;
        this.meter = meter;
        this.importanceFunction = importanceFunction;
        this.numberOfClasses = numberOfClasses;
    }

    public static Knns of(int numberOfNeighbors, KnnDistMeter meter, KnnImportanceFunction importanceFunction, int numberOfClasses) {
        return new Knns(numberOfNeighbors, meter, importanceFunction, numberOfClasses);
    }

    @Override
    public KnnClassifier build(Matrix trainFeatures, IntList trainClasses) {
        return new KnnClassifier(numberOfNeighbors, meter, importanceFunction, trainFeatures, trainClasses, numberOfClasses);
    }
}
