package com.github.itmodreamteam.ml.labs.lab1.knn;

import com.github.itmodreamteam.ml.utils.collections.IntList;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.classification.ClassifierFactory;

public class Knns implements ClassifierFactory {
    private final int numberOfNeighbor;
    private final KnnDistMeter core;

    private Knns(int numberOfNeighbor, KnnDistMeter core) {
        this.numberOfNeighbor = numberOfNeighbor;
        this.core = core;
    }

    public static Knns of(int numberOfNeighbor, KnnDistMeter core) {
        return new Knns(numberOfNeighbor, core);
    }

    @Override
    public KnnClassifier build(Matrix trainFeatures, IntList trainClasses) {
        return new KnnClassifier(numberOfNeighbor, core, trainFeatures, trainClasses);
    }
}
