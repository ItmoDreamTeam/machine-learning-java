package com.github.itmodreamteam.ml.classification.knn;

import com.github.itmodreamteam.ml.utils.collections.IntList;
import com.github.itmodreamteam.ml.utils.collections.Lists;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.classification.ClassifierFactory;
import com.github.itmodreamteam.ml.utils.matrixes.Matrixes;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;
import com.github.itmodreamteam.ml.validation.Samples;

public class Knns implements ClassifierFactory<Vector, Integer> {
    private final int numberOfNeighbors;
    private final KnnClosestFunction meter;
    private final KnnImportanceFunction importanceFunction;
    private final int numberOfClasses;

    private Knns(int numberOfNeighbors, KnnClosestFunction meter, KnnImportanceFunction importanceFunction, int numberOfClasses) {
        this.numberOfNeighbors = numberOfNeighbors;
        this.meter = meter;
        this.importanceFunction = importanceFunction;
        this.numberOfClasses = numberOfClasses;
    }

    public static Knns of(int numberOfNeighbors, KnnClosestFunction meter, KnnImportanceFunction importanceFunction, int numberOfClasses) {
        return new Knns(numberOfNeighbors, meter, importanceFunction, numberOfClasses);
    }

    @Override
    public KnnClassifier build(Samples<Vector, Integer> samples) {
        Matrix trainFeatures = Matrixes.joinRows(samples.getFeatures().toArray(new Vector[0]));
        IntList trainClasses = Lists.of(samples.getAnswers());
        return new KnnClassifier(numberOfNeighbors, meter, importanceFunction, trainFeatures, trainClasses, numberOfClasses);
    }
}
