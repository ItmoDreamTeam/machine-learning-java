package com.github.itmodreamteam.ml.classification.knn;

@FunctionalInterface
public interface KnnImportanceFunction {
    double importance(double distance);
}
