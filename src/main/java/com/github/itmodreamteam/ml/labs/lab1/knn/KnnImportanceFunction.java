package com.github.itmodreamteam.ml.labs.lab1.knn;

@FunctionalInterface
public interface KnnImportanceFunction {
    double importance(double distance);
}
