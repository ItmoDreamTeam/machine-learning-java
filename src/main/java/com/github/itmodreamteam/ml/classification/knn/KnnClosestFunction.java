package com.github.itmodreamteam.ml.classification.knn;

import com.github.itmodreamteam.ml.utils.matrixes.Vector;

public interface KnnClosestFunction {
    static KnnClosestFunction euclidian() {
        return new MinkowskiClosestFunction(2);
    }

    static KnnClosestFunction manhattan() {
        return new MinkowskiClosestFunction(1);
    }

    static KnnClosestFunction mahalanobis() {
        return new MahalanobisClosestFunction();
    }

    static KnnClosestFunction cosSimilarity() {
        return new CosSimilarityClosestFunction();
    }

    double dist(Vector from, Vector to);
}
