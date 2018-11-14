package com.github.itmodreamteam.ml.classification.knn;

import com.github.itmodreamteam.ml.utils.matrixes.Vector;

public class MahalanobisClosestFunction implements KnnClosestFunction {
    @Override
    public double dist(Vector from, Vector to) {
        Vector diff = from.minus(to);
        return Math.sqrt(diff.multAsRow(from.covariance(to).inverse()).multInner(diff));
    }

    @Override
    public String toString() {
        return "Mahalanobis";
    }
}
