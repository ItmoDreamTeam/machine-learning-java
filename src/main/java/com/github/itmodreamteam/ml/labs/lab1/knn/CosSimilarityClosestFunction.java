package com.github.itmodreamteam.ml.labs.lab1.knn;

import com.github.itmodreamteam.ml.utils.matrixes.Operation;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;

public class CosSimilarityClosestFunction implements KnnClosestFunction {
    @Override
    public double dist(Vector from, Vector to) {
        double d1 = from.assign(to, (f, t) -> f * t).sum();
        double d2 = Math.sqrt(from.assign(Operation.SQUARE).sum());
        double d3 = Math.sqrt(to.assign(Operation.SQUARE).sum());
        return 1.0 / (d1 / (d2 * d3));
    }
}
