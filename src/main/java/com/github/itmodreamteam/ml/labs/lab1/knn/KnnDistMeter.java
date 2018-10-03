package com.github.itmodreamteam.ml.labs.lab1.knn;

import com.github.itmodreamteam.ml.utils.matrixes.Vector;

public interface KnnDistMeter {
    static KnnDistMeter euclidian() {
        return new MinkowskiKnnDistMeter(2);
    }

    static KnnDistMeter manhattan() {
        return new MinkowskiKnnDistMeter(1);
    }

    static KnnDistMeter mahalanobis() {
        return new MahalanobisKnnDistMeter();
    }

    static KnnDistMeter cosSimilarity() {
        return new CosSimilarityKnnDistMeter();
    }

    double dist(Vector from, Vector to);
}
