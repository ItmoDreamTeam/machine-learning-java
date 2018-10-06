package com.github.itmodreamteam.ml.labs.lab1.knn;

import com.github.itmodreamteam.ml.utils.matrixes.Vector;

public class MinkowskiClosestFunction implements KnnClosestFunction {
    private final int power;

    public MinkowskiClosestFunction(int power) {
        this.power = power;
    }

    @Override
    public double dist(Vector from, Vector to) {
        return Math.pow(from.minus(to).abs().power(power).sum(), 1.0 / power);
    }

    @Override
    public String toString() {
        if (power == 1) {
            return "Manhattan";
        }
        if (power == 2) {
            return "Euclidian";
        }
        return "Minkowski(" + power + ")";
    }
}
