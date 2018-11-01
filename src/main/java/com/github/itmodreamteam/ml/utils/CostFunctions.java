package com.github.itmodreamteam.ml.utils;

import com.github.itmodreamteam.ml.utils.matrixes.Vector;

public class CostFunctions {
    public static double computeMse(Vector expected, Vector actual) {
        return expected.minus(actual).power(2).sum() / expected.size();
    }
}
