package com.github.itmodreamteam.ml.utils.matrixes;

import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import com.github.itmodreamteam.ml.utils.matrixes.impl.colt.ColtMatrix;

public class Matrixes {
    public static Matrix dense(double[][] matrix) {
        return new ColtMatrix(new DenseDoubleMatrix2D(matrix));
    }
}
