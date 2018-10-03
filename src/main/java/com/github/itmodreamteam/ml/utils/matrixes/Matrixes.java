package com.github.itmodreamteam.ml.utils.matrixes;

import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import com.github.itmodreamteam.ml.utils.matrixes.impl.colt.ColtMatrix;
import com.github.itmodreamteam.ml.utils.matrixes.impl.colt.ColtMatrixFactory;

public class Matrixes {
    public static Matrix dense(double[][] matrix) {
        return new ColtMatrix(new DenseDoubleMatrix2D(matrix));
    }

    public static MatrixFactory factory() {
        return new ColtMatrixFactory();
    }
}
