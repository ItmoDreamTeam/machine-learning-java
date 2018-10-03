package com.github.itmodreamteam.ml.utils.matrixes.impl.colt;

import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.MatrixFactory;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;

public class ColtMatrixFactory implements MatrixFactory {
    @Override
    public Matrix create(double[][] matrix) {
        return ColtUtils.matrix(matrix);
    }

    @Override
    public Matrix joinColumns(Vector... columns) {
        double[][] matrix = new double[columns.length][];
        for (int i = 0; i < columns.length; i++) {
            matrix[i] = columns[i].toArray();
        }
        return new ColtMatrix(Algebra.DEFAULT.transpose(new DenseDoubleMatrix2D(matrix)));
    }
}
