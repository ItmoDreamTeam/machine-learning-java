package com.github.itmodreamteam.ml.utils.matrixes.impl.colt;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;

class ColtUtils {
    public static DoubleMatrix1D vector(Vector that) {
        if (that instanceof ColtVector) {
            return ((ColtVector) that).colt;
        } else{
            return new DenseDoubleMatrix1D(that.toArray());
        }
    }

    public static DoubleMatrix2D matrix(Matrix that) {
        if (that instanceof ColtMatrix) {
            return ((ColtMatrix) that).colt;
        } else{
            return new DenseDoubleMatrix2D(that.toArray());
        }
    }

    public static ColtMatrix matrix(double[][] matrix) {
        return new ColtMatrix(new DenseDoubleMatrix2D(matrix));
    }

    public static ColtVector vector(double... values) {
        return new ColtVector(new DenseDoubleMatrix1D(values));
    }
}
