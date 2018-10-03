package com.github.itmodreamteam.ml.utils.matrixes;

import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import com.github.itmodreamteam.ml.utils.matrixes.impl.colt.ColtVector;

public class Vectors {
    public static Vector dense(double... elements) {
        return new ColtVector(new DenseDoubleMatrix1D(elements));
    }
}
