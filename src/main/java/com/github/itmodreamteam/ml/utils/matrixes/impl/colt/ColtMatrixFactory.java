package com.github.itmodreamteam.ml.utils.matrixes.impl.colt;

import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.MatrixFactory;

public class ColtMatrixFactory implements MatrixFactory {
    @Override
    public Matrix create(double[][] matrix) {
        return ColtUtils.matrix(matrix);
    }
}
