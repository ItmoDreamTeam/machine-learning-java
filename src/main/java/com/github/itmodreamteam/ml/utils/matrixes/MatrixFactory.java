package com.github.itmodreamteam.ml.utils.matrixes;

public interface MatrixFactory {
    Matrix create(double[][] matrix);

    Matrix joinColumns(Vector... columns);
}
