package com.github.itmodreamteam.ml.utils.matrixes;

import java.util.function.IntPredicate;

public interface Matrix {
    int rows();

    int cols();

    double[][] toArray();

    Vector row(int rowNumber);

    Vector col(int colNumber);

    Matrix slice(IntPredicate rows, IntPredicate cols);

    Matrix assign(Operation operation);

    Matrix assign(Matrix that, BiOperation operation);

    double sum();

    Matrix mult(double value);

    Matrix enrich(EnrichFunction... functions);

    Matrix inverse();

    default Matrix slice(IntPredicate rows, boolean cols) {
        return slice(rows, colNumber -> cols);
    }

    default Matrix slice(boolean rows, IntPredicate cols) {
        return slice(rowNumber -> rows, cols);
    }
}
