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

    default Matrix slice(int startRow, int endRow, boolean cols) {
        return slice(row -> row >= startRow && row < endRow, col -> true);
    }

    default Matrix slice(boolean rows, int startCol, int endCol) {
        return slice(rows, col -> col >= startCol && col < endCol);
    }

    default Matrix slice(int startRow, int endRow, int startCol, int endCol) {
        return slice(row -> row >= startRow && row < endRow, col -> col >= startCol && col < endCol);
    }

    default Matrix slice(IntPredicate rows, boolean cols) {
        return slice(rows, colNumber -> cols);
    }

    default Matrix slice(boolean rows, IntPredicate cols) {
        return slice(rowNumber -> rows, cols);
    }
}
