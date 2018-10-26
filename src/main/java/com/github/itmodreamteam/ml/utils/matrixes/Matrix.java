package com.github.itmodreamteam.ml.utils.matrixes;

import java.util.function.IntPredicate;
import java.util.function.UnaryOperator;

public interface Matrix extends ColumnsProvider, RowsProvider {
    int rows();

    int cols();

    double[][] toArray();

    Vector row(int rowNumber);

    Vector col(int colNumber);

    Matrix slice(IntPredicate rows, IntPredicate cols);

    Matrix assign(Operation operation);

    Matrix assign(Matrix that, BiOperation operation);

    double sum();

    double max();

    Matrix mult(double value);

    Vector multColumn(Vector that);

    Matrix enrich(EnrichFunction... functions);

    Matrix inverse();

    Matrix transpose();

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

    Matrix forEachColumn(UnaryOperator<Vector> column);
}
