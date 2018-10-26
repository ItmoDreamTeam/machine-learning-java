package com.github.itmodreamteam.ml.utils.matrixes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.UnaryOperator;

public abstract class AbstractMatrix implements Matrix {
    private final MatrixFactory matrixFactory;
    private final VectorFactory vectorFactory;

    public AbstractMatrix(MatrixFactory matrixFactory, VectorFactory vectorFactory) {
        this.matrixFactory = matrixFactory;
        this.vectorFactory = vectorFactory;
    }

    @Override
    public double max() {
        return Arrays.stream(toArray()).flatMapToDouble(Arrays::stream)
                .max().getAsDouble();
    }

    @Override
    public Matrix enrich(EnrichFunction... functions) {
        int rows = rows();
        List<Vector> enriched = new ArrayList<>();
        for (int rowNumber = 0; rowNumber < rows(); ++rowNumber) {
            Vector row = row(rowNumber);
            enriched.add(row.enrich(functions));
        }
        int cols = enriched.get(0).size();
        double[][] target = new double[rows][cols];
        for (int rowIndex = 0; rowIndex < rows; ++rowIndex) {
            for (int colIndex = 0; colIndex < cols; ++colIndex) {
                target[rowIndex][colIndex] = enriched.get(rowIndex).get(colIndex);
            }
        }
        return matrixFactory.create(target);
    }

    @Override
    public Matrix forEachColumn(UnaryOperator<Vector> operator) {
        int numberOfRows = rows();
        int numberOfCols = cols();
        double[][] result = new double[numberOfRows][numberOfCols];
        for (int col = 0; col < numberOfCols; ++col) {
            double[] column = operator.apply(col(col)).toArray();
            for (int row = 0; row < numberOfRows; ++row) {
                result[row][col] = column[row];
            }
        }
        return matrixFactory.create(result);
    }

    @Override
    public List<Vector> getColumns() {
        return transpose().getRows();
    }

    @Override
    public List<Vector> getRows() {
        double[][] matrix = toArray();
        List<Vector> rows = new ArrayList<>();
        for (int rowNumber = 0, numberOfRows = rows(); rowNumber < numberOfRows; ++rowNumber) {
            Vector row = vectorFactory.create(matrix[rowNumber]);
            rows.add(row);
        }
        return rows;
    }
}
