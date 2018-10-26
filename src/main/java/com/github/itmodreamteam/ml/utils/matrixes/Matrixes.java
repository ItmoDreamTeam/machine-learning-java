package com.github.itmodreamteam.ml.utils.matrixes;

import com.github.itmodreamteam.ml.utils.matrixes.impl.colt.ColtMatrixFactory;

import java.util.Arrays;
import java.util.List;

import static java.util.stream.Collectors.toList;

public class Matrixes {
    private static final MatrixFactory FACTORY = new ColtMatrixFactory();

    public static Matrix dense(double[][] matrix) {
        return FACTORY.create(matrix);
    }

    public static Matrix joinColumns(ColumnsProvider... providers) {
        List<Vector> columns = Arrays.stream(providers)
                .map(ColumnsProvider::getColumns)
                .flatMap(List::stream)
                .collect(toList());
        int numberOfColumns = columns.size();
        double[][] matrix = new double[numberOfColumns][];
        for (int columnNumber = 0; columnNumber < numberOfColumns; ++columnNumber) {
            matrix[columnNumber] = columns.get(columnNumber).toArray();
        }
        return FACTORY.create(matrix).transpose();
    }

    public static Matrix joinRows(RowsProvider... providers) {
        List<Vector> rows = Arrays.stream(providers)
                .map(RowsProvider::getRows)
                .flatMap(List::stream)
                .collect(toList());
        int numberOfRows = rows.size();
        double[][] matrix = new double[numberOfRows][];
        for (int rowNumber = 0; rowNumber < numberOfRows; ++rowNumber) {
            matrix[rowNumber] = rows.get(rowNumber).toArray();
        }
        return FACTORY.create(matrix);
    }
}
