package com.github.itmodreamteam.ml.utils.matrixes.impl.colt;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.doublealgo.Statistic;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import com.github.itmodreamteam.ml.utils.matrixes.*;

import java.util.ArrayList;
import java.util.List;
import java.util.function.IntPredicate;
import java.util.stream.IntStream;

public class ColtMatrix implements Matrix {
    final DoubleMatrix2D colt;

    public ColtMatrix(DoubleMatrix2D colt) {
        this.colt = colt;
    }

    @Override
    public int rows() {
        return colt.rows();
    }

    @Override
    public int cols() {
        return colt.columns();
    }

    @Override
    public double[][] toArray() {
        return colt.toArray();
    }

    @Override
    public Vector row(int rowNumber) {
        return new ColtVector(colt.viewRow(rowNumber));
    }

    @Override
    public Vector col(int colNumber) {
        return new ColtVector(colt.viewColumn(colNumber));
    }

    @Override
    public Matrix slice(IntPredicate rows, IntPredicate cols) {
        int[] rowNumbers = IntStream.range(0, rows()).filter(rows).toArray();
        int[] colNumbers = IntStream.range(0, cols()).filter(cols).toArray();
        return new ColtMatrix(colt.viewSelection(rowNumbers, colNumbers));
    }

    @Override
    public Matrix assign(Operation operation) {
        return new ColtMatrix(colt.copy().assign(operation::apply));
    }

    @Override
    public Matrix assign(Matrix that, BiOperation operation) {
        if (that instanceof ColtMatrix) {
            return new ColtMatrix(colt.copy().assign(((ColtMatrix) that).colt, operation::apply));
        } else{
            return new ColtMatrix(colt.copy().assign(new DenseDoubleMatrix2D(that.toArray()), operation::apply));
        }
    }

    @Override
    public double sum() {
        return colt.zSum();
    }

    @Override
    public Matrix mult(double value) {
        return assign(el -> el * value);
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
        return new ColtMatrix(new DenseDoubleMatrix2D(target));
    }

    @Override
    public Matrix inverse() {
        return assign(Operation.INVERSE);
    }

    @Override
    public String toString() {
        return colt.toString();
    }
}
