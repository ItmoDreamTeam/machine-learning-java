package com.github.itmodreamteam.ml.utils.matrixes.impl.colt;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import com.github.itmodreamteam.ml.utils.matrixes.*;

import java.util.function.IntPredicate;
import java.util.stream.IntStream;

class ColtMatrix extends AbstractMatrix {
    private static final ColtMatrixFactory MATRIX_FACTORY = new ColtMatrixFactory();
    private static final ColtVectorFactory VECTOR_FACTORY = new ColtVectorFactory();
    final DoubleMatrix2D colt;

    ColtMatrix(DoubleMatrix2D colt) {
        super(MATRIX_FACTORY, VECTOR_FACTORY);
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
        return new ColtMatrix(colt.copy().assign(ColtUtils.matrix(that), operation::apply));
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
    public Vector multColumn(Vector that) {
        return new ColtVector(Algebra.DEFAULT.mult(colt, ColtUtils.vector(that)));
    }

    @Override
    public Matrix transpose() {
        return new ColtMatrix(Algebra.DEFAULT.transpose(colt));
    }

    @Override
    public String toString() {
        return colt.toString();
    }
}
