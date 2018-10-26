package com.github.itmodreamteam.ml.utils.matrixes.impl.colt;

import cern.colt.list.DoubleArrayList;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.doublealgo.Statistic;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import com.github.itmodreamteam.ml.utils.matrixes.*;

public class ColtVector implements Vector {
    final DoubleMatrix1D colt;

    public ColtVector(DoubleMatrix1D colt) {
        this.colt = colt;
    }

    @Override
    public int size() {
        return colt.size();
    }

    @Override
    public double get(int index) {
        return colt.get(index);
    }

    @Override
    public double[] toArray() {
        return colt.toArray();
    }

    @Override
    public Vector assign(Vector that, BiOperation operation) {
        return new ColtVector(colt.copy().assign(ColtUtils.vector(that), operation::apply));
    }

    @Override
    public Vector mult(double val) {
        return new ColtVector(colt.copy().assign(el -> el * val));
    }

    @Override
    public Matrix multOuter(Vector that) {
        return new ColtMatrix(Algebra.DEFAULT.multOuter(colt, ColtUtils.vector(that), new DenseDoubleMatrix2D(size(), size())));
    }

    @Override
    public double multInner(Vector that) {
        return Algebra.DEFAULT.mult(colt, ColtUtils.vector(that));
    }

    @Override
    public Vector multAsRow(Matrix that) {
        double[] values = new double[colt.size()];
        colt.toArray(values);
        DoubleMatrix2D matrix = new DenseDoubleMatrix2D(new double[][] {values});
        return new ColtVector(Algebra.DEFAULT.mult(matrix, ColtUtils.matrix(that)).viewRow(0));
    }

    @Override
    public Vector devide(Vector denominator) {
        return assign(denominator, BiOperation.DIVIDE);
    }

    @Override
    public Vector plus(Vector that) {
        return assign(that, BiOperation.SUM);
    }

    @Override
    public Vector minus(Vector that) {
        return assign(that, BiOperation.DIFF);
    }

    @Override
    public Vector minus(double value) {
        return new ColtVector(colt.copy().assign(el -> el - value));
    }

    @Override
    public Vector power(double power) {
        return assign(e -> Math.pow(e, power));
    }

    @Override
    public Vector assign(Operation operation) {
        return new ColtVector(colt.copy().assign(operation::apply));
    }

    @Override
    public double sum() {
        return colt.zSum();
    }

    @Override
    public double mean() {
        return Statistic.bin(colt).mean();
    }

    @Override
    public double min() {
        return Statistic.bin(colt).min();
    }

    @Override
    public double max() {
        return Statistic.bin(colt).max();
    }

    @Override
    public Vector abs() {
        return new ColtVector(colt.copy().assign(Operation.ABS::apply));
    }

    @Override
    public Vector enrich(EnrichFunction... functions) {
        DoubleArrayList list = new DoubleArrayList();
        double[] source = colt.toArray();
        for (EnrichFunction func : functions) {
            double[] enriched = func.enrich(source);
            for (double anEnriched : enriched) {
                list.add(anEnriched);
            }
        }
        list.trimToSize();
        return new ColtVector(new DenseDoubleMatrix1D(list.elements()));
    }

    @Override
    public Matrix covariance(Vector that) {
        int n = size();
        return (this.minus(mean()).multOuter(that.minus(that.mean()))).mult(1.0 / n);
    }

    @Override
    public Vector inverse() {
        return assign(Operation.INVERSE);
    }

    @Override
    public Vector normalize() {
        double max = max();
        double min = min();
        double[] result = new double[size()];
        double[] elements = toArray();
        for (int i = 0; i < elements.length; ++i) {
            result[i] = (elements[i] - min) / (max - min) - 0.5;
        }
        return new ColtVector(new DenseDoubleMatrix1D(result));
    }

    @Override
    public String toString() {
        return colt.toString();
    }
}
