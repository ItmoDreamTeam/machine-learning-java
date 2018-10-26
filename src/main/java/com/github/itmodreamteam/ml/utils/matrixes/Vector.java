package com.github.itmodreamteam.ml.utils.matrixes;

public interface Vector {
    int size();

    double get(int index);

    double[] toArray();

    Vector assign(Vector that, BiOperation operation);

    Vector mult(double val);

    Matrix multOuter(Vector that);

    double multInner(Vector that);

    Vector multAsRow(Matrix that);

    Vector devide(Vector denominator);

    Vector plus(Vector that);

    Vector minus(Vector that);

    Vector minus(double value);

    Vector power(double power);

    Vector assign(Operation operation);

    double sum();

    double mean();

    double min();

    double max();

    Vector abs();

    Vector enrich(EnrichFunction... func);

    Matrix covariance(Vector that);

    Vector inverse();

    Vector normalize();
}
