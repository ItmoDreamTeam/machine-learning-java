package com.github.itmodreamteam.ml.utils.matrixes;

public interface Vector {
    int size();

    double get(int index);

    double[] toArray();

    Vector assign(Vector that, BiOperation operation);

    Matrix multOuter(Vector that);

    double multInner(Vector that);

    Vector multAsRow(Matrix that);

    Vector plus(Vector that);

    Vector minus(Vector that);

    Vector minus(double value);

    Vector power(double power);

    Vector assign(Operation operation);

    double sum();

    double mean();

    Vector abs();

    Vector enrich(EnrichFunction... func);

    Matrix covariance(Vector that);

    Vector inverse();
}
