package com.github.itmodreamteam.ml.utils.matrixes;

import com.github.itmodreamteam.ml.utils.matrixes.impl.colt.ColtVector;

public class ColtVectorView implements VectorView {
    private final ColtVector vector;
    private final Operation operation;

    public ColtVectorView(ColtVector vector, Operation operation) {
        this.vector = vector;
        this.operation = operation;
    }

    @Override
    public double aggregate(AggregateFunction aggregate) {
        double accumulator = 0.0;
        for (int i = 0; i < vector.size(); ++i) {
//            if (that != null) {
//                accumulator = aggregate.reduce(accumulator, biOperation.apply(operation.apply(vector.get(i)), that.get(i)));
//            } else {
                accumulator = aggregate.reduce(accumulator, operation.apply(vector.get(i)));
//            }
        }
        return accumulator;
    }

    @Override
    public VectorView then(Operation operation) {
        return new ColtVectorView(vector, this.operation.andThen(operation));
    }

    @Override
    public Vector compute() {
        return vector.assign(operation);
    }
}
