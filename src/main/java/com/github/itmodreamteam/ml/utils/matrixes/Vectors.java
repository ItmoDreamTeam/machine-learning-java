package com.github.itmodreamteam.ml.utils.matrixes;

import com.github.itmodreamteam.ml.utils.matrixes.impl.colt.ColtVectorFactory;

import java.util.stream.DoubleStream;

public class Vectors {
    private static final VectorFactory factory = new ColtVectorFactory();

    public static Vector dense(double... elements) {
        return factory.create(elements);
    }

    public static Vector zeros(int size) {
        return factory.create(new double[size]);
    }

    public static Vector ones(int size) {
        return factory.create(DoubleStream.iterate(1.0, d -> d).limit(size).toArray());
    }
}
