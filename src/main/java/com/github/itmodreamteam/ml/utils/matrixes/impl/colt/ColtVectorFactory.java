package com.github.itmodreamteam.ml.utils.matrixes.impl.colt;

import com.github.itmodreamteam.ml.utils.matrixes.Vector;
import com.github.itmodreamteam.ml.utils.matrixes.VectorFactory;

public class ColtVectorFactory implements VectorFactory {
    @Override
    public Vector create(double... vector) {
        return ColtUtils.vector(vector);
    }
}
