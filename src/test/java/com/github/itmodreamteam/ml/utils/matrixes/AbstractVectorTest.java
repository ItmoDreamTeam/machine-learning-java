package com.github.itmodreamteam.ml.utils.matrixes;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public abstract class AbstractVectorTest {
    private final VectorFactory vectorFactory;
    private final MatrixFactory matrixFactory;

    public AbstractVectorTest(VectorFactory vectorFactory, MatrixFactory matrixFactory) {
        this.vectorFactory = vectorFactory;
        this.matrixFactory = matrixFactory;
    }

    @Test
    public void multAsRow() {
        Vector vector = vectorFactory.create(1, 2, 3);
        Matrix matrix = matrixFactory.create(new double[][] {
                {1, 2},
                {4, 5},
                {7, 8}
        });
        Vector res = vector.multAsRow(matrix);
        // 1x3 * 3x2 = 1x2
        assertEquals(2, res.size());
    }

    @Test
    public void multInner() {
        Vector vector1 = vectorFactory.create(1, 2, 3);
        Vector vector2 = vectorFactory.create(1, 2, 3);

        double res = vector1.multInner(vector2);

        assertEquals(14.0, res, 0.0);
    }

    @Test
    public void power() {
        Vector vector = vectorFactory.create(1, 2, 3);
        Vector power = vector.power(2);

        assertEquals(3, vector.size());
        assertEquals(1, power.get(0), 0.0);
        assertEquals(4, power.get(1), 0.0);
        assertEquals(9, power.get(2), 0.0);
    }
}
