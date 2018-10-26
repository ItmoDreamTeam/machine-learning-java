package com.github.itmodreamteam.ml.utils.matrixes;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public abstract class AbstractMatrixTest {
    private static final double DELTA = 1e-9;
    private final VectorFactory vectorFactory;
    private final MatrixFactory matrixFactory;

    public AbstractMatrixTest(VectorFactory vectorFactory, MatrixFactory matrixFactory) {
        this.vectorFactory = vectorFactory;
        this.matrixFactory = matrixFactory;
    }

    @Test
    public void multColumn() {
        Matrix matrix = matrixFactory.create(new double[][]{
                {1, 2},
                {3, 4},
                {5, 6}
        });
        Vector column = vectorFactory.create(1, 2);
        Vector result = matrix.multColumn(column);

        assertEquals(3, result.size());
        assertEquals(5, (int) result.get(0));
        assertEquals(11, (int) result.get(1));
        assertEquals(17, (int) result.get(2));
    }

    @Test
    public void inverseTest() {
        Matrix matrix = matrixFactory.create(new double[][]{
                {-1, 1.5},
                {1, -1}
        });
        Matrix inverted = matrix.inverse();
        assertEquals(2, inverted.get(0, 0), DELTA);
        assertEquals(3, inverted.get(0, 1), DELTA);
        assertEquals(2, inverted.get(1, 0), DELTA);
        assertEquals(2, inverted.get(1, 1), DELTA);
    }
}
