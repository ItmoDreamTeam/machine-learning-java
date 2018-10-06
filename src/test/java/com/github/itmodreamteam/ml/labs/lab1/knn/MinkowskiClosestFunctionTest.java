package com.github.itmodreamteam.ml.labs.lab1.knn;

import com.github.itmodreamteam.ml.utils.matrixes.Vectors;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class MinkowskiClosestFunctionTest {
    @Test
    public void euclidian() {
        MinkowskiClosestFunction euclidian = new MinkowskiClosestFunction(2);
        assertEquals(Math.sqrt(2), euclidian.dist(Vectors.dense(0, 0), Vectors.dense(1, 1)), 0.0001);
    }
}