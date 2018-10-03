package com.github.itmodreamteam.ml.utils.matrixes.impl.colt;

import com.github.itmodreamteam.ml.utils.matrixes.AbstractVectorTest;

public class ColtVectorTest extends AbstractVectorTest {
    public ColtVectorTest() {
        super(new ColtVectorFactory(), new ColtMatrixFactory());
    }
}