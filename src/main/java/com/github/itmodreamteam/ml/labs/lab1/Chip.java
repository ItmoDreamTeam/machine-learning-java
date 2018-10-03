package com.github.itmodreamteam.ml.labs.lab1;

public class Chip {
    private final double x;
    private final double y;
    private final int label;

    public Chip(double x, double y, int label) {
        this.x = x;
        this.y = y;
        this.label = label;
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    public int getLabel() {
        return label;
    }
}
