package com.github.itmodreamteam.ml.labs.lab2;

import com.github.itmodreamteam.ml.regression.*;
import com.github.itmodreamteam.ml.utils.ClassPathResources;
import com.github.itmodreamteam.ml.utils.CostFunctions;
import com.github.itmodreamteam.ml.utils.io.Csv;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Matrixes;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;
import com.github.itmodreamteam.ml.utils.matrixes.Vectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Lab2 {
    private static final Scanner SCANNER = new Scanner(System.in);
    private static final Logger LOGGER = LoggerFactory.getLogger(Lab2.class);

    public static void main(final String... args) throws Exception {
        Csv csv = Csv.read(ClassPathResources.getFile("prices.txt"), ",", false);
        Matrix features = Matrixes.dense(csv.doubles("area", "rooms"));
        Vector prices = Vectors.dense(csv.doubles("price"));
        LinearRegressionFactory factory1 = new GradientDescentLinearRegressionFactory(1000000, 0.1, 0.9);
        LinearRegressionFactory factory2 = new NormalEquationSolverLinearRegressionFactory();
        LinearRegressionFactory genetic1 = new GeneticLinearRegressionFactory(
                100,
                80,
                20,
                false,
                20
        );

        List<LinearRegressionFactory> regressions = new ArrayList<>();
        regressions.add(factory1);
        regressions.add(factory2);
        regressions.add(genetic1);

        for (LinearRegressionFactory factory : regressions) {
            System.out.println("regression: " + factory);
            LinearRegression regression = factory.make(features, prices);
            printStatus(regression, features, prices);
            test(regression);
        }
    }

    private static void printStatus(LinearRegression regression, Matrix features, Vector expected) {
        System.out.println(String.format("%s: mse: %s", regression, CostFunctions.computeMse(expected, regression.answer(features))));
    }

    private static void test(LinearRegression regression) {
        while (true) {
            System.out.print("Want to test? (y/n) ");
            String line = SCANNER.nextLine();
            if (line.trim().equalsIgnoreCase("y")) {
                System.out.print("Enter area: ");
                double area = Double.parseDouble(SCANNER.nextLine());
                System.out.print("Enter number of rooms: ");
                double numberOfRooms = Double.parseDouble(SCANNER.nextLine());
                Vector features = Vectors.dense(area, numberOfRooms);
                System.out.println(regression.answer(features));
            } else {
                break;
            }
        }
    }
}
