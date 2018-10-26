package com.github.itmodreamteam.ml.utils;

import java.io.File;
import java.net.URISyntaxException;

public class ClassPathResources {
    public static File getFile(String name) {
        try {
            return new File(ClassPathResources.class.getClassLoader().getResource(name).toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException("cannot read classpath resource " + name, e);
        }
    }
}
