package com.example.android.machinelearningmodel;

import org.bytedeco.javacv.FrameFilter;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.io.Resource;

import java.io.File;

public class GarbageClassifier {

    private final String[] classes = {"metal", "paper", "plastic"};
    private final int TARGET_HEIGHT = 224;
    private final int TARGET_WIDTH = 224;
    private final String H5_FILE = "machinelearning/garbage_classification.h5";
    private MultiLayerNetwork garbageModel;

    public void setGarbageModel() throws java.io.IOException, InvalidKerasConfigurationException,
            UnsupportedKerasConfigurationException {
        setGarbageModel(this.H5_FILE);
    }

    public void setGarbageModel(String h5FilePath) throws java.io.IOException, InvalidKerasConfigurationException,
            UnsupportedKerasConfigurationException {
        ClassLoader loader = ClassLoader.getSystemClassLoader();
        String filePath = new File(loader.getResource(h5FilePath).getFile()).getPath();

        this.garbageModel = KerasModelImport.importKerasSequentialModelAndWeights(filePath, false);
    }

    public String predictClass(String imgPath) throws Exception {
        NativeImageLoader loader = new NativeImageLoader(this.TARGET_HEIGHT, this.TARGET_WIDTH, 3);
        INDArray img = loader.asMatrix(imgPath);
        return predictClass(img);
    }

    private String predictClass(INDArray img) throws Exception{
        int[] results = this.garbageModel.predict(img);
        int max = getMaxFromArray(results);
        return this.classes[max];
    }

    private static int getMaxFromArray(int[] arr) throws Exception{
        try {
            int max = arr[0];
            for (int i = 1; i < arr.length; ++i) {
                max = (max > arr[i]) ? max : arr[i];
            }
            return max;
        }
        catch (IndexOutOfBoundsException e) {
            throw new Exception("Array cannot be empty");
        }

    }

    /**
     * Quick test
     */
    public static void main(String[] args) throws java.lang.Exception {
        GarbageClassifier classifier = new GarbageClassifier();
        classifier.setGarbageModel();

        String imgRelPath = "machinelearning/test.jpg";
        ClassLoader loader = ClassLoader.getSystemClassLoader();
        String imgPath = new File(loader.getResource(imgRelPath).getFile()).getPath();

        System.out.println(classifier.predictClass(imgPath));
    }
}
