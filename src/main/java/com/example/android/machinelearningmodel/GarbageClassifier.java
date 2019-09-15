package com.example.android.machinelearningmodel;

import org.bytedeco.javacv.FrameFilter;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

public class GarbageClassifier {

    public void setModel(String path) {

        try {
            MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(path);
        }
        catch (java.io.IOException e) {
            //
        }
        catch (org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException e) {
            //
        }
        catch (org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException e) {
            //
        }
    }
}
