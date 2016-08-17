/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multilayerperceptron.weka;

import Funciones.MyListArgs;
import Funciones.MySintaxis;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.pmml.jaxbbindings.Output;

/**
 *
 * @author miguel
 */
public class MultiLayerPerceptronWEKA {

     String MODEL_PATH    = "/Volumes/HDD/pendiente/MultiLayerPerceptron-WEKA/data/model2.model";
     String TEST_OUT_PATH = "/Volumes/HDD/pendiente/MultiLayerPerceptron-WEKA/data/predict2.txt";
     String TEST_FILE     = "";
     String TRAIN_FILE    = "";
     int    hiddenLayers  = 7;
     double learningRate  = 0.01;
     int    CLASS_INDEX   = 0;
     int    TRAINING_TIME = 5000;
     
     MultiLayerPerceptronWEKA(String[]args)
     {
        MyListArgs         Param;
        String        ConfigFile;
        String                IN;
        String               OUT;
        
        BufferedImage zoomedBI = null;

        Param      = new MyListArgs(args)                  ;
        ConfigFile = Param.ValueArgsAsString("-CONFIG", "");

        if (!ConfigFile.equals(""))
        {
            Param.AddArgsFromFile(ConfigFile);
        }//fin if

        String Sintaxis    = "((-TRAIN_FILE:str [-HIDDEN_LAYERS:int][-CLASS_INDEX:int][-TRAINING_TIME:int][-LEARNING_RATE:float])|(-TEST_FILE:str -PREDICTION_PATH:str))";
        MySintaxis Review  = new MySintaxis(Sintaxis, Param);

        //code for train mode
        this.TRAIN_FILE    = Param.ValueArgsAsString ("-TRAIN_FILE"      ,   "");
        this.hiddenLayers  = Param.ValueArgsAsInteger("-HIDDEN_LAYERS"   ,   10);
        this.CLASS_INDEX   = Param.ValueArgsAsInteger("-CLASS_INDEX"     ,    0);
        this.TRAINING_TIME = Param.ValueArgsAsInteger("-TRAINING_TIME"   , 1000);
        this.learningRate  = Param.ValueArgsAsFloat  ("-LEARNING_RATE"   , 0.1f);
        //code for test mode
        this.TEST_FILE     = Param.ValueArgsAsString ("-TEST_FILE"       ,   "");
        this.TEST_OUT_PATH = Param.ValueArgsAsString ("-PREDICTION_PATH" ,   "");
     }
     
     public void process()
     {
         if(!this.TRAIN_FILE.equals(""))
         {
             //train mode
             this.simpleWekaTrain(this.TRAIN_FILE);             
         }
         else
         {
             //test mode
             this.SimpleWekaTest(this.TEST_FILE);         
         }
     }

    public  void simpleWekaTrain(String filepath)
    {
        FileReader trainreader;
        Instances train = null;
        MultilayerPerceptron mlp = null;
        try{
            //Reading training arff file
            trainreader = new FileReader(filepath);
            train = new Instances(trainreader);
            train.setClassIndex(train.numAttributes() - 1);
            train.setClassIndex(CLASS_INDEX);

            //Instance of NN
            mlp = new MultilayerPerceptron();

            //Setting Parameters
            mlp.setLearningRate(learningRate);
            mlp.setMomentum(0.01);
            mlp.setTrainingTime(TRAINING_TIME);
            mlp.setHiddenLayers(String.valueOf(hiddenLayers));

            mlp.buildClassifier(train);
            weka.core.SerializationHelper.write(MODEL_PATH, mlp);
        }
        catch(Exception ex){
            ex.printStackTrace();
        }
        Evaluation eval = null;
        try {
            eval = new Evaluation(train);
        } catch (Exception ex) {
            Logger.getLogger(MultiLayerPerceptronWEKA.class.getName()).log(Level.SEVERE, null, ex);
        }
        try{
            eval.evaluateModel(mlp, train);
        }catch(Exception e){}

        System.out.println(eval.errorRate()); //Printing Training Mean root squared Error
        System.out.println(eval.toSummaryString()); //Summary of Training
    }
    
    public  void SimpleWekaTest(String filePath)
    {
        
        MultilayerPerceptron mlp = null;
        
        try{
            mlp = (MultilayerPerceptron) weka.core.SerializationHelper.read(MODEL_PATH);
        }catch(Exception e){}
        
        Instances datapredict = null;
        try {
            datapredict = new Instances(
                    new BufferedReader(
                            new FileReader(filePath)));
        } catch (IOException ex) {}
        
        datapredict.setClassIndex(datapredict.numAttributes() - 1);
        datapredict.setClassIndex(CLASS_INDEX);
        Instances predicteddata = new Instances(datapredict);
        
        //Predict each value
        for (int i = 0; i < datapredict.numInstances(); i++) {
            try{
                double clsLabel = mlp.classifyInstance(datapredict.instance(i));
                predicteddata.instance(i).setClassValue(clsLabel);
                System.out.println(predicteddata.instance(i));
            } catch(Exception e){}       
        }
        
        //Storing again in arff
        BufferedWriter writer = null;
        
        try {
            writer = new BufferedWriter(
                    new FileWriter(TEST_OUT_PATH));
            
            writer.write(predicteddata.toString());
            writer.newLine();
            writer.flush();
            writer.close();
        } catch (IOException ex) {}
        
    }
}
