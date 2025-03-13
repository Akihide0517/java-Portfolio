import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {
	public static void main(String[] args) {
        // ハイパーパラメータ設定 
        int numFeatures = 3;
        int numHiddenLayers = 2;
        int[] hiddenLayerSizes = {2, 2, 2};
        int numOutputNodes = 3;
        double learningRate = 0.1; // 学習率を調整
        int epochs = 2800;
        int batchSize = 4;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;
        double l2Regularization = 0.01;
        boolean useSMOTE = true; // SMOTEは一旦false

        // 活性化関数 
        ActivationFunction relu = new ReLU();
        OutputActivationFunction softmax = new Softmax();

        // MLPインスタンス生成 
        MultilayerPerceptron mlp = new MultilayerPerceptron(numFeatures, numHiddenLayers, hiddenLayerSizes, numOutputNodes, relu, softmax, learningRate, beta1, beta2, epsilon, l2Regularization);


        // 学習データ (ORゲート) 
        List<double[]> inputData = new ArrayList<>();
        List<double[]> targetData = new ArrayList<>();
        inputData.add(new double[]{0, 0, 0}); targetData.add(new double[]{1, 0, 0}); // クラス0 (0)
        inputData.add(new double[]{0, 1, 0}); targetData.add(new double[]{0, 1, 0}); // クラス1 (1)
        inputData.add(new double[]{1, 0, 0}); targetData.add(new double[]{0, 1, 0}); // クラス1 (1)
        inputData.add(new double[]{0, 1, 1}); targetData.add(new double[]{0, 0, 1}); // クラス1 (2)

        // 学習実行 
        mlp.train(inputData, targetData, epochs, batchSize, useSMOTE);

        // テストデータ 
        List<double[]> testInputData = new ArrayList<>();
        testInputData.add(new double[]{0, 0, 0});
        testInputData.add(new double[]{0, 1, 0});
        testInputData.add(new double[]{1, 0, 1});
        testInputData.add(new double[]{0, 1, 1});

        // 予測と評価 
        System.out.println("\nPredictions:");
        for (double[] input : testInputData) {
            double[] prediction = mlp.predict(input);
            System.out.println("Input: " + Arrays.toString(input) + ", Prediction: " + Arrays.toString(prediction) + ", Predicted Class: " + mlp.getPredictedClass(prediction));
        }

        // 誤差評価 
        double crossEntropyError = mlp.calculateCrossEntropyError(testInputData, targetData);
        System.out.println("\nFinal Cross-Entropy Error on Test Data: " + crossEntropyError);
    }
}
